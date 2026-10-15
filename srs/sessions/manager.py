from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional

from smolagents.memory import ActionStep, PlanningStep

from ..config.config_loader import (
    CONFIG,
    get_available_models,
    get_default_model,
    get_experts_config,
    get_memory_config,
    get_sessions_config,
    resolve_model_entry,
)
from ..llm.agent_builder import build_agent
from ..llm.model_factory import create_model
from ..llm.smol_engine import SmolEngine
from ..llm.tool_registry import get_tool, list_tool_names
from ..memory.compression import MemoryCompressor
from ..memory.plan import UpdatePlanTool, default_plan, merge_plan, utc_now
from ..memory.prompt import build_instructions
from .store import Session, SessionStore


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if hasattr(value, "dict") and callable(value.dict):
        return _jsonable(value.dict())
    if is_dataclass(value):
        return _jsonable(asdict(value))
    return str(value)


class SessionManager:
    """Coordinate a main engine with persisted session files."""

    def __init__(
        self,
        store: Optional[SessionStore] = None,
        model_label: Optional[str] = None,
        compress_on_switch: Optional[bool] = None,
        compress_on_exit: Optional[bool] = None,
    ) -> None:
        self.store = store or SessionStore()
        self.model_label = model_label or get_default_model(CONFIG)
        if not self.model_label:
            available = get_available_models(CONFIG)
            if not available:
                raise RuntimeError("No models are configured.")
            self.model_label = available[0]["label"]

        sessions_config = get_sessions_config(CONFIG)
        self.compress_on_switch = (
            sessions_config.get("compress_on_switch", True)
            if compress_on_switch is None
            else compress_on_switch
        )
        self.compress_on_exit = (
            sessions_config.get("compress_on_exit", True)
            if compress_on_exit is None
            else compress_on_exit
        )
        self.auto_save_steps = sessions_config.get("auto_save_steps", True)

        self.current: Optional[Session] = None
        self.engine: Optional[Any] = None
        self._compressor: Optional[MemoryCompressor] = None
        self.last_warning = ""

    @property
    def session(self) -> Optional[Session]:
        return self.current

    def open(
        self,
        identifier: Optional[str] = None,
        create_if_missing: bool = True,
    ) -> Session:
        if self.current is not None:
            self.save()

        session = (
            self.store.get(identifier)
            if identifier
            else self.store.latest()
        )
        if session is None:
            if not create_if_missing:
                raise KeyError("No sessions exist.")
            session = self.store.create(model_label=self.model_label)

        self.current = session
        self._activate_session()
        return session

    def new(self, name: Optional[str] = None) -> Session:
        if self.current is not None:
            leave = self._prepare_to_leave(self.compress_on_switch)
            self.last_warning = leave["warning"]
        self.current = self.store.create(
            name=name,
            model_label=self.model_label,
        )
        self._activate_session()
        return self.current

    def _prepare_to_leave(self, compress: bool) -> Dict[str, Any]:
        warning = ""
        compressed = False
        if self.current is not None:
            if compress and self.current.read_steps():
                try:
                    self.compress()
                    compressed = True
                except Exception as error:
                    warning = str(error)
            self.save()
        return {"compressed": compressed, "warning": warning}

    def _activate_session(self) -> None:
        if self.current is None:
            raise RuntimeError("No session is open.")

        expert_names = [expert["role"] for expert in get_experts_config(CONFIG)]
        self.engine = build_agent(
            main_model_label=self.model_label,
            expert_names=expert_names,
        )
        if isinstance(self.engine, SmolEngine):
            if self.auto_save_steps:
                self.engine.register_step_callback(self._on_step)
            self._attach_saved_tools()
            self.engine.register_tools([UpdatePlanTool(self.current)])
            self.engine.set_instructions(
                build_instructions(self.current, warn_uncompressed=True)
            )
            self.engine.reset_memory()

    def _attach_saved_tools(self) -> None:
        if self.current is None or not self.current.active_tools:
            return

        tools = []
        loaded = []
        for name in self.current.active_tools:
            try:
                tools.append(get_tool(name))
                loaded.append(name)
            except KeyError as error:
                print(f"Warning: {error}")

        if loaded != self.current.active_tools:
            self.current.active_tools = loaded
            self.current.save_meta()
        if tools:
            self.engine.register_tools(tools)

    def list_tools(self) -> Dict[str, Any]:
        active = list(self.current.active_tools) if self.current else []
        return {"registered": list_tool_names(), "active": active}

    def add_tool(self, name: str) -> None:
        if self.current is None:
            raise RuntimeError("No session is open.")
        if name not in list_tool_names():
            raise KeyError(f"Tool '{name}' is not registered.")
        if name in self.current.active_tools:
            return

        tool = get_tool(name)
        if isinstance(self.engine, SmolEngine):
            self.engine.register_tools([tool])
        self.current.active_tools = [*self.current.active_tools, name]
        self.current.save_meta()

    def remove_tool(self, name: str) -> None:
        if self.current is None:
            raise RuntimeError("No session is open.")
        if name not in self.current.active_tools:
            return

        if isinstance(self.engine, SmolEngine):
            try:
                tool_name = get_tool(name).name
            except KeyError:
                tool_name = name
            self.engine.unregister_tools([tool_name])
        self.current.active_tools = [
            item for item in self.current.active_tools if item != name
        ]
        self.current.save_meta()

    def run(self, task: str) -> str:
        if self.current is None:
            self.open()
        if self.engine is None:
            self._activate_session()

        task = task.strip()
        if not task:
            return ""

        session = self.current
        assert session is not None

        plan = session.load_plan()
        if not plan.get("goal"):
            plan["goal"] = task
        plan = merge_plan(
            plan,
            status="in_progress",
            last_error="",
        )
        session.save_plan(plan)
        session.append_step(
            {"kind": "TaskStep", "task": task, "timestamp": utc_now()}
        )
        if hasattr(self.engine, "set_instructions"):
            self.engine.set_instructions(build_instructions(session))

        try:
            result = self.engine.run(task)
        except Exception as error:
            plan = merge_plan(
                session.load_plan(),
                status="error",
                last_error=str(error),
            )
            session.save_plan(plan)
            session.append_step(
                {
                    "kind": "ErrorStep",
                    "error": str(error),
                    "timestamp": utc_now(),
                }
            )
            session.save_meta()
            raise

        plan = merge_plan(
            session.load_plan(),
            status="idle",
            last_error="",
        )
        session.save_plan(plan)
        session.append_step(
            {
                "kind": "FinalAnswerStep",
                "output": result,
                "timestamp": utc_now(),
            }
        )
        session.save_meta()
        return result

    def _on_step(self, step: Any) -> None:
        if self.current is None:
            return
        try:
            record = self._step_record(step)
            self.current.append_step(record)
            plan = self.current.load_plan()
            plan["current_step"] = int(plan.get("current_step", 0)) + 1
            if record.get("error"):
                plan["status"] = "error"
                plan["last_error"] = record["error"]
            else:
                plan["status"] = "in_progress"
            self.current.save_plan(merge_plan(plan))
            self.current.save_meta()
        except Exception as error:
            self.last_warning = f"Could not persist step: {error}"

    @staticmethod
    def _step_record(step: Any) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "kind": type(step).__name__,
            "timestamp": utc_now(),
        }
        step_number = getattr(step, "step_number", None)
        if step_number is not None:
            record["step"] = step_number

        if isinstance(step, PlanningStep):
            record["plan"] = getattr(step, "plan", "")
        elif isinstance(step, ActionStep):
            record["model_output"] = getattr(step, "model_output", "")
            record["observations"] = getattr(step, "observations", "")
            record["error"] = (
                str(step.error) if getattr(step, "error", None) else None
            )
            record["tool_calls"] = _jsonable(
                getattr(step, "tool_calls", None) or []
            )
        else:
            record["output"] = getattr(step, "output", "")

        token_usage = getattr(step, "token_usage", None)
        if token_usage is not None:
            record["token_usage"] = _jsonable(token_usage)
        return record

    def _get_compressor(self) -> MemoryCompressor:
        if self._compressor is not None:
            return self._compressor

        memory_config = get_memory_config(CONFIG)
        compression = memory_config.get("compression", {})
        label = compression.get("model") or self.model_label
        entry = resolve_model_entry(CONFIG, label)
        model = create_model(entry["provider"], entry["model"])
        self._compressor = MemoryCompressor(
            model,
            max_summary_tokens=int(
                compression.get("max_summary_tokens", 1200)
            ),
        )
        return self._compressor

    def compress(self) -> str:
        if self.current is None:
            raise RuntimeError("No session is open.")
        if not self.current.read_steps():
            return self.current.load_memory()

        summary = self._get_compressor().compress(self.current)
        if isinstance(self.engine, SmolEngine):
            self.engine.recreate_agent()
        if self.engine is not None and hasattr(self.engine, "set_instructions"):
            self.engine.set_instructions(build_instructions(self.current))
        self.current.save_meta()
        return summary

    def switch(self, identifier: str) -> Dict[str, Any]:
        leave = self._prepare_to_leave(self.compress_on_switch)

        session = self.store.get(identifier)
        self.current = session
        self._activate_session()
        self.last_warning = leave["warning"]
        return {
            "session": session,
            "compressed": leave["compressed"],
            "warning": leave["warning"],
        }

    def reset(self, hard: bool = False) -> None:
        if self.current is None:
            raise RuntimeError("No session is open.")

        if hard:
            self.current.save_memory("", {"kind": "reset-hard"})
        self.current.clear_current_run()
        self.current.save_plan(default_plan())
        if isinstance(self.engine, SmolEngine):
            self.engine.reset_memory()
        if self.engine is not None and hasattr(self.engine, "set_instructions"):
            self.engine.set_instructions(build_instructions(self.current))
        self.current.save_meta()

    def save(self) -> None:
        if self.current is not None:
            self.current.save_meta()

    def close(self, compress: Optional[bool] = None) -> None:
        warning = ""
        should_compress = self.compress_on_exit if compress is None else compress
        if (
            self.current is not None
            and should_compress
            and self.current.read_steps()
        ):
            try:
                self.compress()
            except Exception as error:
                warning = str(error)
        self.save()
        self.last_warning = warning
        self.engine = None
        self.current = None

    def rename(self, name: str) -> None:
        if self.current is None:
            raise RuntimeError("No session is open.")
        name = name.strip()
        if not name:
            raise ValueError("Session name cannot be empty.")
        self.current.name = name
        self.current.save_meta()

    def delete(self, identifier: str) -> None:
        session = self.store.get(identifier)
        if self.current is not None and session.session_id == self.current.session_id:
            raise ValueError("Cannot delete the active session.")
        self.store.delete(identifier)

    def list_sessions(self) -> List[Dict[str, Any]]:
        return [session.summary() for session in self.store.list()]
