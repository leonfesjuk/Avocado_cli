import json
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional

from smolagents.memory import ActionStep, PlanningStep

from ..config.config_loader import (
    CONFIG,
    get_available_models,
    get_compression_prompt,
    get_default_model,
    get_experts_config,
    get_memory_config,
    get_model_capabilities,
    get_sessions_config,
    resolve_model_entry,
)
from ..llm.agent_builder import build_agent
from ..llm.chat_engine import ChatEngine
from ..llm.expert_tools import expert_tool_name, last_expert_calls
from ..llm.model_factory import create_model
from ..llm.smol_engine import SmolEngine
from ..llm.tool_registry import (
    get_tool,
    list_builtin_tool_names,
    list_tool_names,
)
from ..memory.compression import MemoryCompressor
from ..memory.plan import UpdatePlanTool, default_plan, merge_plan, utc_now
from ..memory.prompt import build_instructions
from ..memory.token_counter import estimate_tokens
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
        self._explicit_model = model_label is not None
        self.model_label = model_label or get_default_model(CONFIG)
        if not self.model_label:
            available = get_available_models(CONFIG)
            if not available:
                raise RuntimeError("No models are configured.")
            self.model_label = available[0]["label"]

        compression_config = get_memory_config(CONFIG).get("compression", {})
        self._default_compression_model = compression_config.get("model") or ""
        self.compression_model_label = self._default_compression_model

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
        self.last_compression: Dict[str, Any] = {}
        self.last_expert_calls: List[Dict[str, Any]] = []

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
        elif self._explicit_model:
            session.model = self.model_label
            session.save_meta()
        elif session.model:
            self.model_label = session.model
        else:
            session.model = self.model_label
            session.save_meta()

        self._explicit_model = False
        self.current = session
        self._sync_compression_model(session)
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
        if self.compression_model_label:
            self.current.compression_model = self.compression_model_label
            self.current.save_meta()
        self._activate_session()
        return self.current

    def _prepare_to_leave(self, compress: bool) -> Dict[str, Any]:
        warning = ""
        compressed = False
        compression: Dict[str, Any] = {}
        if self.current is not None:
            if compress and self.current.read_steps():
                try:
                    self.compress()
                    compressed = True
                    compression = dict(self.last_compression)
                except Exception as error:
                    warning = str(error)
            self.save()
        return {
            "compressed": compressed,
            "compression": compression,
            "warning": warning,
        }

    def _activate_session(self) -> None:
        if self.current is None:
            raise RuntimeError("No session is open.")
        self.last_compression = {}

        configured = get_experts_config(CONFIG)
        expert_names = self._active_expert_roles(configured)
        if self.current.active_experts != expert_names:
            self.current.active_experts = expert_names
            self.current.save_meta()

        has_tools = bool(self.current.active_tools or expert_names)
        self.engine = build_agent(
            main_model_label=self.model_label,
            expert_names=expert_names,
            expert_overrides=self.current.expert_overrides,
            tools_enabled=True if has_tools else None,
        )
        if isinstance(self.engine, SmolEngine):
            if self.auto_save_steps:
                self.engine.register_step_callback(self._on_step)
            self._attach_saved_tools()
            self.engine.register_tools([UpdatePlanTool(self.current)])
        if hasattr(self.engine, "set_instructions"):
            self.engine.set_instructions(
                self._build_instructions(self.current, warn_uncompressed=True)
            )
        if hasattr(self.engine, "reset_memory"):
            self.engine.reset_memory()

    def _sync_compression_model(self, session: Optional[Session]) -> None:
        if session is not None and session.compression_model:
            self.compression_model_label = session.compression_model
        else:
            self.compression_model_label = self._default_compression_model
        self._compressor = None

    def _build_instructions(
        self, session: Session, warn_uncompressed: bool = False
    ) -> str:
        return build_instructions(
            session,
            warn_uncompressed=warn_uncompressed,
            experts=self._active_expert_roles(),
        )

    def _active_expert_roles(
        self,
        configured: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        if configured is None:
            configured = get_experts_config(CONFIG)
        roles = [expert["role"] for expert in configured]
        if self.current is None or self.current.active_experts is None:
            return []
        return list(
            dict.fromkeys(
                role
                for role in self.current.active_experts
                if role in roles
            )
        )

    def _attach_saved_tools(self) -> None:
        if self.current is None or not self.current.active_tools:
            return

        attached = {tool.name for tool in self.engine.tools}
        tools = []
        loaded = []
        for name in self.current.active_tools:
            try:
                tool = get_tool(name)
            except KeyError as error:
                print(f"Warning: {error}")
                continue
            if tool.name in attached:
                print(f"Warning: tool '{name}' conflicts with an attached tool.")
                continue
            attached.add(tool.name)
            tools.append(tool)
            loaded.append(name)

        if loaded != self.current.active_tools:
            self.current.active_tools = loaded
            self.current.save_meta()
        if tools:
            self.engine.register_tools(tools)

    def list_tools(self) -> Dict[str, Any]:
        active = list(self.current.active_tools) if self.current else []
        return {
            "registered": list_tool_names(),
            "builtin": list_builtin_tool_names(),
            "active": active,
        }

    def _model_supports_tools(self, model_label: Optional[str] = None) -> bool:
        entry = resolve_model_entry(CONFIG, model_label or self.model_label)
        capabilities = get_model_capabilities(entry["model"])
        return bool(
            capabilities["supports_code_loop"]
            or capabilities["supports_tool_calling"]
        )

    def _ensure_tool_capable(self, model_label: Optional[str] = None) -> None:
        label = model_label or self.model_label
        if not self._model_supports_tools(label):
            raise ValueError(
                f"Model '{label}' is chat-only and cannot use tools or experts. "
                "Switch to a tool-capable model first."
            )

    def _expert_tool_names(self) -> set:
        return {
            expert_tool_name(role)
            for role in self._active_expert_roles()
        }

    def add_tool(self, name: str) -> None:
        if self.current is None:
            raise RuntimeError("No session is open.")
        if name not in list_tool_names():
            raise KeyError(f"Tool '{name}' is not registered.")
        if name in self.current.active_tools:
            return

        tool = get_tool(name)
        if tool.name in self._expert_tool_names():
            raise ValueError(
                f"Tool '{tool.name}' conflicts with an active expert tool."
            )
        self._ensure_tool_capable()

        self.current.active_tools = [*self.current.active_tools, name]
        self.current.save_meta()
        self._activate_session()

    def add_all_tools(self) -> Dict[str, List[str]]:
        if self.current is None:
            raise RuntimeError("No session is open.")

        added: List[str] = []
        skipped: List[str] = []
        expert_names = self._expert_tool_names()
        active = set(self.current.active_tools)
        for name in list_builtin_tool_names():
            if name in active:
                skipped.append(name)
                continue
            tool = get_tool(name)
            if tool.name in expert_names:
                skipped.append(name)
                continue
            added.append(name)

        if not added:
            return {"added": [], "skipped": skipped}

        self._ensure_tool_capable()
        self.current.active_tools = [
            *self.current.active_tools,
            *added,
        ]
        self.current.save_meta()
        self._activate_session()
        return {"added": added, "skipped": skipped}

    def remove_tool(self, name: str) -> None:
        if self.current is None:
            raise RuntimeError("No session is open.")
        if name not in self.current.active_tools:
            return

        self.current.active_tools = [
            item for item in self.current.active_tools if item != name
        ]
        self.current.save_meta()
        self._activate_session()

    def clear_tools(self) -> None:
        if self.current is None:
            raise RuntimeError("No session is open.")
        if not self.current.active_tools:
            return
        self.current.active_tools = []
        self.current.save_meta()
        self._activate_session()

    def list_experts(self) -> List[Dict[str, Any]]:
        if self.current is None:
            return []

        configured = get_experts_config(CONFIG)
        active = set(self._active_expert_roles(configured))
        experts = []
        for expert in configured:
            role = expert["role"]
            override = self.current.expert_overrides.get(role, {})
            experts.append(
                {
                    "role": role,
                    "name": expert.get("name") or role,
                    "description": expert.get("description", ""),
                    "expertise": expert.get("expertise", ""),
                    "model": expert.get("model") or self.model_label,
                    "active": role in active,
                    "allow_tools": override.get(
                        "allow_tools", expert.get("allow_tools", False)
                    ),
                    "tools": list(
                        override.get("tools", expert.get("tools", []))
                    ),
                    "configured_tools": list(expert.get("tools", [])),
                }
            )
        return experts

    def enable_expert(self, role: str) -> None:
        if self.current is None:
            raise RuntimeError("No session is open.")

        configured = {
            expert["role"]: expert for expert in get_experts_config(CONFIG)
        }
        if role not in configured:
            raise KeyError(f"Expert '{role}' is not configured.")

        active = self._active_expert_roles()
        if role in active:
            return

        self._ensure_tool_capable()
        self.current.active_experts = [*active, role]
        self.current.save_meta()
        self._activate_session()

    def disable_expert(self, role: str) -> None:
        if self.current is None:
            raise RuntimeError("No session is open.")

        active = self._active_expert_roles()
        if role not in active:
            return

        self.current.active_experts = [item for item in active if item != role]
        self.current.save_meta()
        self._activate_session()

    def enable_all_experts(self) -> List[str]:
        if self.current is None:
            raise RuntimeError("No session is open.")

        roles = [expert["role"] for expert in get_experts_config(CONFIG)]
        if not roles:
            return []
        self._ensure_tool_capable()
        self.current.active_experts = roles
        self.current.save_meta()
        self._activate_session()
        return roles

    def disable_all_experts(self) -> None:
        if self.current is None:
            raise RuntimeError("No session is open.")
        if not self.current.active_experts:
            return
        self.current.active_experts = []
        self.current.save_meta()
        self._activate_session()

    def set_expert_tools(self, role: str, names: List[str]) -> None:
        if self.current is None:
            raise RuntimeError("No session is open.")

        configured = {
            expert["role"]: expert for expert in get_experts_config(CONFIG)
        }
        if role not in configured:
            raise KeyError(f"Expert '{role}' is not configured.")

        names = list(dict.fromkeys(name for name in names if name))
        unknown = sorted(set(names) - set(list_tool_names()))
        if unknown:
            raise KeyError(
                f"Unknown tool(s) for expert '{role}': " + ", ".join(unknown)
            )
        if names:
            self._ensure_tool_capable(
                configured[role].get("model") or self.model_label
            )

        override = self.current.expert_overrides.setdefault(role, {})
        override["allow_tools"] = bool(names)
        override["tools"] = list(names)
        self.current.save_meta()
        if role in self._active_expert_roles():
            self._activate_session()

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
            self.engine.set_instructions(self._build_instructions(session))

        last_expert_calls.clear()
        try:
            result = self.engine.run(task)
        except Exception as error:
            self.last_expert_calls = list(last_expert_calls)
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

        self.last_expert_calls = list(last_expert_calls)
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
        label = self.compression_model_label or self.model_label
        entry = resolve_model_entry(CONFIG, label)
        model = create_model(entry["provider"], entry["model"])
        prompt = get_compression_prompt(CONFIG)
        compressor_kwargs = {}
        if prompt:
            compressor_kwargs["prompt_template"] = prompt
        self._compressor = MemoryCompressor(
            model,
            max_summary_tokens=int(
                compression.get("max_summary_tokens", 1200)
            ),
            max_transcript_tokens=int(
                compression.get("max_transcript_tokens", 6000)
            ),
            **compressor_kwargs,
        )
        return self._compressor

    def compress(self) -> str:
        if self.current is None:
            raise RuntimeError("No session is open.")
        steps = self.current.read_steps()
        if not steps:
            self.last_compression = {}
            return self.current.load_memory()

        before = self.context_usage()
        summary = self._get_compressor().compress(self.current)
        if isinstance(self.engine, SmolEngine):
            self.engine.recreate_agent()
        if self.engine is not None and hasattr(self.engine, "set_instructions"):
            self.engine.set_instructions(self._build_instructions(self.current))
        after = self.context_usage()

        meta = self.current.load_memory_meta()
        delta_tokens = before["used"] - after["used"]
        meta.update(
            {
                "context_before_tokens": before["used"],
                "context_after_tokens": after["used"],
                "context_delta_tokens": delta_tokens,
                "context_delta_percent": (
                    round(delta_tokens / before["used"] * 100, 1)
                    if before["used"]
                    else 0.0
                ),
                "context_limit": after["limit"],
                "context_model": after["model"],
                "context_memory_before": before["memory_tokens"],
                "context_memory_after": after["memory_tokens"],
                "context_history_before": before["history_tokens"],
                "context_history_after": after["history_tokens"],
                "context_raw_before": before["raw_history_tokens"],
                "context_raw_after": after["raw_history_tokens"],
            }
        )
        self.current.save_memory(self.current.load_memory(), meta)
        self.current.save_meta()

        self.last_compression = {
            "context_before": before,
            "context_after": after,
            "context_delta_tokens": delta_tokens,
            "context_delta_percent": (
                round(delta_tokens / before["used"] * 100, 1)
                if before["used"]
                else 0.0
            ),
            "summary_tokens": meta.get("estimated_tokens", 0),
            "source_tokens": meta.get("source_tokens", 0),
            "raw_tokens": meta.get("raw_tokens", 0),
            "journal_saved_tokens": meta.get("saved_tokens", 0),
            "journal_saved_percent": meta.get("saved_percent", 0.0),
        }
        return summary

    def switch(self, identifier: str) -> Dict[str, Any]:
        leave = self._prepare_to_leave(self.compress_on_switch)

        session = self.store.get(identifier)
        if session.model:
            self.model_label = session.model
        else:
            session.model = self.model_label
            session.save_meta()
        self.current = session
        self._sync_compression_model(session)
        self._activate_session()
        self.last_warning = leave["warning"]
        return {
            "session": session,
            "compressed": leave["compressed"],
            "compression": leave.get("compression", {}),
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
            self.engine.set_instructions(self._build_instructions(self.current))
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

    def list_models(self) -> List[Dict[str, Any]]:
        models = []
        for entry in get_available_models(CONFIG):
            capabilities = get_model_capabilities(entry["model"])
            chat_only = not (
                capabilities["supports_code_loop"]
                or capabilities["supports_tool_calling"]
            )
            models.append(
                {
                    "label": entry["label"],
                    "current": entry["label"].lower()
                    == self.model_label.lower(),
                    "chat_only": chat_only,
                    "supports_code_loop": capabilities["supports_code_loop"],
                    "supports_tool_calling": capabilities[
                        "supports_tool_calling"
                    ],
                }
            )
        return models

    def set_model(self, label: str) -> None:
        entry = resolve_model_entry(CONFIG, label)
        active_experts = self._active_expert_roles()
        active_tools = self.current.active_tools if self.current else []
        if active_tools or active_experts:
            self._ensure_tool_capable(entry["label"])
        self.model_label = entry["label"]
        if self.current is not None:
            self.current.model = self.model_label
            self.current.save_meta()
            self._activate_session()

    def list_compression_models(self) -> List[Dict[str, Any]]:
        current = (self.compression_model_label or self.model_label).lower()
        models = self.list_models()
        for item in models:
            item["current"] = item["label"].lower() == current
        return models

    def set_compression_model(self, label: str) -> str:
        if not label or label.strip().lower() in {"default", "auto", "session"}:
            self.compression_model_label = self._default_compression_model
            self._compressor = None
            if self.current is not None:
                self.current.compression_model = ""
                self.current.save_meta()
            return self.compression_model_label

        entry = resolve_model_entry(CONFIG, label)
        self.compression_model_label = entry["label"]
        self._compressor = None
        if self.current is not None:
            self.current.compression_model = self.compression_model_label
            self.current.save_meta()
        return self.compression_model_label

    def context_usage(self) -> Dict[str, Any]:
        empty = {
            "model": self.model_label or "",
            "limit": 0,
            "used": 0,
            "available": 0,
            "percent": 0.0,
            "prompt_tokens": 0,
            "instructions_tokens": 0,
            "history_tokens": 0,
            "memory_tokens": 0,
            "plan_tokens": 0,
            "raw_history_tokens": 0,
            "raw_steps": 0,
            "last_input_tokens": 0,
        }
        if self.current is None:
            return empty

        entry = resolve_model_entry(CONFIG, self.model_label)
        limit = int(entry["model"].get("context", 0) or 0)

        instructions = self._build_instructions(
            self.current, warn_uncompressed=True
        )
        instructions_tokens = estimate_tokens(instructions)

        if self.engine is not None:
            used = self.engine.context_tokens()
            history = self.engine.history_tokens()
        else:
            used = instructions_tokens
            history = 0

        prompt_tokens = max(0, used - history)
        memory_tokens = estimate_tokens(self.current.load_memory())
        plan_tokens = estimate_tokens(
            json.dumps(self.current.load_plan(), ensure_ascii=False)
        )

        steps = self.current.read_steps()
        raw_history_tokens = (
            estimate_tokens(json.dumps(steps, ensure_ascii=False, default=str))
            if steps
            else 0
        )

        last_input_tokens = 0
        for record in reversed(steps):
            usage = record.get("token_usage")
            if isinstance(usage, dict):
                value = usage.get("input_tokens") or usage.get("prompt_tokens")
                if isinstance(value, int) and value > 0:
                    last_input_tokens = value
                    break

        percent = round(used / limit * 100, 1) if limit else 0.0
        return {
            "model": self.model_label,
            "limit": limit,
            "used": used,
            "available": max(0, limit - used) if limit else 0,
            "percent": percent,
            "prompt_tokens": prompt_tokens,
            "instructions_tokens": instructions_tokens,
            "history_tokens": history,
            "memory_tokens": memory_tokens,
            "plan_tokens": plan_tokens,
            "raw_history_tokens": raw_history_tokens,
            "raw_steps": len(steps),
            "last_input_tokens": last_input_tokens,
        }

    def list_sessions(self) -> List[Dict[str, Any]]:
        return [session.summary() for session in self.store.list()]
