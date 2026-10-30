from typing import Any, Callable, Dict, List, Optional

from smolagents import CodeAgent, ToolCallingAgent
from smolagents.memory import ActionStep, PlanningStep, SystemPromptStep
from smolagents.monitoring import AgentLogger, LogLevel

from ..config.config_loader import CONFIG, get_agent_config, get_memory_config
from ..memory.token_counter import estimate_tokens
from .base import BaseEngine


class SmolEngine(BaseEngine):
    """Agent loop backed by smolagents' ``CodeAgent`` or ``ToolCallingAgent``."""

    def __init__(
        self,
        name: str,
        description: str = "",
        agent_type: str = "code",
        logger: Optional[AgentLogger] = None,
    ) -> None:
        super().__init__(name=name, description=description, engine_type="smol")
        self.agent_type = agent_type
        self._agent: Optional[Any] = None
        self._instructions = ""
        self._step_callback: Optional[Callable[[Any], None]] = None
        self._session_started = False
        self._logger = logger or AgentLogger(level=LogLevel.OFF)

    def set_model(self, model: Any) -> None:
        self.model = model
        self._rebuild()

    def register_tools(self, tools: List[Any]) -> None:
        if not tools:
            return

        merged = {tool.name: tool for tool in self.tools}
        for tool in tools:
            merged[tool.name] = tool
        self.tools = list(merged.values())
        self._rebuild()

    def unregister_tools(self, names: List[str]) -> None:
        """Remove tools by name and rebuild the loop if anything changed."""
        removed = set(names)
        remaining = [tool for tool in self.tools if tool.name not in removed]
        if len(remaining) == len(self.tools):
            return

        self.tools = remaining
        self._rebuild()

    def register_step_callback(self, callback: Optional[Callable[[Any], None]]) -> None:
        self._step_callback = callback
        self._rebuild()

    def set_instructions(self, instructions: str) -> None:
        self._instructions = instructions or ""
        if self._agent is not None:
            self._agent.instructions = self._instructions or None
            self._agent.memory.system_prompt = SystemPromptStep(
                system_prompt=self._agent.system_prompt
            )

    def run(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        reset: Optional[bool] = None,
    ) -> str:
        if self._agent is None:
            raise RuntimeError("Engine is not initialized. Set a model first.")

        prompt = task
        if context:
            context_lines = "\n".join(
                f"{key}: {value}" for key, value in context.items()
            )
            prompt = f"Context:\n{context_lines}\n\nTask: {task}"

        if reset is None:
            reset = self._should_reset()
        result = self._agent.run(prompt, reset=reset)
        self._session_started = True
        return str(result) if result is not None else ""

    def reset_memory(
        self,
        *,
        clear_state: bool = True,
        clear_executor: bool = True,
    ) -> None:
        """Clear conversation steps while keeping the engine and instructions."""
        if self._agent is None:
            return

        self._agent.memory.reset()
        if clear_state:
            self._agent.state.clear()
        if clear_executor and hasattr(self._agent, "python_executor"):
            self._agent.python_executor = self._agent.create_python_executor()
        self._session_started = False

    def recreate_agent(self) -> None:
        """Build a fresh native agent without carrying conversation state."""
        self._agent = None
        self._session_started = False
        self._rebuild()

    def get_native_agent(self) -> Optional[Any]:
        return self._agent

    def context_tokens(self) -> int:
        if self._agent is None:
            return estimate_tokens(self._instructions)
        total = estimate_tokens(getattr(self._agent, "system_prompt", ""))
        return total + self.history_tokens()

    def history_tokens(self) -> int:
        if self._agent is None:
            return 0
        return sum(
            self._step_tokens(step)
            for step in getattr(self._agent.memory, "steps", [])
        )

    @staticmethod
    def _step_tokens(step: Any) -> int:
        if isinstance(step, SystemPromptStep):
            text = getattr(step, "system_prompt", "")
        elif isinstance(step, PlanningStep):
            text = getattr(step, "plan", "")
        elif isinstance(step, ActionStep):
            text = "\n".join(
                str(value)
                for value in (
                    getattr(step, "model_output", ""),
                    getattr(step, "observations", ""),
                    getattr(step, "error", ""),
                    getattr(step, "tool_calls", ""),
                )
                if value
            )
        else:
            text = getattr(step, "task", None) or getattr(
                step, "output", ""
            )
        return estimate_tokens(str(text or ""))

    def _should_reset(self) -> bool:
        mode = get_memory_config(CONFIG).get("mode", "session")
        if mode == "stateless":
            return True
        return not self._session_started

    @staticmethod
    def _max_steps() -> int:
        return int(get_agent_config(CONFIG)["max_steps"])

    def _rebuild(self) -> None:
        if self.model is None:
            return

        previous_agent = self._agent
        previous_steps = (
            list(previous_agent.memory.steps) if previous_agent is not None else []
        )
        previous_state = (
            dict(previous_agent.state) if previous_agent is not None else {}
        )
        previous_executor = (
            getattr(previous_agent, "python_executor", None)
            if previous_agent is not None
            else None
        )
        previous_step_number = (
            getattr(previous_agent, "step_number", 0)
            if previous_agent is not None
            else 0
        )
        previous_monitor = (
            getattr(previous_agent, "monitor", None)
            if previous_agent is not None
            else None
        )

        agent_class = CodeAgent if self.agent_type == "code" else ToolCallingAgent
        kwargs: Dict[str, Any] = {
            "tools": self.tools,
            "model": self.model,
            "max_steps": self._max_steps(),
            "instructions": self._instructions or None,
            "logger": self._logger,
        }
        authorized_imports = get_agent_config(CONFIG).get(
            "additional_authorized_imports", []
        )
        if self.agent_type == "code":
            kwargs["code_block_tags"] = "markdown"
            if authorized_imports:
                kwargs["additional_authorized_imports"] = list(authorized_imports)
        if self._step_callback is not None:
            kwargs["step_callbacks"] = {
                ActionStep: self._step_callback,
                PlanningStep: self._step_callback,
            }

        new_agent = agent_class(**kwargs)

        if (
            previous_agent is not None
            and type(previous_agent) is type(new_agent)
        ):
            new_agent.memory.steps = previous_steps
            new_agent.state.update(previous_state)
            new_agent.step_number = previous_step_number
            if (
                previous_monitor is not None
                and getattr(previous_agent, "model", None) is self.model
            ):
                new_agent.monitor = previous_monitor
            if previous_executor is not None and hasattr(new_agent, "python_executor"):
                new_agent.python_executor = previous_executor
            new_agent.memory.system_prompt = SystemPromptStep(
                system_prompt=new_agent.system_prompt
            )

        self._agent = new_agent
