from typing import Any, Callable, Dict, List, Optional

from smolagents import CodeAgent, ToolCallingAgent
from smolagents.memory import ActionStep, PlanningStep, SystemPromptStep

from ..config.config_loader import CONFIG, get_agent_config, get_memory_config
from .base import BaseEngine


class SmolEngine(BaseEngine):
    """Agent loop backed by smolagents' ``CodeAgent`` or ``ToolCallingAgent``."""

    def __init__(
        self,
        name: str,
        description: str = "",
        agent_type: str = "code",
    ) -> None:
        super().__init__(name=name, description=description, engine_type="smol")
        self.agent_type = agent_type
        self._agent: Optional[Any] = None
        self._managed_experts: List[Any] = []
        self._instructions = ""
        self._step_callback: Optional[Callable[[Any], None]] = None
        self._session_started = False

    def set_model(self, model: Any) -> None:
        self.model = model
        self._rebuild()

    def register_tools(self, tools: List[Any]) -> None:
        self.tools.extend(tools)
        self._rebuild()

    def register_expert(self, role: str, expert: BaseEngine) -> None:
        native_agent = expert.get_native_agent()
        if native_agent is None:
            raise RuntimeError(
                f"Expert '{role}' has no built agent. Set a model on the "
                f"expert before registering it."
            )

        if not getattr(native_agent, "name", None):
            native_agent.name = role
        if not getattr(native_agent, "description", None):
            template = get_agent_config(CONFIG).get("expert_description_template")
            if template:
                native_agent.description = template.format(role=role)

        self._managed_experts.append(native_agent)
        self.experts[role] = expert
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

        agent_class = CodeAgent if self.agent_type == "code" else ToolCallingAgent
        kwargs: Dict[str, Any] = {
            "tools": self.tools,
            "model": self.model,
            "managed_agents": self._managed_experts or None,
            "max_steps": self._max_steps(),
            "instructions": self._instructions or None,
        }
        verbosity = get_agent_config(CONFIG).get("verbosity_level")
        if verbosity is not None:
            kwargs["verbosity_level"] = verbosity
        authorized_imports = get_agent_config(CONFIG).get(
            "additional_authorized_imports", []
        )
        if self.agent_type == "code" and authorized_imports:
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
            if previous_executor is not None and hasattr(new_agent, "python_executor"):
                new_agent.python_executor = previous_executor
            new_agent.memory.system_prompt = SystemPromptStep(
                system_prompt=new_agent.system_prompt
            )

        self._agent = new_agent
