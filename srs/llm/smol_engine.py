from typing import Any, Dict, List, Optional

from smolagents import CodeAgent, ToolCallingAgent

from ..config.config_loader import CONFIG, get_agent_config
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

    def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> str:
        if self._agent is None:
            raise RuntimeError("Engine is not initialized. Set a model first.")

        prompt = task
        if context:
            context_lines = "\n".join(
                f"{key}: {value}" for key, value in context.items()
            )
            prompt = f"Context:\n{context_lines}\n\nTask: {task}"

        result = self._agent.run(prompt)
        return str(result) if result is not None else ""

    def get_native_agent(self) -> Optional[Any]:
        return self._agent

    @staticmethod
    def _max_steps() -> int:
        return int(get_agent_config(CONFIG)["max_steps"])

    def _rebuild(self) -> None:
        if self.model is None:
            return

        agent_class = CodeAgent if self.agent_type == "code" else ToolCallingAgent
        kwargs: Dict[str, Any] = {
            "tools": self.tools,
            "model": self.model,
            "managed_agents": self._managed_experts or None,
            "max_steps": self._max_steps(),
        }
        authorized_imports = get_agent_config(CONFIG).get(
            "additional_authorized_imports", []
        )
        if self.agent_type == "code" and authorized_imports:
            kwargs["additional_authorized_imports"] = list(authorized_imports)

        self._agent = agent_class(**kwargs)
