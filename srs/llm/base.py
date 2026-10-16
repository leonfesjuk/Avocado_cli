from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseEngine(ABC):
    """Abstract agent-loop engine.

    The rest of the application only depends on this interface, so additional
    loops (for example a custom ReAct loop) can be introduced next to the
    smolagents-backed implementation without changing callers.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        engine_type: str = "smol",
    ) -> None:
        self.name = name
        self.description = description
        self.engine_type = engine_type
        self.model: Optional[Any] = None
        self.tools: List[Any] = []
        self.experts: Dict[str, Any] = {}

    @abstractmethod
    def set_model(self, model: Any) -> None:
        """Attach a ready-to-use model and (re)build the loop."""

    @abstractmethod
    def register_tools(self, tools: List[Any]) -> None:
        """Attach tool instances to the loop."""

    @abstractmethod
    def run(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        reset: Optional[bool] = None,
    ) -> str:
        """Execute a task and return the final text answer."""

    @abstractmethod
    def get_native_agent(self) -> Optional[Any]:
        """Return the underlying framework agent, if any."""

    def generate(
        self,
        input_data: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Backward-compatible alias for :meth:`run`."""
        return self.run(input_data, context)
