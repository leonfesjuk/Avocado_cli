from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class BaseAgent(ABC):
    def __init__(self, name: str):
        self.name = name
        self.model: Optional[Any] = None
        self.tools: List[Any] = []
        self.experts: Dict[str, Any] = {}

    @abstractmethod
    def create_model(self, model_id: str, **kwargs: Any) -> Any:
        """initialisation of LLM model with given model_id and parameters."""
        pass
        
    @abstractmethod
    def register_tools(self, tools: List[Any]) -> None:
        """Registering tools for the agent."""
        pass
        
    @abstractmethod
    def register_expert(self, role: str, expert_agent: Any) -> None:
        """Registering expert agents for the agent."""
        pass
        
    @abstractmethod
    def generate(self, input_data: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generating a response based on the input data and context."""
        pass