from typing import Any, Callable, Dict, List, Optional

from smolagents import ChatMessage, MessageRole

from ..memory.token_counter import estimate_tokens
from .base import BaseEngine


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    return "" if content is None else str(content)


class ChatEngine(BaseEngine):
    """Stateless single-turn chat engine used when no tools are active."""

    supports_tools = False

    def __init__(
        self,
        name: str,
        description: str = "",
    ) -> None:
        super().__init__(name=name, description=description, engine_type="chat")
        self.supports_tools = False
        self._instructions = ""

    def set_model(self, model: Any) -> None:
        self.model = model

    def register_tools(self, tools: List[Any]) -> None:
        if tools:
            raise ValueError(
                "ChatEngine does not support tools. Switch to a tool-capable model first."
            )

    def unregister_tools(self, names: List[str]) -> None:
        return None

    def register_step_callback(self, callback: Optional[Callable[[Any], None]]) -> None:
        return None

    def set_instructions(self, instructions: str) -> None:
        self._instructions = instructions or ""

    def reset_memory(
        self,
        *,
        clear_state: bool = True,
        clear_executor: bool = True,
    ) -> None:
        return None

    def recreate_agent(self) -> None:
        return None

    def get_native_agent(self) -> Optional[Any]:
        return None

    def context_tokens(self) -> int:
        return estimate_tokens(self._instructions)

    def run(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        reset: Optional[bool] = None,
    ) -> str:
        if self.model is None:
            raise RuntimeError("Engine is not initialized. Set a model first.")

        prompt = task
        if context:
            context_lines = "\n".join(
                f"{key}: {value}" for key, value in context.items()
            )
            prompt = f"Context:\n{context_lines}\n\nTask: {task}"

        messages = []
        if self._instructions.strip():
            messages.append(
                ChatMessage(
                    role=MessageRole.SYSTEM,
                    content=self._instructions.strip(),
                )
            )
        messages.append(ChatMessage(role=MessageRole.USER, content=prompt))
        response = self.model.generate(messages)
        return _content_to_text(getattr(response, "content", response)).strip()
