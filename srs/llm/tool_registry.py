from typing import Any, Callable, Dict, List

# Maps a tool name to a zero-argument factory. Empty by default: the agent
# starts without tools and tools are attached on demand (for example from a
# slash command in an interactive session).
TOOL_REGISTRY: Dict[str, Callable[[], Any]] = {}


def register_tool(name: str, factory: Callable[[], Any]) -> None:
    """Register a tool factory under ``name``."""
    TOOL_REGISTRY[name] = factory


def get_tool(name: str) -> Any:
    """Instantiate a registered tool by name."""
    try:
        factory = TOOL_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Tool '{name}' is not registered. Add it to TOOL_REGISTRY."
        ) from None
    return factory()


def build_tools(names: List[str]) -> List[Any]:
    """Instantiate tools by name, skipping unknown entries with a warning."""
    tools = []
    for name in names:
        try:
            tools.append(get_tool(name))
        except KeyError as error:
            print(f"Warning: {error}")
    return tools
