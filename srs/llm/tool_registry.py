from typing import Any, Callable, Dict, List

from smolagents import Tool

# Tool name -> zero-argument factory. Built-ins are registered on import;
# sessions can attach more tools with the /tools command.
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

    tool = factory()
    if not isinstance(tool, Tool):
        raise TypeError(
            f"Factory for tool '{name}' returned {type(tool).__name__}, "
            "expected a smolagents Tool."
        )
    return tool


def build_tools(names: List[str]) -> List[Any]:
    """Instantiate tools by name, skipping unknown entries with a warning."""
    tools = []
    for name in names:
        try:
            tools.append(get_tool(name))
        except KeyError as error:
            print(f"Warning: {error}")
    return tools


def list_tool_names() -> List[str]:
    """Return every registered tool name in sorted order."""
    return sorted(TOOL_REGISTRY)


def list_builtin_tool_names() -> List[str]:
    """Return the packaged built-in tool names, excluding user additions."""
    from .builtin_tools import BUILTIN_TOOLS

    return sorted(BUILTIN_TOOLS)


def register_builtin_tools() -> None:
    """Load the default tools once, without replacing user registrations."""
    from .builtin_tools import BUILTIN_TOOLS

    for name, factory in BUILTIN_TOOLS.items():
        TOOL_REGISTRY.setdefault(name, factory)


register_builtin_tools()
