import keyword
import re
from typing import Callable

from smolagents import Tool


def expert_tool_name(role: str) -> str:
    """Turn an expert role into a valid tool name."""
    name = re.sub(r"\W", "_", role).strip("_")
    if not name:
        name = "expert"
    if name[0].isdigit() or keyword.iskeyword(name):
        name = f"expert_{name}"
    return name


class ExpertTool(Tool):
    """A stateless, one-shot expert exposed as a tool to the main agent."""

    name = "expert"
    description = "Delegate a self-contained task to a stateless expert."
    inputs = {
        "task": {
            "type": "string",
            "description": (
                "A fully self-contained task. The expert has no access to the "
                "main conversation, so include all relevant context."
            ),
        }
    }
    output_type = "string"

    def __init__(
        self,
        tool_name: str,
        display_name: str,
        description: str,
        runner: Callable[[str], str],
    ) -> None:
        self.name = tool_name
        summary = f"{display_name}: {description}".strip()
        self.description = (
            f"{summary} This expert is stateless: each call starts fresh, "
            "so provide the complete task and all needed context."
        )
        self._runner = runner
        super().__init__()

    def forward(self, task: str) -> str:
        return str(self._runner(task))
