from typing import Callable

from smolagents import Tool


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
        role: str,
        name: str,
        description: str,
        runner: Callable[[str], str],
    ) -> None:
        self.name = role
        self.description = (
            f"{description} This expert is stateless: each call starts fresh, "
            "so provide the complete task and all needed context."
        )
        self._runner = runner
        super().__init__()

    def forward(self, task: str) -> str:
        return str(self._runner(task))
