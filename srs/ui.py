from typing import Any, Dict, List

from rich.console import Console
from rich.markdown import CodeBlock, Markdown
from rich.text import Text
from rich.theme import Theme

from .theme import AVOCADO_CODE_THEME, AVOCADO_LIME

CONSOLE = Console(
    theme=Theme(
        {
            "markdown.code": "bold #8be9fd",
            "markdown.code_block": "none",
            "markdown.link": "underline #8be9fd",
            "markdown.link_url": "#8be9fd",
        }
    )
)


class AvocadoCodeBlock(CodeBlock):
    @classmethod
    def create(cls, markdown, token):
        node_info = token.info or ""
        lexer_name = node_info.partition(" ")[0]
        return cls(lexer_name or "text", AVOCADO_CODE_THEME)


class AvocadoMarkdown(Markdown):
    elements = {
        **Markdown.elements,
        "fence": AvocadoCodeBlock,
        "code_block": AvocadoCodeBlock,
    }

    def __init__(self, markup: str, **kwargs):
        kwargs.setdefault("code_theme", AVOCADO_CODE_THEME)
        kwargs.setdefault("inline_code_theme", AVOCADO_CODE_THEME)
        super().__init__(markup, **kwargs)


def render_expert_calls(calls: List[Dict[str, Any]]) -> None:
    if not calls:
        return
    for call in calls:
        label = Text("Expert:", style=f"bold {AVOCADO_LIME}")
        CONSOLE.print(label, Text(f" {call['role']}"))
        if call.get("error"):
            CONSOLE.print(Text(f"Error: {call['error']}", style="red"))
        else:
            CONSOLE.print(Text("Task:", style="bold"))
            CONSOLE.print(Text(call["task"]))
            CONSOLE.print(Text("Response:", style="bold"))
            CONSOLE.print(Text(call["response"]))
        CONSOLE.print()


def _looks_like_markdown(text: str) -> bool:
    stripped = text.lstrip()
    return (
        "```" in text
        or "\n#" in text
        or "\n- " in text
        or "\n* " in text
        or "\n|" in text
        or "\n>" in text
        or stripped.startswith(("# ", "## ", "- ", "* ", "> ", "|"))
    )


def render_answer(text: str) -> None:
    if not text:
        return

    label = Text("Avocado:", style=f"bold {AVOCADO_LIME}")
    if _looks_like_markdown(text):
        CONSOLE.print(label)
        CONSOLE.print(AvocadoMarkdown(text))
    else:
        CONSOLE.print(label + Text(" ") + Text(text))
