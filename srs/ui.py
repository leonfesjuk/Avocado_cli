from typing import Any, Dict, List

from rich import box
from rich.console import Console
from rich.markdown import CodeBlock, Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text
from rich.theme import Theme

from .theme import (
    AVOCADO_BORDER,
    AVOCADO_CODE_THEME,
    AVOCADO_FAINT,
    AVOCADO_LIME,
    AVOCADO_MUTED,
)

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


def render_banner(
    session: Any,
    model_label: str,
    compression_label: str = "",
) -> None:
    """Small startup header: brand, session and the active models."""
    body = Text()
    body.append("session  ", style=AVOCADO_FAINT)
    body.append(session.name, style="white")
    body.append(f"  ({session.session_id})", style=AVOCADO_MUTED)
    body.append("\n")
    body.append("model    ", style=AVOCADO_FAINT)
    body.append(model_label, style="white")
    body.append("\n")
    body.append("compress ", style=AVOCADO_FAINT)
    body.append(
        compression_label or "follows the session model",
        style=AVOCADO_MUTED,
    )
    CONSOLE.print(
        Panel(
            body,
            title=Text("Avocado", style=f"bold {AVOCADO_LIME}"),
            border_style=AVOCADO_BORDER,
            title_align="left",
            box=box.ROUNDED,
            padding=(0, 1),
            expand=False,
        )
    )


def render_status(message: str) -> None:
    """Dim, non-essential output such as context counters."""
    CONSOLE.print(Text(message, style=AVOCADO_MUTED))


def render_expert_calls(calls: List[Dict[str, Any]]) -> None:
    if not calls:
        return
    for call in calls:
        header = Text("  ↳ expert", style=AVOCADO_FAINT)
        header.append(f" · {call['role']}", style=AVOCADO_MUTED)
        CONSOLE.print(header)
        if call.get("error"):
            CONSOLE.print(
                Text("    error:  ", style=AVOCADO_FAINT)
                + Text(str(call["error"]), style="red")
            )
        else:
            CONSOLE.print(
                Text("    task:   ", style=AVOCADO_FAINT)
                + Text(str(call.get("task", "")), style=AVOCADO_MUTED)
            )
            CONSOLE.print(
                Text("    answer: ", style=AVOCADO_FAINT)
                + Text(str(call.get("response", "")), style=AVOCADO_MUTED)
            )


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
    """The final, user-facing answer from the main Avocado model."""
    if not text:
        return

    CONSOLE.print()
    CONSOLE.print(
        Rule(
            Text("Avocado", style=f"bold {AVOCADO_LIME}"),
            characters="─",
            style=AVOCADO_BORDER,
            align="left",
        )
    )
    if _looks_like_markdown(text):
        CONSOLE.print(AvocadoMarkdown(text))
    else:
        CONSOLE.print(Text(text))
