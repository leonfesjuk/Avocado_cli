from typing import Any

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.text import Text
from smolagents.monitoring import AgentLogger, LogLevel
from smolagents.utils import sanitize_for_rich

from ..theme import AVOCADO_FAINT, AVOCADO_MUTED


def _is_error_text(value: Text) -> bool:
    style = value.style
    return style is not None and "red" in str(style)


def _muted_text(value: Text) -> Text:
    return Text(value.plain, style=AVOCADO_MUTED)


def _muted_title(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Text):
        return Text(value.plain, style=AVOCADO_FAINT)
    return Text.from_markup(str(value), style=AVOCADO_FAINT)


def _dim(value: Any) -> Any:
    """Recursively render agent plumbing in a muted gray."""
    if isinstance(value, Text):
        if _is_error_text(value):
            return value
        return _muted_text(value)
    if isinstance(value, str):
        return Text(sanitize_for_rich(value), style=AVOCADO_MUTED)
    if isinstance(value, Rule):
        return Rule(
            _muted_title(value.title),
            characters="─",
            style=AVOCADO_FAINT,
            align="left",
        )
    if isinstance(value, Panel):
        return Panel(
            _dim(value.renderable),
            title=value.title,
            subtitle=value.subtitle,
            border_style=AVOCADO_FAINT,
            title_align="left",
            subtitle_align="left",
            box=box.ROUNDED,
            padding=(0, 1),
            expand=value.expand,
        )
    if isinstance(value, Group):
        return Group(*[_dim(item) for item in value.renderables])
    if isinstance(value, Syntax):
        return Text(value.code, style=AVOCADO_MUTED)
    return value


class AvocadoAgentLogger(AgentLogger):
    """Agent logger that keeps intermediate work quiet and gray.

    Only the final answer is rendered by the UI layer, in Avocado lime.
    """

    def log(self, *args, level: int | str | LogLevel = LogLevel.INFO, **kwargs) -> None:
        if args and isinstance(args[0], Text):
            if args[0].plain.startswith("Final answer:"):
                return
        if isinstance(level, str):
            level = LogLevel[level.upper()]
        if level == LogLevel.ERROR:
            super().log(*args, level=level, **kwargs)
            return
        super().log(*[_dim(arg) for arg in args], level=level, **kwargs)

    def log_rule(self, title: str, level: int = LogLevel.INFO) -> None:
        self.log(
            Rule(
                Text(f"· {title}", style=AVOCADO_FAINT),
                characters="─",
                style=AVOCADO_FAINT,
                align="left",
            ),
            level=level,
        )

    def log_task(
        self,
        content: str,
        subtitle: str,
        title: str | None = None,
        level: int = LogLevel.INFO,
    ) -> None:
        safe_content = sanitize_for_rich(content)
        safe_subtitle = sanitize_for_rich(subtitle)
        panel_title = "· task"
        if title:
            panel_title += f" · {title}"

        self.log(
            Panel(
                Text(safe_content, style=AVOCADO_MUTED),
                title=Text(panel_title, style=AVOCADO_FAINT),
                subtitle=(
                    Text(safe_subtitle, style=AVOCADO_FAINT)
                    if safe_subtitle
                    else None
                ),
                border_style=AVOCADO_FAINT,
                title_align="left",
                subtitle_align="left",
                box=box.ROUNDED,
                padding=(0, 1),
            ),
            level=level,
        )

    def log_code(
        self,
        title: str,
        content: str,
        level: int = LogLevel.INFO,
    ) -> None:
        self.log(
            Panel(
                Text(content, style=AVOCADO_MUTED),
                title=Text(f"· {title.rstrip(':')}", style=AVOCADO_FAINT),
                border_style=AVOCADO_FAINT,
                title_align="left",
                box=box.ROUNDED,
                padding=(0, 1),
            ),
            level=level,
        )

    def log_markdown(
        self,
        content: str,
        title: str | None = None,
        level: int = LogLevel.INFO,
        style: str | None = None,
    ) -> None:
        if title:
            self.log_rule(title, level=level)
        self.log(Text(sanitize_for_rich(content), style=AVOCADO_MUTED), level=level)
