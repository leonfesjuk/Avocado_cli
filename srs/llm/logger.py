from rich import box
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.text import Text
from smolagents.monitoring import AgentLogger, LogLevel
from smolagents.utils import sanitize_for_rich

from ..theme import (
    AVOCADO_BORDER,
    AVOCADO_CODE_THEME,
    AVOCADO_LIME,
    CODE_BACKGROUND,
)


class AvocadoAgentLogger(AgentLogger):
    """Agent logger with avocado-colored run frames and step rules."""

    def log(self, *args, **kwargs) -> None:
        if args and isinstance(args[0], Text):
            if args[0].plain.startswith("Final answer:"):
                return
        super().log(*args, **kwargs)

    def log_rule(self, title: str, level: int = LogLevel.INFO) -> None:
        self.log(
            Rule(
                "[bold white]" + title,
                characters="━",
                style=AVOCADO_LIME,
            ),
            level=LogLevel.INFO,
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
        content_text = Text("\n") + Text(safe_content, style="bold") + Text("\n")
        subtitle_text = Text(safe_subtitle)
        panel_title = f"[bold {AVOCADO_LIME}]New run"
        if title:
            panel_title += f" - {title}"

        self.log(
            Panel(
                content_text,
                title=panel_title,
                subtitle=subtitle_text,
                border_style=AVOCADO_BORDER,
                subtitle_align="left",
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
                Syntax(
                    content,
                    lexer="python",
                    theme=AVOCADO_CODE_THEME,
                    word_wrap=True,
                    background_color=CODE_BACKGROUND,
                ),
                title="[bold]" + title,
                title_align="left",
                box=box.HORIZONTALS,
            ),
            level=level,
        )
