import unittest
from unittest.mock import patch

from rich.text import Text

from srs.llm.logger import AvocadoAgentLogger
from srs.theme import (
    AVOCADO_FAINT,
    AVOCADO_MUTED,
    AvocadoCodeStyle,
    CODE_BACKGROUND,
)
from srs.ui import CONSOLE, render_answer, render_banner


class UiTests(unittest.TestCase):
    def test_code_background_is_dark_gray(self):
        self.assertEqual(CODE_BACKGROUND, "#2b2b2b")
        self.assertEqual(AvocadoCodeStyle.background_color, "#2b2b2b")

    def test_render_plain_answer_has_prefix(self):
        with CONSOLE.capture() as capture:
            render_answer("hello")
        output = capture.get()
        self.assertIn("Avocado", output)
        self.assertIn("hello", output)

    def test_render_markdown_answer_uses_prefix_line(self):
        with CONSOLE.capture() as capture:
            render_answer("```python\nprint('hi')\n```")
        output = capture.get()
        self.assertIn("Avocado", output)
        self.assertIn("print", output)

    def test_banner_shows_models_and_session(self):
        class FakeSession:
            name = "demo"
            session_id = "abc123"

        with CONSOLE.capture() as capture:
            render_banner(FakeSession(), "OpenAI/gpt-4o", "")
        output = capture.get()
        self.assertIn("Avocado", output)
        self.assertIn("demo", output)
        self.assertIn("OpenAI/gpt-4o", output)
        self.assertIn("follows the session model", output)

    def test_final_answer_line_is_suppressed(self):
        logger = AvocadoAgentLogger()
        with logger.console.capture() as capture:
            logger.log(Text("Final answer: done"))
            logger.log(Text("visible"))
        output = capture.get()
        self.assertNotIn("Final answer", output)
        self.assertIn("visible", output)

    def test_log_code_is_muted(self):
        logger = AvocadoAgentLogger()
        with patch.object(logger, "log") as log:
            logger.log_code("Executing", "print(1)")
        panel = log.call_args.args[0]
        self.assertEqual(panel.border_style, AVOCADO_FAINT)
        self.assertEqual(panel.renderable.style, AVOCADO_MUTED)


if __name__ == "__main__":
    unittest.main()
