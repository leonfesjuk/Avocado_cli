import unittest
from unittest.mock import patch

from rich.text import Text

from srs.llm.logger import AvocadoAgentLogger
from srs.theme import AVOCADO_CODE_THEME, AvocadoCodeStyle, CODE_BACKGROUND
from srs.ui import CONSOLE, render_answer


class UiTests(unittest.TestCase):
    def test_code_background_is_dark_gray(self):
        self.assertEqual(CODE_BACKGROUND, "#2b2b2b")
        self.assertEqual(AvocadoCodeStyle.background_color, "#2b2b2b")

    def test_render_plain_answer_has_prefix(self):
        with CONSOLE.capture() as capture:
            render_answer("hello")
        self.assertIn("Avocado: hello", capture.get())

    def test_render_markdown_answer_uses_prefix_line(self):
        with CONSOLE.capture() as capture:
            render_answer("```python\nprint('hi')\n```")
        output = capture.get()
        self.assertIn("Avocado:", output)
        self.assertIn("print", output)

    def test_final_answer_line_is_suppressed(self):
        logger = AvocadoAgentLogger()
        with logger.console.capture() as capture:
            logger.log(Text("Final answer: done"))
            logger.log(Text("visible"))
        output = capture.get()
        self.assertNotIn("Final answer", output)
        self.assertIn("visible", output)

    def test_log_code_uses_avocado_theme(self):
        logger = AvocadoAgentLogger()
        with patch.object(logger, "log") as log:
            logger.log_code("Executing", "print(1)")
        panel = log.call_args.args[0]
        self.assertIs(panel.renderable._theme, AVOCADO_CODE_THEME)


if __name__ == "__main__":
    unittest.main()
