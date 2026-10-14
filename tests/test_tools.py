import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from smolagents import Tool

from srs.llm.builtin_tools import (
    HttpGetTool,
    ListDirTool,
    ReadFileTool,
    RunShellTool,
    WriteFileTool,
)
from srs.llm.smol_engine import SmolEngine
from srs.llm.tool_registry import (
    TOOL_REGISTRY,
    build_tools,
    get_tool,
    list_tool_names,
    register_tool,
)
from tests.test_sessions_memory import FakeCodeModel


class EchoTool(Tool):
    name = "echo"
    description = "Return the input text."
    inputs = {"text": {"type": "string", "description": "Text to echo."}}
    output_type = "string"

    def forward(self, text: str) -> str:
        return text


class RegistryTests(unittest.TestCase):
    def test_builtin_tools_are_registered(self):
        names = set(list_tool_names())
        self.assertTrue(
            {
                "read_file",
                "write_file",
                "list_dir",
                "http_get",
                "run_shell",
            }
            <= names
        )

    def test_register_and_instantiate_tool(self):
        register_tool("test_echo", EchoTool)
        try:
            tool = get_tool("test_echo")
            self.assertIsInstance(tool, EchoTool)
            self.assertEqual(tool.forward("hello"), "hello")
        finally:
            TOOL_REGISTRY.pop("test_echo", None)

    def test_factory_must_return_tool(self):
        register_tool("bad_tool", lambda: "not a tool")
        try:
            with self.assertRaises(TypeError):
                get_tool("bad_tool")
        finally:
            TOOL_REGISTRY.pop("bad_tool", None)

    def test_build_tools_skips_unknown_names(self):
        register_tool("test_echo", EchoTool)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                tools = build_tools(["test_echo", "missing_tool"])
        finally:
            TOOL_REGISTRY.pop("test_echo", None)

        self.assertEqual([tool.name for tool in tools], ["echo"])


class FileToolTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_write_read_and_list(self):
        path = self.root / "nested" / "note.txt"
        write_result = WriteFileTool().forward(str(path), "one\ntwo\nthree\n")
        self.assertIn("Wrote", write_result)

        content = ReadFileTool().forward(str(path), 2, 3)
        self.assertIn("two", content)
        self.assertIn("three", content)
        self.assertNotIn("one", content)

        listing = ListDirTool().forward(str(self.root))
        self.assertIn("nested/", listing)

    def test_read_missing_file_returns_error(self):
        result = ReadFileTool().forward(str(self.root / "missing.txt"))
        self.assertIn("read_file failed", result)


class ShellToolTests(unittest.TestCase):
    def test_run_shell_returns_output_and_code(self):
        result = RunShellTool().forward("printf hello")
        self.assertIn("exit_code: 0", result)
        self.assertIn("hello", result)


class EngineToolTests(unittest.TestCase):
    def _engine(self):
        engine = SmolEngine(name="test", agent_type="code")
        engine.set_model(FakeCodeModel())
        return engine

    def test_register_tools_replaces_duplicate_names(self):
        engine = self._engine()
        engine.register_tools([EchoTool()])
        engine.register_tools([EchoTool()])

        names = [tool.name for tool in engine.tools]
        self.assertEqual(names.count("echo"), 1)

    def test_unregister_tools_removes_only_requested(self):
        engine = self._engine()
        engine.register_tools([EchoTool()])
        engine.unregister_tools(["echo"])

        self.assertNotIn("echo", [tool.name for tool in engine.tools])

    def test_rebuild_keeps_step_number_and_monitor(self):
        engine = self._engine()
        engine.run("first")

        native = engine.get_native_agent()
        step_number = native.step_number
        monitor = native.monitor

        engine.register_tools([EchoTool()])
        rebuilt = engine.get_native_agent()

        self.assertEqual(rebuilt.step_number, step_number)
        self.assertIs(rebuilt.monitor, monitor)


if __name__ == "__main__":
    unittest.main()
