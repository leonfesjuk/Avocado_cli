import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from smolagents import Tool

from srs.cli import _handle_command
from srs.llm.builtin_tools import (
    BashTool,
    CopyPathTool,
    DeletePathTool,
    EditFileTool,
    HttpGetTool,
    ListDirTool,
    MakeDirTool,
    MovePathTool,
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
from srs.sessions.manager import SessionManager
from srs.sessions.store import SessionStore
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
                "edit_file",
                "list_dir",
                "delete_path",
                "move_path",
                "copy_path",
                "make_dir",
                "http_get",
                "run_shell",
                "bash",
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

    def test_edit_file_replaces_one_or_all(self):
        path = self.root / "edit.txt"
        path.write_text("one one two", encoding="utf-8")

        result = EditFileTool().forward(str(path), "one", "1")
        self.assertIn("Replaced 1", result)
        self.assertEqual(path.read_text(encoding="utf-8"), "1 one two")

        EditFileTool().forward(str(path), "one", "1", replace_all=True)
        self.assertEqual(path.read_text(encoding="utf-8"), "1 1 two")

        missing = EditFileTool().forward(str(path), "absent", "x")
        self.assertIn("not found", missing)

    def test_copy_move_make_dir_and_delete(self):
        source = self.root / "source.txt"
        source.write_text("payload", encoding="utf-8")
        copied = self.root / "nested" / "copy.txt"
        moved = self.root / "nested" / "moved.txt"
        directory = self.root / "created" / "deep"

        self.assertIn("Copied", CopyPathTool().forward(str(source), str(copied)))
        self.assertEqual(copied.read_text(encoding="utf-8"), "payload")
        self.assertIn("Moved", MovePathTool().forward(str(copied), str(moved)))
        self.assertTrue(moved.exists())
        self.assertFalse(copied.exists())
        self.assertIn("Created", MakeDirTool().forward(str(directory)))
        self.assertTrue(directory.is_dir())
        self.assertIn("Deleted", DeletePathTool().forward(str(moved)))
        self.assertFalse(moved.exists())

    def test_delete_path_refuses_protected_paths(self):
        result = DeletePathTool().forward(str(Path.cwd()))
        self.assertIn("refusing", result)


class ShellToolTests(unittest.TestCase):
    def test_run_shell_returns_output_and_code(self):
        result = RunShellTool().forward("printf hello")
        self.assertIn("exit_code: 0", result)
        self.assertIn("hello", result)

    def test_bash_alias_works(self):
        result = BashTool().forward("printf bash-ok")
        self.assertEqual(BashTool.name, "bash")
        self.assertIn("bash-ok", result)


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


class ManagerToolTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = SessionStore(root=Path(self.tempdir.name) / "sessions")
        self.model_patch = patch(
            "srs.llm.agent_builder.create_model",
            side_effect=lambda *args, **kwargs: FakeCodeModel(),
        )
        self.model_patch.start()
        register_tool("test_echo", EchoTool)

    def tearDown(self):
        self.model_patch.stop()
        TOOL_REGISTRY.pop("test_echo", None)
        self.tempdir.cleanup()

    def _manager(self):
        return SessionManager(
            store=self.store,
            model_label="OpenAI/gpt-4o",
            compress_on_switch=False,
            compress_on_exit=False,
        )

    def test_add_and_remove_tool_updates_engine(self):
        manager = self._manager()
        manager.open()
        self.assertEqual(manager.engine.engine_type, "chat")

        manager.add_tool("test_echo")
        self.assertIn("test_echo", manager.session.active_tools)
        self.assertIn("echo", [tool.name for tool in manager.engine.tools])
        self.assertEqual(manager.engine.engine_type, "smol")

        manager.remove_tool("test_echo")
        self.assertNotIn("test_echo", manager.session.active_tools)
        self.assertNotIn("echo", [tool.name for tool in manager.engine.tools])
        self.assertEqual(manager.engine.engine_type, "chat")

    def test_active_tool_survives_restart(self):
        manager = self._manager()
        manager.open()
        manager.add_tool("test_echo")
        session_id = manager.session.session_id
        manager.close(compress=False)

        restarted = self._manager()
        restarted.open(session_id)

        self.assertIn("test_echo", restarted.session.active_tools)
        self.assertIn("echo", [tool.name for tool in restarted.engine.tools])

    def test_add_unknown_tool_raises(self):
        manager = self._manager()
        manager.open()

        with self.assertRaises(KeyError):
            manager.add_tool("missing_tool")

    def test_tools_command_lists_and_mutates(self):
        manager = self._manager()
        manager.open()

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            _handle_command(manager, "/tools list")
        self.assertIn("test_echo", output.getvalue())
        self.assertIn("(none)", output.getvalue())

        with contextlib.redirect_stdout(io.StringIO()):
            _handle_command(manager, "/tools add test_echo")
        self.assertIn("test_echo", manager.session.active_tools)

        with contextlib.redirect_stdout(io.StringIO()):
            _handle_command(manager, "/tools remove test_echo")
        self.assertNotIn("test_echo", manager.session.active_tools)

    def test_add_all_builtin_tools_and_clear(self):
        manager = self._manager()
        manager.open()

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            _handle_command(manager, "/tools add *")
        self.assertIn("bash", manager.session.active_tools)
        self.assertIn("edit_file", manager.session.active_tools)
        self.assertIn("Added:", output.getvalue())
        self.assertEqual(manager.engine.engine_type, "smol")

        with contextlib.redirect_stdout(io.StringIO()):
            _handle_command(manager, "/tools clear")
        self.assertEqual(manager.session.active_tools, [])
        self.assertEqual(manager.engine.engine_type, "chat")

    def test_chat_only_model_blocks_tools(self):
        manager = self._manager()
        manager.set_model("Groq/openai/gpt-oss-20b")
        manager.open()

        with self.assertRaises(ValueError):
            manager.add_tool("read_file")


if __name__ == "__main__":
    unittest.main()
