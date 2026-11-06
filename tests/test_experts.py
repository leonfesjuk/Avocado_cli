import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from smolagents import Tool

from srs.cli import _handle_command
from srs.config.config_loader import ConfigError, validate_config
from srs.llm.agent_builder import build_agent, build_expert_tool
from srs.llm.expert_tools import ExpertTool, expert_tool_name
from srs.llm.tool_registry import TOOL_REGISTRY, register_tool
from srs.sessions.manager import SessionManager
from srs.sessions.store import SessionStore
from tests.test_smoke import _minimal_config
from tests.test_sessions_memory import FakeCodeModel
from tests.test_tools import EchoTool


class CodeReviewerTool(Tool):
    name = "code_reviewer"
    description = "Collides with the configured expert tool name."
    inputs = {"text": {"type": "string", "description": "Text."}}
    output_type = "string"

    def forward(self, text: str) -> str:
        return text


def _expert(role="code_reviewer", **overrides):
    expert = {
        "role": role,
        "name": "Code Reviewer",
        "description": "Reviews code",
        "agent_type": "code",
        "allow_tools": False,
        "tools": [],
    }
    expert.update(overrides)
    return expert


def _config_with_experts(experts):
    config = _minimal_config()
    config["settings"]["experts"] = experts
    return config


class ExpertConfigTests(unittest.TestCase):
    def test_valid_experts_pass(self):
        config = _config_with_experts(
            [_expert(model="Test/model"), _expert(role="logic_specialist")]
        )
        self.assertIs(validate_config(config), config)

    def test_duplicate_tool_names_are_rejected(self):
        config = _config_with_experts(
            [_expert(role="code-reviewer"), _expert(role="code_reviewer")]
        )
        with self.assertRaises(ConfigError) as context:
            validate_config(config)
        self.assertIn("duplicate tool name", str(context.exception))

    def test_bad_allow_tools_is_rejected(self):
        config = _config_with_experts([_expert(allow_tools="yes")])
        with self.assertRaises(ConfigError) as context:
            validate_config(config)
        self.assertIn("allow_tools", str(context.exception))

    def test_unknown_model_is_rejected(self):
        config = _config_with_experts([_expert(model="Missing/model")])
        with self.assertRaises(ConfigError) as context:
            validate_config(config)
        self.assertIn("unknown model", str(context.exception))

    def test_unknown_agent_type_is_rejected(self):
        config = _config_with_experts([_expert(agent_type="invalid")])
        with self.assertRaises(ConfigError) as context:
            validate_config(config)
        self.assertIn("agent_type", str(context.exception))

    def test_chat_agent_type_and_expertise_are_allowed(self):
        config = _config_with_experts(
            [_expert(agent_type="chat", expertise="security")]
        )
        self.assertIs(validate_config(config), config)


class ExpertToolTests(unittest.TestCase):
    def test_tool_names_are_sanitized(self):
        self.assertEqual(expert_tool_name("code-reviewer"), "code_reviewer")
        self.assertEqual(expert_tool_name("123"), "expert_123")
        self.assertEqual(expert_tool_name("class"), "expert_class")

    def test_description_names_the_expert(self):
        tool = ExpertTool("code_reviewer", "Code Reviewer", "Reviews code", str)
        self.assertIn("Code Reviewer", tool.description)
        self.assertIn("stateless", tool.description)

    def test_overrides_are_read_on_every_call(self):
        class RecordingExpert:
            def __init__(self):
                self.tools = []

            def register_tools(self, tools):
                self.tools.extend(tools)

            def run(self, task, reset=False):
                return f"result:{task}"

        recorded = RecordingExpert()
        requested_types = []
        overrides = {"code_reviewer": {"allow_tools": False, "tools": []}}
        register_tool("test_echo", EchoTool)

        def fake_build(*args, **kwargs):
            requested_types.append(args[3])
            return recorded

        try:
            with patch(
                "srs.llm.agent_builder._build_engine",
                side_effect=fake_build,
            ):
                tool = build_expert_tool(
                    "code_reviewer",
                    _expert(agent_type="chat"),
                    "Test/model",
                    "code",
                    overrides,
                )
                tool.forward("one")
                self.assertEqual(recorded.tools, [])

                overrides["code_reviewer"] = {
                    "allow_tools": True,
                    "tools": ["test_echo"],
                }
                tool.forward("two")
                self.assertEqual([item.name for item in recorded.tools], ["echo"])
        finally:
            TOOL_REGISTRY.pop("test_echo", None)

        self.assertEqual(requested_types, ["chat", "tool"])

    def test_builder_rejects_unknown_expert(self):
        with patch(
            "srs.llm.agent_builder.create_model",
            side_effect=lambda *args, **kwargs: FakeCodeModel(),
        ):
            with self.assertRaises(KeyError):
                build_agent(
                    main_model_label="OpenAI/gpt-4o",
                    expert_names=["missing"],
                )


class ManagerExpertTests(unittest.TestCase):
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
        TOOL_REGISTRY.pop("code_reviewer", None)
        self.tempdir.cleanup()

    def _manager(self):
        return SessionManager(
            store=self.store,
            model_label="OpenAI/gpt-4o",
            compress_on_switch=False,
            compress_on_exit=False,
        )

    def _experts(self, manager):
        return {item["role"]: item for item in manager.list_experts()}

    def test_experts_start_inactive_and_can_be_toggled(self):
        manager = self._manager()
        manager.open()

        experts = self._experts(manager)
        self.assertFalse(experts["code_reviewer"]["active"])
        self.assertEqual(manager.engine.engine_type, "smol")

        manager.enable_expert("code_reviewer")
        self.assertTrue(self._experts(manager)["code_reviewer"]["active"])
        self.assertIn(
            "code_reviewer", [tool.name for tool in manager.engine.tools]
        )

        manager.disable_expert("code_reviewer")
        self.assertFalse(self._experts(manager)["code_reviewer"]["active"])
        self.assertEqual(manager.engine.engine_type, "smol")

    def test_expert_delegation_guidance_follows_active_experts(self):
        manager = self._manager()
        manager.open()

        self.assertNotIn(
            "Delegating to experts", manager.engine._instructions
        )

        manager.enable_expert("code_reviewer")
        self.assertIn("Delegating to experts", manager.engine._instructions)
        self.assertIn("only the context", manager.engine._instructions)

        manager.disable_expert("code_reviewer")
        self.assertNotIn(
            "Delegating to experts", manager.engine._instructions
        )

    def test_expert_tools_and_state_survive_restart(self):
        manager = self._manager()
        manager.open()
        manager.disable_expert("code_reviewer")
        manager.set_expert_tools("code_reviewer", ["test_echo"])
        session_id = manager.session.session_id
        manager.close(compress=False)

        restarted = self._manager()
        restarted.open(session_id)
        expert = self._experts(restarted)["code_reviewer"]

        self.assertFalse(expert["active"])
        self.assertTrue(expert["allow_tools"])
        self.assertEqual(expert["tools"], ["test_echo"])
        self.assertNotIn(
            "code_reviewer", [tool.name for tool in restarted.engine.tools]
        )

    def test_set_expert_tools_rejects_unknown_tool(self):
        manager = self._manager()
        manager.open()

        with self.assertRaises(KeyError):
            manager.set_expert_tools("code_reviewer", ["missing_tool"])

    def test_set_expert_tools_rebuilds_active_expert(self):
        manager = self._manager()
        manager.open()
        manager.enable_expert("code_reviewer")

        manager.set_expert_tools("code_reviewer", ["test_echo"])

        self.assertTrue(self._experts(manager)["code_reviewer"]["active"])
        self.assertIn("code_reviewer", manager.engine.experts)

    def test_add_tool_rejects_collision_with_expert(self):
        manager = self._manager()
        manager.open()
        manager.enable_expert("code_reviewer")
        register_tool("code_reviewer", CodeReviewerTool)

        with self.assertRaises(ValueError):
            manager.add_tool("code_reviewer")

    def test_experts_command_lists_and_mutates(self):
        manager = self._manager()
        manager.open()

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            _handle_command(manager, "/experts list")
        self.assertIn("code_reviewer", output.getvalue())
        self.assertIn("allow_tools=False", output.getvalue())

        with contextlib.redirect_stdout(io.StringIO()):
            _handle_command(manager, "/experts enable code_reviewer")
        self.assertTrue(self._experts(manager)["code_reviewer"]["active"])

        with contextlib.redirect_stdout(io.StringIO()):
            _handle_command(manager, "/experts disable code_reviewer")
        self.assertFalse(self._experts(manager)["code_reviewer"]["active"])

        with contextlib.redirect_stdout(io.StringIO()):
            _handle_command(manager, "/experts tools code_reviewer test_echo")
        expert = self._experts(manager)["code_reviewer"]
        self.assertTrue(expert["allow_tools"])
        self.assertEqual(expert["tools"], ["test_echo"])

        with contextlib.redirect_stdout(io.StringIO()):
            _handle_command(manager, "/experts tools code_reviewer none")
        expert = self._experts(manager)["code_reviewer"]
        self.assertFalse(expert["allow_tools"])
        self.assertEqual(expert["tools"], [])

    def test_experts_star_command_toggles_all(self):
        manager = self._manager()
        manager.open()

        with contextlib.redirect_stdout(io.StringIO()):
            _handle_command(manager, "/experts enable *")
        experts = self._experts(manager)
        self.assertTrue(all(item["active"] for item in experts.values()))
        self.assertIn("code_reviewer", [tool.name for tool in manager.engine.tools])
        self.assertIn(
            "logic_specialist", [tool.name for tool in manager.engine.tools]
        )

        with contextlib.redirect_stdout(io.StringIO()):
            _handle_command(manager, "/experts disable *")
        experts = self._experts(manager)
        self.assertFalse(any(item["active"] for item in experts.values()))
        self.assertEqual(manager.engine.engine_type, "smol")

    def test_expert_listing_includes_expertise_and_model(self):
        manager = self._manager()
        manager.open()

        expert = self._experts(manager)["code_reviewer"]
        self.assertEqual(expert["expertise"], "code review")
        self.assertEqual(expert["model"], "Groq/qwen/qwen3.8-27b")


if __name__ == "__main__":
    unittest.main()
