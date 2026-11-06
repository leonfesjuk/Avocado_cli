import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from smolagents import ChatMessage, MessageRole, Model, TokenUsage

from srs.llm.agent_builder import build_agent
from srs.llm.chat_engine import ChatEngine
from srs.llm.expert_tools import ExpertTool
from srs.llm.smol_engine import SmolEngine
from srs.memory.compression import MemoryCompressor
from srs.memory.plan import UpdatePlanTool, default_plan
from srs.memory.prompt import build_instructions
from srs.sessions.manager import SessionManager
from srs.sessions.store import SessionStore


class FakeCodeModel(Model):
    def __init__(self) -> None:
        super().__init__(model_id="fake-code")

    def generate(
        self,
        messages,
        stop_sequences=None,
        response_format=None,
        tools_to_call_from=None,
        **kwargs,
    ) -> ChatMessage:
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content='```python\nfinal_answer("done")\n```',
            token_usage=TokenUsage(input_tokens=0, output_tokens=0),
        )


class FakeChatModel(Model):
    def __init__(self) -> None:
        super().__init__(model_id="fake-chat")
        self.calls = []

    def generate(
        self,
        messages,
        stop_sequences=None,
        response_format=None,
        tools_to_call_from=None,
        **kwargs,
    ) -> ChatMessage:
        self.calls.append(list(messages))
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content="done",
            token_usage=TokenUsage(input_tokens=0, output_tokens=0),
        )


class FakeSummaryModel(Model):
    def __init__(self) -> None:
        super().__init__(model_id="fake-summary")
        self.prompts = []

    def generate(
        self,
        messages,
        stop_sequences=None,
        response_format=None,
        tools_to_call_from=None,
        **kwargs,
    ) -> ChatMessage:
        self.prompts.append(messages[0].content)
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content="## Summary\n- goal: remember the session",
            token_usage=TokenUsage(input_tokens=0, output_tokens=0),
        )


class SessionStoreTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = SessionStore(root=Path(self.tempdir.name) / "sessions")

    def tearDown(self):
        self.tempdir.cleanup()

    def test_create_list_and_get(self):
        session = self.store.create(name="My Session", model_label="model")

        self.assertTrue(session.meta_path.exists())
        self.assertTrue(session.plan_path.exists())
        self.assertTrue(session.memory_path.exists())
        self.assertTrue(session.run_path.exists())
        self.assertEqual(self.store.get(session.session_id).name, "My Session")
        self.assertEqual(self.store.get("My Session").session_id, session.session_id)
        self.assertEqual(self.store.latest().session_id, session.session_id)

    def test_plan_memory_and_steps_round_trip(self):
        session = self.store.create(name="round-trip")
        plan = default_plan("ship it")
        session.save_plan(plan)
        session.save_memory("# memory\n", {"kind": "test"})
        session.append_step({"kind": "TaskStep", "task": "hello"})

        self.assertEqual(session.load_plan()["goal"], "ship it")
        self.assertEqual(session.load_memory(), "# memory\n")
        self.assertEqual(session.load_memory_meta()["kind"], "test")
        self.assertEqual(session.read_steps()[0]["task"], "hello")

    def test_archive_current_run_moves_file(self):
        session = self.store.create(name="archive")
        session.append_step({"kind": "TaskStep", "task": "hello"})

        archive = session.archive_current_run()

        self.assertIsNotNone(archive)
        self.assertTrue(archive.exists())
        self.assertEqual(session.read_steps(), [])
        self.assertIn("TaskStep", archive.read_text(encoding="utf-8"))

    def test_delete_removes_session(self):
        session = self.store.create(name="delete-me")
        self.store.delete(session.session_id)
        self.assertFalse(session.path.exists())
        self.assertEqual(self.store.list(), [])


class ChatEngineTests(unittest.TestCase):
    def test_chat_engine_is_stateless(self):
        model = FakeChatModel()
        engine = ChatEngine(name="chat")
        engine.set_model(model)
        engine.set_instructions("system")

        self.assertEqual(engine.run("first"), "done")
        self.assertEqual(engine.run("second"), "done")

        self.assertEqual(len(model.calls), 2)
        for messages in model.calls:
            self.assertEqual(len(messages), 2)
            self.assertEqual(messages[0].role, MessageRole.SYSTEM)
            self.assertEqual(messages[1].role, MessageRole.USER)

    def test_chat_engine_rejects_tools(self):
        engine = ChatEngine(name="chat")
        with self.assertRaises(ValueError):
            engine.register_tools([object()])


class EngineMemoryTests(unittest.TestCase):
    def _engine(self):
        engine = SmolEngine(name="test", agent_type="code")
        engine.set_model(FakeCodeModel())
        return engine

    def test_session_mode_keeps_memory_between_runs(self):
        engine = self._engine()
        self.assertEqual(engine.run("first"), "done")
        self.assertEqual(engine.run("second"), "done")

        steps = engine.get_native_agent().memory.steps
        self.assertEqual(sum(type(step).__name__ == "TaskStep" for step in steps), 2)

    def test_stateless_mode_resets_each_run(self):
        with patch(
            "srs.llm.smol_engine.get_memory_config",
            return_value={"mode": "stateless"},
        ):
            engine = self._engine()
            engine.run("first")
            engine.run("second")

        steps = engine.get_native_agent().memory.steps
        self.assertEqual(sum(type(step).__name__ == "TaskStep" for step in steps), 1)

    def test_reset_memory_clears_steps(self):
        engine = self._engine()
        engine.run("first")
        engine.reset_memory()
        self.assertEqual(engine.get_native_agent().memory.steps, [])

    def test_rebuild_preserves_steps(self):
        engine = self._engine()
        engine.run("first")
        steps_before = list(engine.get_native_agent().memory.steps)
        engine.register_tools([])
        self.assertEqual(engine.get_native_agent().memory.steps, steps_before)


class ExpertTests(unittest.TestCase):
    def test_expert_tool_is_one_shot(self):
        calls = []

        def runner(task):
            calls.append(task)
            return f"result:{task}"

        tool = ExpertTool("reviewer", "Reviewer", "Reviews code", runner)
        self.assertEqual(tool.forward("one"), "result:one")
        self.assertEqual(tool.forward("two"), "result:two")
        self.assertEqual(calls, ["one", "two"])

    @patch(
        "srs.llm.agent_builder.create_model",
        side_effect=lambda *args, **kwargs: FakeCodeModel(),
    )
    def test_builder_registers_experts_as_tools(self, _create_model):
        agent = build_agent(
            main_model_label="OpenAI/gpt-4o",
            expert_names=["code_reviewer"],
        )
        self.assertIsInstance(agent.experts["code_reviewer"], ExpertTool)


class CompressionTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = SessionStore(root=Path(self.tempdir.name) / "sessions")
        self.session = self.store.create(name="compress")
        self.session.save_plan(default_plan("remember the session"))
        self.session.save_memory("# old memory\n")
        self.session.append_step(
            {"kind": "TaskStep", "task": "inspect files", "timestamp": "now"}
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def test_compress_writes_memory_and_archives_events(self):
        model = FakeSummaryModel()
        summary = MemoryCompressor(model, max_summary_tokens=100).compress(
            self.session
        )

        self.assertIn("goal: remember the session", summary)
        self.assertIn("old memory", model.prompts[0])
        self.assertIn("inspect files", model.prompts[0])
        self.assertTrue(self.session.memory_path.read_text().startswith("## Summary"))
        self.assertEqual(self.session.read_steps(), [])
        self.assertEqual(
            self.session.load_memory_meta()["compacted_steps"],
            1,
        )
        self.assertTrue(list(self.session.archive_dir.glob("*.jsonl")))
        self.assertTrue(self.session.load_memory_meta()["archive_path"])

    def test_transcript_budget_keeps_recent_events(self):
        for index in range(30):
            self.session.append_step(
                {
                    "kind": "TaskStep",
                    "task": f"event {index}",
                    "timestamp": "now",
                }
            )
        model = FakeSummaryModel()

        MemoryCompressor(
            model,
            max_summary_tokens=100,
            max_transcript_tokens=80,
        ).compress(self.session)

        prompt = model.prompts[0]
        self.assertIn("event 29", prompt)
        self.assertNotIn("event 0", prompt)
        self.assertIn("older events omitted", prompt)

    def test_custom_compression_prompt_is_used(self):
        model = FakeSummaryModel()
        template = (
            "custom {max_tokens} {plan} {memory} {transcript}"
        )

        MemoryCompressor(
            model,
            max_summary_tokens=100,
            prompt_template=template,
        ).compress(self.session)

        self.assertTrue(model.prompts[0].startswith("custom 100"))

    def test_default_compression_prompt_excludes_plan(self):
        model = FakeSummaryModel()
        MemoryCompressor(model, max_summary_tokens=100).compress(self.session)
        self.assertNotIn("{plan}", model.prompts[0])
        self.assertNotIn("## Current plan", model.prompts[0])

    def test_archive_exists_before_memory_is_saved(self):
        observed = {}
        original_save = self.session.save_memory

        def save_memory(text, metadata=None):
            observed["archive"] = bool(
                list(self.session.archive_dir.glob("*.jsonl"))
            )
            return original_save(text, metadata)

        self.session.save_memory = save_memory
        MemoryCompressor(FakeSummaryModel(), max_summary_tokens=100).compress(
            self.session
        )

        self.assertTrue(observed["archive"])
        meta = self.session.load_memory_meta()
        self.assertIn("source_tokens", meta)
        self.assertIn("saved_tokens", meta)
        self.assertIn("saved_percent", meta)


class VerboseCodeModel(Model):
    def __init__(self) -> None:
        super().__init__(model_id="fake-verbose-code")

    def generate(
        self,
        messages,
        stop_sequences=None,
        response_format=None,
        tools_to_call_from=None,
        **kwargs,
    ) -> ChatMessage:
        padding = "# " + "x" * 50000 + "\n"
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=f"```python\n{padding}final_answer(\"done\")\n```",
            token_usage=TokenUsage(input_tokens=0, output_tokens=0),
        )


class ContextUsageTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = SessionStore(root=Path(self.tempdir.name) / "sessions")
        self.model_patch = patch(
            "srs.llm.agent_builder.create_model",
            side_effect=lambda *args, **kwargs: VerboseCodeModel(),
        )
        self.model_patch.start()

    def tearDown(self):
        self.model_patch.stop()
        self.tempdir.cleanup()

    def _manager(self):
        return SessionManager(
            store=self.store,
            model_label="Google/gemini-3.8-flash",
            compress_on_switch=False,
            compress_on_exit=False,
        )

    def test_context_usage_reports_limit_and_history(self):
        manager = self._manager()
        manager.open()
        manager.add_tool("read_file")
        self.assertEqual(manager.engine.engine_type, "smol")
        manager.run("inspect the repo")

        usage = manager.context_usage()

        self.assertEqual(usage["model"], "Google/gemini-3.8-flash")
        self.assertGreater(usage["limit"], 0)
        self.assertGreater(usage["used"], 0)
        self.assertGreater(usage["history_tokens"], 0)
        self.assertEqual(usage["raw_steps"], 3)
        self.assertGreaterEqual(usage["percent"], 0)

    def test_compress_reports_context_before_after(self):
        manager = self._manager()
        manager.open()
        manager.add_tool("read_file")
        manager._compressor = MemoryCompressor(FakeSummaryModel(), 100)
        manager.run("inspect the repo")

        summary = manager.compress()

        self.assertEqual(summary, "## Summary\n- goal: remember the session")
        self.assertEqual(manager.session.read_steps(), [])
        self.assertEqual(
            manager.engine.get_native_agent().memory.steps, []
        )

        stats = manager.last_compression
        self.assertIn("context_before", stats)
        self.assertIn("context_after", stats)
        self.assertGreater(stats["context_before"]["history_tokens"], 0)
        self.assertEqual(stats["context_after"]["history_tokens"], 0)
        self.assertEqual(stats["context_after"]["raw_history_tokens"], 0)
        self.assertGreater(stats["context_delta_tokens"], 0)
        self.assertGreater(stats["context_delta_percent"], 0)


class PromptAndPlanTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = SessionStore(root=Path(self.tempdir.name) / "sessions")
        self.session = self.store.create(name="prompt")

    def tearDown(self):
        self.tempdir.cleanup()

    def test_instructions_include_plan_and_memory(self):
        self.session.save_plan(default_plan("finish the task"))
        self.session.save_memory("remember this fact")

        instructions = build_instructions(self.session)

        self.assertIn("finish the task", instructions)
        self.assertIn("remember this fact", instructions)
        self.assertIn("You are Avocado", instructions)

    def test_expert_guidance_only_when_experts_are_active(self):
        without_experts = build_instructions(self.session)
        with_experts = build_instructions(
            self.session, experts=["code_reviewer"]
        )

        self.assertNotIn("Delegating to experts", without_experts)
        self.assertIn("Delegating to experts", with_experts)
        self.assertIn("only the context", with_experts)
        self.assertIn("Do not dump the whole conversation", with_experts)

    def test_update_plan_tool_updates_semantic_fields(self):
        tool = UpdatePlanTool(self.session)
        result = tool.forward(
            json.dumps({"steps": [{"title": "one"}], "next_action": "do it"})
        )

        plan = self.session.load_plan()
        self.assertEqual(result, "Plan updated.")
        self.assertEqual(plan["steps"], [{"title": "one"}])
        self.assertEqual(plan["next_action"], "do it")

    def test_update_plan_tool_rejects_status(self):
        tool = UpdatePlanTool(self.session)
        result = tool.forward(json.dumps({"status": "done"}))
        self.assertIn("unsupported", result)


class ManagerTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = SessionStore(root=Path(self.tempdir.name) / "sessions")
        self.model_patch = patch(
            "srs.llm.agent_builder.create_model",
            side_effect=lambda *args, **kwargs: FakeChatModel(),
        )
        self.model_patch.start()

    def tearDown(self):
        self.model_patch.stop()
        self.tempdir.cleanup()

    def _manager(self, model_label="OpenAI/gpt-4o"):
        return SessionManager(
            store=self.store,
            model_label=model_label,
            compress_on_switch=False,
            compress_on_exit=False,
        )

    def test_run_persists_plan_and_journal(self):
        manager = self._manager()
        manager.open()
        self.assertEqual(manager.engine.engine_type, "chat")
        self.assertEqual(manager.run("first task"), "done")
        self.assertEqual(manager.run("second task"), "done")

        session = manager.session
        plan = session.load_plan()
        self.assertEqual(plan["goal"], "first task")
        self.assertEqual(plan["current_step"], 0)
        kinds = [step["kind"] for step in session.read_steps()]
        self.assertIn("TaskStep", kinds)
        self.assertIn("FinalAnswerStep", kinds)

    def test_compress_and_reset(self):
        manager = self._manager()
        manager.open()
        manager.run("first task")
        manager._compressor = MemoryCompressor(FakeSummaryModel(), 100)

        summary = manager.compress()

        self.assertIn("Summary", summary)
        self.assertIn("Summary", manager.session.load_memory())
        self.assertEqual(manager.session.read_steps(), [])
        self.assertIsNone(manager.engine.get_native_agent())

        manager.reset()
        self.assertEqual(manager.session.load_plan()["goal"], "")
        self.assertIn("Summary", manager.session.load_memory())

        manager.reset(hard=True)
        self.assertEqual(manager.session.load_memory(), "")

    def test_last_session_model_is_the_default(self):
        manager = self._manager(model_label="Google/gemini-3.8-flash")
        manager.open()
        manager.run("hello")
        manager.close(compress=False)

        restarted = SessionManager(
            store=self.store,
            compress_on_switch=False,
            compress_on_exit=False,
        )
        restarted.open()
        self.assertEqual(restarted.model_label, "Google/gemini-3.8-flash")

    def test_set_model_persists_and_blocks_chat_only(self):
        manager = self._manager()
        manager.open()

        manager.set_model("Google/gemini-3.8-flash")
        self.assertEqual(manager.session.model, "Google/gemini-3.8-flash")

        manager.add_tool("read_file")
        self.assertEqual(manager.engine.engine_type, "smol")
        with self.assertRaises(ValueError):
            manager.set_model("Groq/openai/gpt-oss-20b")
        self.assertEqual(manager.session.model, "Google/gemini-3.8-flash")

    def test_switch_loads_other_session(self):
        manager = self._manager()
        manager.open()
        first = manager.session
        manager.run("first task")
        second = manager.new("second")
        manager.run("second task")

        result = manager.switch(first.session_id)

        self.assertEqual(result["session"].session_id, first.session_id)
        self.assertEqual(manager.session.session_id, first.session_id)
        self.assertEqual(manager.session.load_plan()["goal"], "first task")


if __name__ == "__main__":
    unittest.main()
