import unittest
from unittest.mock import patch

from smolagents import ChatMessage, MessageRole, Model, TokenUsage

from srs.config.config_loader import (
    CONFIG,
    get_available_models,
    get_experts_config,
    resolve_model_entry,
)
from srs.llm.agent_builder import build_agent
from srs.llm.smol_engine import SmolEngine


class FakeModel(Model):
    """Deterministic in-memory model so tests never touch the network."""

    def __init__(self) -> None:
        super().__init__(model_id="fake")

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
            content="done",
            token_usage=TokenUsage(input_tokens=0, output_tokens=0),
        )


class ConfigTests(unittest.TestCase):
    def test_available_models_have_labels(self):
        labels = [entry["label"] for entry in get_available_models(CONFIG)]
        self.assertIn("Groq/openai/gpt-oss-20b", labels)

    def test_resolve_is_case_insensitive(self):
        entry = resolve_model_entry(CONFIG, "groq/openai/gpt-oss-20b")
        self.assertEqual(entry["model"]["name"], "openai/gpt-oss-20b")

    def test_experts_are_configured(self):
        roles = {expert["role"] for expert in get_experts_config(CONFIG)}
        self.assertIn("code_reviewer", roles)


class EngineTests(unittest.TestCase):
    def test_engine_builds_with_model(self):
        engine = SmolEngine(name="test", agent_type="code")
        engine.set_model(FakeModel())
        self.assertIsNotNone(engine.get_native_agent())

    def test_register_expert_requires_built_expert(self):
        main = SmolEngine(name="main", agent_type="code")
        main.set_model(FakeModel())
        expert = SmolEngine(name="expert", agent_type="code")

        with self.assertRaises(RuntimeError):
            main.register_expert("reviewer", expert)

    def test_register_expert_after_model(self):
        main = SmolEngine(name="main", agent_type="code")
        main.set_model(FakeModel())
        expert = SmolEngine(name="expert", agent_type="code")
        expert.set_model(FakeModel())

        main.register_expert("reviewer", expert)

        self.assertIn("reviewer", main.experts)


class BuilderTests(unittest.TestCase):
    @patch("srs.llm.agent_builder.create_model", side_effect=lambda *a, **k: FakeModel())
    def test_build_agent_with_experts(self, _create_model):
        agent = build_agent(
            main_model_label="Groq/openai/gpt-oss-20b",
            expert_names=["code_reviewer"],
        )

        self.assertIn("code_reviewer", agent.experts)


if __name__ == "__main__":
    unittest.main()
