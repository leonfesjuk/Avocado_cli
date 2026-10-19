import unittest
from unittest.mock import patch

from smolagents import ChatMessage, MessageRole, Model, TokenUsage

from srs.config.config_loader import (
    CONFIG,
    PROJECT_NAME,
    ConfigError,
    get_agent_config,
    get_available_models,
    get_defaults,
    get_experts_config,
    resolve_model_entry,
    validate_config,
)
from srs.llm.agent_builder import build_agent
from srs.llm.model_factory import _resolve_api_key, _resolve_base_url, create_model
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


def _minimal_config(**overrides):
    config = {
        "project_name": "Test",
        "agent_config": {
            "agent_name": "Tester",
            "agent_type": "code",
            "max_steps": 5,
        },
        "settings": {
            "defaults": {"base_url": "https://default/v1"},
            "providers": [
                {
                    "id": "test",
                    "name": "Test",
                    "models": [{"name": "model"}],
                }
            ],
        },
    }
    config.update(overrides)
    return config


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

    def test_project_name_comes_from_config(self):
        self.assertEqual(PROJECT_NAME, CONFIG["project_name"])

    def test_defaults_are_configured(self):
        defaults = get_defaults(CONFIG)
        self.assertIn("base_url", defaults)
        self.assertIn("accept_encoding", defaults)

    def test_validation_rejects_missing_base_url(self):
        config = _minimal_config()
        config["settings"]["defaults"] = {}
        with self.assertRaises(ConfigError):
            validate_config(config)

    def test_validation_accepts_default_base_url(self):
        config = _minimal_config()
        self.assertIs(validate_config(config), config)

    def test_compression_model_must_exist(self):
        config = _minimal_config()
        config["agent_config"]["memory"] = {
            "compression": {"model": "Missing/model"}
        }
        with self.assertRaises(ConfigError) as context:
            validate_config(config)
        self.assertIn("compression.model", str(context.exception))

    def test_compression_transcript_budget_must_be_positive(self):
        config = _minimal_config()
        config["agent_config"]["memory"] = {
            "compression": {"max_transcript_tokens": 0}
        }
        with self.assertRaises(ConfigError) as context:
            validate_config(config)
        self.assertIn("max_transcript_tokens", str(context.exception))


class ModelFactoryTests(unittest.TestCase):
    def test_prefers_provider_base_url(self):
        url = _resolve_base_url(
            {"id": "test", "base_url": "https://provider/v1"},
            {"base_url": "https://default/v1"},
        )
        self.assertEqual(url, "https://provider/v1")

    def test_base_url_falls_back_to_defaults(self):
        url = _resolve_base_url(
            {"id": "test"},
            {"base_url": "https://default/v1"},
        )
        self.assertEqual(url, "https://default/v1")

    def test_missing_base_url_raises(self):
        with self.assertRaises(ConfigError):
            _resolve_base_url({"id": "test"}, {})

    def test_anthropic_url_is_used_verbatim(self):
        entry = resolve_model_entry(CONFIG, "Anthropic/claude-3-5-sonnet")
        self.assertEqual(
            _resolve_base_url(entry["provider"]),
            "https://api.anthropic.com/v1",
        )

    @patch("srs.llm.model_factory.get_secret", return_value=None)
    def test_api_key_fallback_comes_from_provider_config(self, _get_secret):
        api_key = _resolve_api_key(
            {"id": "local", "api_key_fallback": "local"}
        )
        self.assertEqual(api_key, "local")

    @patch("srs.llm.model_factory.OpenAIServerModel")
    def test_create_model_reads_defaults_from_config(self, model_class):
        create_model(
            {"id": "test", "base_url": "https://example/v1", "api_key": "key"},
            {"name": "model", "params": {"temperature": 0.2, "max_steps": 99}},
        )

        kwargs = model_class.call_args.kwargs
        self.assertEqual(kwargs["api_base"], "https://example/v1")
        self.assertEqual(
            kwargs["client_kwargs"]["default_headers"]["Accept-Encoding"],
            get_defaults(CONFIG)["accept_encoding"],
        )
        self.assertEqual(kwargs["temperature"], 0.2)
        self.assertNotIn("max_steps", kwargs)


class EngineTests(unittest.TestCase):
    def test_engine_builds_with_model(self):
        engine = SmolEngine(name="test", agent_type="code")
        engine.set_model(FakeModel())
        self.assertIsNotNone(engine.get_native_agent())

    def test_engine_reads_limits_from_config(self):
        engine = SmolEngine(name="test", agent_type="code")
        engine.set_model(FakeModel())

        native_agent = engine.get_native_agent()
        agent_config = get_agent_config(CONFIG)
        self.assertEqual(native_agent.max_steps, agent_config["max_steps"])
        self.assertEqual(
            native_agent.additional_authorized_imports,
            agent_config["additional_authorized_imports"],
        )

class BuilderTests(unittest.TestCase):
    def test_build_agent_uses_agent_config_defaults(self):
        with patch(
            "srs.llm.agent_builder.create_model",
            side_effect=lambda *a, **k: FakeModel(),
        ):
            agent = build_agent(main_model_label="Groq/allam-2-7b")

        agent_config = get_agent_config(CONFIG)
        self.assertEqual(agent.name, agent_config["agent_name"])
        self.assertEqual(agent.agent_type, agent_config["agent_type"])

    def test_model_code_loop_flag_is_used(self):
        with patch(
            "srs.llm.agent_builder.create_model",
            side_effect=lambda *a, **k: FakeModel(),
        ):
            agent = build_agent(main_model_label="Groq/openai/gpt-oss-20b")

        self.assertEqual(agent.agent_type, "tool")

    @patch(
        "srs.llm.agent_builder.create_model",
        side_effect=lambda *a, **k: FakeModel(),
    )
    def test_build_agent_with_experts(self, _create_model):
        agent = build_agent(
            main_model_label="Groq/openai/gpt-oss-20b",
            expert_names=["code_reviewer"],
        )

        self.assertIn("code_reviewer", agent.experts)


if __name__ == "__main__":
    unittest.main()
