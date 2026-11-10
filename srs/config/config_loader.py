import json
import keyword
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from ..paths import ensure_user_config, user_config_path

DEFAULT_CONFIG_FILE_PATH = Path(__file__).resolve().parent / "config.json"
CONFIG_PATH_ENV = "AVOCADO_CONFIG_PATH"


def _expert_tool_name(role: str) -> str:
    name = re.sub(r"\W", "_", role).strip("_")
    if not name:
        name = "expert"
    if name[0].isdigit() or keyword.iskeyword(name):
        name = f"expert_{name}"
    return name


class ConfigError(ValueError):
    """Raised when the configuration file is missing or malformed."""


def _resolve_config_path() -> Path:
    override = os.getenv(CONFIG_PATH_ENV)
    if override:
        return Path(override).expanduser()
    return ensure_user_config(DEFAULT_CONFIG_FILE_PATH)


CONFIG_FILE_PATH = _resolve_config_path()


def load_config(path: Path = CONFIG_FILE_PATH) -> Dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError as error:
        raise ConfigError(f"Config file not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ConfigError(f"Config file is not valid JSON: {path}: {error}") from error


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate required settings and raise a single descriptive error."""
    errors: List[str] = []

    if not config.get("project_name"):
        errors.append("'project_name' is required.")

    agent_config = config.get("agent_config")
    if not isinstance(agent_config, dict):
        errors.append("'agent_config' must be an object.")
    else:
        for key in ("agent_name", "agent_type", "max_steps"):
            if agent_config.get(key) in (None, ""):
                errors.append(f"'agent_config.{key}' is required.")
        if agent_config.get("agent_type") not in {None, "chat", "code", "tool"}:
            errors.append(
                "'agent_config.agent_type' must be 'chat', 'code' or 'tool'."
            )
        system_prompt = agent_config.get("system_prompt")
        if system_prompt is not None and not isinstance(system_prompt, str):
            errors.append("'agent_config.system_prompt' must be a string.")
        imports = agent_config.get("additional_authorized_imports", [])
        if not isinstance(imports, list):
            errors.append(
                "'agent_config.additional_authorized_imports' must be a list."
            )
        default_model = agent_config.get("default_model", "")
        if not isinstance(default_model, str):
            errors.append("'agent_config.default_model' must be a string.")

        memory = agent_config.get("memory", {})
        if not isinstance(memory, dict):
            errors.append("'agent_config.memory' must be an object.")
        else:
            mode = memory.get("mode", "session")
            if mode not in {"session", "stateless"}:
                errors.append(
                    "'agent_config.memory.mode' must be 'session' or 'stateless'."
                )
            compression = memory.get("compression", {})
            if not isinstance(compression, dict):
                errors.append(
                    "'agent_config.memory.compression' must be an object."
                )
            else:
                if not isinstance(compression.get("model", ""), str):
                    errors.append(
                        "'agent_config.memory.compression.model' must be a string."
                    )
                max_tokens = compression.get("max_summary_tokens", 1200)
                if not isinstance(max_tokens, int) or max_tokens < 0:
                    errors.append(
                        "'agent_config.memory.compression.max_summary_tokens' "
                        "must be a non-negative integer (0 disables the limit)."
                    )
                transcript_tokens = compression.get(
                    "max_transcript_tokens", 6000
                )
                if (
                    not isinstance(transcript_tokens, int)
                    or transcript_tokens < 0
                ):
                    errors.append(
                        "'agent_config.memory.compression.max_transcript_tokens' "
                        "must be a non-negative integer (0 disables the limit)."
                    )
                prompt = compression.get("prompt")
                if prompt is not None:
                    if not isinstance(prompt, str) or not prompt.strip():
                        errors.append(
                            "'agent_config.memory.compression.prompt' must be "
                            "a non-empty string."
                        )
                    else:
                        missing = [
                            placeholder
                            for placeholder in (
                                "{max_tokens}",
                                "{memory}",
                                "{transcript}",
                            )
                            if placeholder not in prompt
                        ]
                        if missing:
                            errors.append(
                                "'agent_config.memory.compression.prompt' is "
                                "missing placeholders: " + ", ".join(missing)
                            )

        sessions = agent_config.get("sessions", {})
        if not isinstance(sessions, dict):
            errors.append("'agent_config.sessions' must be an object.")
        else:
            root = sessions.get("root", "")
            if not isinstance(root, str):
                errors.append("'agent_config.sessions.root' must be a string.")

    settings = config.get("settings")
    if not isinstance(settings, dict):
        errors.append("'settings' must be an object.")
        providers: List[Any] = []
        default_base_url = None
    else:
        defaults = settings.get("defaults", {})
        if not isinstance(defaults, dict):
            errors.append("'settings.defaults' must be an object.")
            defaults = {}
        default_base_url = defaults.get("base_url")

        providers = settings.get("providers")
        if not isinstance(providers, list) or not providers:
            errors.append("'settings.providers' must be a non-empty list.")
            providers = []

    seen_labels = set()
    for index, provider in enumerate(providers):
        prefix = f"settings.providers[{index}]"
        if not isinstance(provider, dict):
            errors.append(f"'{prefix}' must be an object.")
            continue

        provider_id = provider.get("id")
        provider_name = provider.get("name")
        if not provider_id:
            errors.append(f"'{prefix}.id' is required.")
        if not provider_name:
            errors.append(f"'{prefix}.name' is required.")
        if not provider.get("base_url") and not default_base_url:
            errors.append(
                f"'{prefix}.base_url' is required unless "
                f"'settings.defaults.base_url' is set."
            )

        models = provider.get("models")
        if not isinstance(models, list) or not models:
            errors.append(f"'{prefix}.models' must be a non-empty list.")
            continue

        for model_index, model in enumerate(models):
            model_prefix = f"{prefix}.models[{model_index}]"
            if not isinstance(model, dict) or not model.get("name"):
                errors.append(f"'{model_prefix}.name' is required.")
                continue

            label = f"{provider_name or provider_id or index}/{model['name']}".lower()
            if label in seen_labels:
                errors.append(f"Duplicate model label '{label}'.")
            seen_labels.add(label)

            for flag in ("supports_code_loop", "supports_tool_calling"):
                if flag in model and not isinstance(model[flag], bool):
                    errors.append(f"'{model_prefix}.{flag}' must be a boolean.")
            model_agent_type = model.get("agent_type")
            if model_agent_type is not None and model_agent_type not in {
                "chat",
                "code",
                "tool",
            }:
                errors.append(
                    f"'{model_prefix}.agent_type' must be 'chat', 'code' or 'tool'."
                )

    if isinstance(agent_config, dict):
        default_model = agent_config.get("default_model", "")
        if (
            isinstance(default_model, str)
            and default_model.strip()
            and default_model.strip().lower() not in seen_labels
        ):
            errors.append(
                "'agent_config.default_model' references unknown model "
                f"'{default_model}'."
            )

    memory = (
        config.get("agent_config", {}).get("memory", {})
        if isinstance(config.get("agent_config"), dict)
        else {}
    )
    compression = memory.get("compression", {}) if isinstance(memory, dict) else {}
    if isinstance(compression, dict):
        label = compression.get("model", "")
        if (
            isinstance(label, str)
            and label.strip()
            and label.strip().lower() not in seen_labels
        ):
            errors.append(
                "'agent_config.memory.compression.model' references unknown "
                f"model '{label}'."
            )

    experts = settings.get("experts", []) if isinstance(settings, dict) else []
    if not isinstance(experts, list):
        errors.append("'settings.experts' must be a list.")
        experts = []

    seen_roles = set()
    seen_expert_tools = set()
    for index, expert in enumerate(experts):
        prefix = f"settings.experts[{index}]"
        if not isinstance(expert, dict):
            errors.append(f"'{prefix}' must be an object.")
            continue

        role = expert.get("role")
        if not isinstance(role, str) or not role.strip():
            errors.append(f"'{prefix}.role' is required.")
            continue
        role = role.strip()
        if role in seen_roles:
            errors.append(f"Duplicate expert role '{role}'.")
        seen_roles.add(role)

        tool_name = _expert_tool_name(role)
        if tool_name in seen_expert_tools:
            errors.append(
                f"Expert role '{role}' maps to duplicate tool name "
                f"'{tool_name}'."
            )
        seen_expert_tools.add(tool_name)

        name = expert.get("name")
        if name is not None and not isinstance(name, str):
            errors.append(f"'{prefix}.name' must be a string.")

        description = expert.get("description")
        if description is not None and not isinstance(description, str):
            errors.append(f"'{prefix}.description' must be a string.")

        agent_type = expert.get("agent_type")
        if agent_type is not None and agent_type not in {"chat", "code", "tool"}:
            errors.append(
                f"'{prefix}.agent_type' must be 'chat', 'code' or 'tool'."
            )

        for key in ("expertise", "system_prompt"):
            value = expert.get(key)
            if value is not None and not isinstance(value, str):
                errors.append(f"'{prefix}.{key}' must be a string.")

        allow_tools = expert.get("allow_tools")
        if allow_tools is not None and not isinstance(allow_tools, bool):
            errors.append(f"'{prefix}.allow_tools' must be a boolean.")

        tools = expert.get("tools", [])
        if not isinstance(tools, list) or not all(
            isinstance(tool, str) for tool in tools
        ):
            errors.append(f"'{prefix}.tools' must be a list of strings.")

        model = expert.get("model")
        if model is not None:
            if not isinstance(model, str) or not model.strip():
                errors.append(f"'{prefix}.model' must be a non-empty string.")
            elif model.strip().lower() not in seen_labels:
                errors.append(
                    f"'{prefix}.model' references unknown model '{model}'."
                )

    if errors:
        raise ConfigError("Invalid configuration: " + " ".join(errors))

    return config


CONFIG = validate_config(load_config())
PROJECT_NAME = CONFIG["project_name"]


def get_model_capabilities(model_cfg: Dict[str, Any]) -> Dict[str, bool]:
    """Return the model's tool-loop capabilities."""
    return {
        "supports_tool_calling": bool(
            model_cfg.get("supports_tool_calling", True)
        ),
        "supports_code_loop": bool(model_cfg.get("supports_code_loop", True)),
    }


def get_available_models(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return every configured model as a flat, label-addressable list."""
    available: List[Dict[str, Any]] = []
    for provider in config.get("settings", {}).get("providers", []):
        for model_cfg in provider.get("models", []):
            capabilities = get_model_capabilities(model_cfg)
            available.append(
                {
                    "label": f"{provider['name']}/{model_cfg['name']}",
                    "provider": provider,
                    "model": model_cfg,
                    **capabilities,
                }
            )
    return available


def resolve_model_entry(config: Dict[str, Any], model_label: str) -> Dict[str, Any]:
    """Resolve a label such as ``Groq/openai/gpt-oss-20b`` to its config entry."""
    target = model_label.strip().lower()
    for entry in get_available_models(config):
        if entry["label"].lower() == target:
            return entry
    raise KeyError(f"Model '{model_label}' not found in config.")


def get_experts_config(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the configured expert definitions."""
    return config.get("settings", {}).get("experts", [])


def get_agent_config(config: Dict[str, Any] = CONFIG) -> Dict[str, Any]:
    """Return the global agent settings."""
    return config["agent_config"]


def get_defaults(config: Dict[str, Any] = CONFIG) -> Dict[str, Any]:
    """Return global settings defaults such as ``base_url``."""
    return config.get("settings", {}).get("defaults", {})


def get_memory_config(config: Dict[str, Any] = CONFIG) -> Dict[str, Any]:
    """Return persisted-memory settings."""
    return config.get("agent_config", {}).get("memory", {})


def get_sessions_config(config: Dict[str, Any] = CONFIG) -> Dict[str, Any]:
    """Return session storage settings."""
    return config.get("agent_config", {}).get("sessions", {})


def get_default_model(config: Dict[str, Any] = CONFIG) -> str:
    """Return the configured default model label, if any."""
    return config.get("agent_config", {}).get("default_model", "")


def get_compression_prompt(config: Dict[str, Any] = CONFIG) -> str:
    """Return the configured compression prompt, if any."""
    return (
        config.get("agent_config", {})
        .get("memory", {})
        .get("compression", {})
        .get("prompt", "")
    )
