import json
import os
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_CONFIG_FILE_PATH = Path(__file__).resolve().parent / "config.json"
CONFIG_PATH_ENV = "AVOCADO_CONFIG_PATH"


class ConfigError(ValueError):
    """Raised when the configuration file is missing or malformed."""


def _resolve_config_path() -> Path:
    override = os.getenv(CONFIG_PATH_ENV)
    return Path(override).expanduser() if override else DEFAULT_CONFIG_FILE_PATH


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
        imports = agent_config.get("additional_authorized_imports", [])
        if not isinstance(imports, list):
            errors.append(
                "'agent_config.additional_authorized_imports' must be a list."
            )
        template = agent_config.get("expert_description_template", "")
        if not isinstance(template, str):
            errors.append(
                "'agent_config.expert_description_template' must be a string."
            )

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

    if errors:
        raise ConfigError("Invalid configuration: " + " ".join(errors))

    return config


CONFIG = validate_config(load_config())
PROJECT_NAME = CONFIG["project_name"]


def get_available_models(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return every configured model as a flat, label-addressable list."""
    available: List[Dict[str, Any]] = []
    for provider in config.get("settings", {}).get("providers", []):
        for model_cfg in provider.get("models", []):
            available.append(
                {
                    "label": f"{provider['name']}/{model_cfg['name']}",
                    "provider": provider,
                    "model": model_cfg,
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
