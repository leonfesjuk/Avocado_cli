import json
from pathlib import Path
from typing import Any, Dict, List

CONFIG_FILE_PATH = Path(__file__).resolve().parent / "config.json"
PROJECT_NAME = "Avocado_AI_CLI"


def load_config(path: Path = CONFIG_FILE_PATH) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as file:
        return json.load(file)


CONFIG = load_config()


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
