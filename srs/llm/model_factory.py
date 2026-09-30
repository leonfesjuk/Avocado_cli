from typing import Any, Dict, Optional

from smolagents import OpenAIServerModel

from ..config.config_loader import ConfigError, get_defaults
from ..secrets.secrets_get import get_secret


def _resolve_api_key(provider: Dict[str, Any]) -> str:
    env_name = provider.get("api_key_env", "")
    api_key = get_secret(env_name, required=False) if env_name else None

    if not api_key:
        api_key = provider.get("api_key")

    if not api_key:
        api_key = provider.get("api_key_fallback")

    if not api_key:
        raise RuntimeError(
            f"API key for provider '{provider.get('id')}' is not set. "
            f"Set the '{env_name}' environment variable or store it in the keyring."
        )
    return api_key


def _resolve_base_url(
    provider: Dict[str, Any],
    defaults: Optional[Dict[str, Any]] = None,
) -> str:
    defaults = get_defaults() if defaults is None else defaults
    base_url = provider.get("base_url") or defaults.get("base_url")

    if not base_url:
        raise ConfigError(
            f"Provider '{provider.get('id')}' has no 'base_url' and "
            "'settings.defaults.base_url' is not configured."
        )
    return base_url


def _client_kwargs(defaults: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    defaults = get_defaults() if defaults is None else defaults
    headers: Dict[str, str] = {}

    accept_encoding = defaults.get("accept_encoding")
    if accept_encoding:
        headers["Accept-Encoding"] = accept_encoding

    return {"default_headers": headers}


def create_model(provider: Dict[str, Any], model_cfg: Dict[str, Any]) -> Any:
    """Build a smolagents model for an OpenAI-compatible provider.

    All configured providers (OpenAI, Anthropic, Groq and local servers such as
    Ollama) expose an OpenAI-compatible chat completions endpoint, so a single
    model class covers every case.
    """
    params = dict(model_cfg.get("params", {}))
    params.pop("max_steps", None)
    defaults = get_defaults()

    return OpenAIServerModel(
        model_id=model_cfg["name"],
        api_base=_resolve_base_url(provider, defaults),
        api_key=_resolve_api_key(provider),
        client_kwargs=_client_kwargs(defaults),
        **params,
    )
