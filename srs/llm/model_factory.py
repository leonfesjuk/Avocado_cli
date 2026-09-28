from typing import Any, Dict

from smolagents import OpenAIServerModel

from ..secrets.secrets_get import get_secret

# Recent OpenAI SDKs use httpx2, whose Brotli decoder calls
# brotli.Decompressor.process() with an unsupported keyword argument. Requesting
# gzip/deflate avoids that broken code path. It also keeps responses small.
_SAFE_ACCEPT_ENCODING = {"Accept-Encoding": "gzip, deflate"}


def _resolve_api_key(provider: Dict[str, Any]) -> str:
    env_name = provider.get("api_key_env", "")
    api_key = get_secret(env_name, required=False) if env_name else None

    if not api_key:
        api_key = provider.get("api_key")

    if not api_key and provider.get("id") == "local":
        api_key = "local"

    if not api_key:
        raise RuntimeError(
            f"API key for provider '{provider.get('id')}' is not set. "
            f"Set the '{env_name}' environment variable or store it in the keyring."
        )
    return api_key


def _resolve_base_url(provider: Dict[str, Any]) -> str:
    base_url = provider.get("base_url") or "https://api.openai.com/v1"
    # Anthropic exposes its OpenAI-compatible endpoint under /v1.
    if provider.get("id") == "anthropic" and not base_url.rstrip("/").endswith("/v1"):
        base_url = base_url.rstrip("/") + "/v1"
    return base_url


def create_model(provider: Dict[str, Any], model_cfg: Dict[str, Any]) -> Any:
    """Build a smolagents model for an OpenAI-compatible provider.

    All configured providers (OpenAI, Anthropic, Groq and local servers such as
    Ollama) expose an OpenAI-compatible chat completions endpoint, so a single
    model class covers every case.
    """
    params = dict(model_cfg.get("params", {}))
    params.pop("max_steps", None)

    return OpenAIServerModel(
        model_id=model_cfg["name"],
        api_base=_resolve_base_url(provider),
        api_key=_resolve_api_key(provider),
        client_kwargs={"default_headers": _SAFE_ACCEPT_ENCODING},
        **params,
    )
