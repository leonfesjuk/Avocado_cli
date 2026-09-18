import os
from typing import Optional

from keyring import get_keyring
from keyring.errors import KeyringError

from ..config.config_loader import PROJECT_NAME

DEFAULT_SERVICE = PROJECT_NAME


class SecretNotFound(Exception):
    """Raised when a required secret cannot be resolved."""


def get_secret(
    key: str,
    *,
    service: Optional[str] = None,
    default: Optional[str] = None,
    required: bool = True,
) -> Optional[str]:
    """Resolve a secret from the environment, then the OS keyring.

    Resolution order: ``os.environ[key]``, keyring ``(service, key)``,
    ``default``. When ``required`` is true and nothing is found, a
    :class:`SecretNotFound` error is raised.
    """
    service = service or DEFAULT_SERVICE

    value = os.getenv(key)
    if value not in (None, ""):
        return value

    try:
        value = get_keyring().get_password(service, key)
        if value is not None:
            return value
    except KeyringError:
        pass

    if default is not None:
        return default

    if required:
        raise SecretNotFound(
            f"Secret '{key}' not found in keyring for service '{service}' "
            f"and no default provided."
        )
    return None
