import os
from keyring import get_keyring
from keyring.errors import KeyringError
from config.config_loader import PROJECT_NAME

DEFAULT_SERVICE = PROJECT_NAME

class SecretNotFound(Exception):
    """Raised when a secret is not found in the keyring."""
    pass

def get_secret(
    key: str,
    *,
    sevice: str | None = None,
    default: str | None = None,
    required: bool = True) -> str | None:
    """
    os.environ[key] → keyring(service, key) → default.
    service  = project_name from config.json
    """
    service = sevice or DEFAULT_SERVICE
    
    val = os.getenv(key)
    if val not in (None, ""):
        return val
    
    try:
        val = get_keyring().get_password(service, key)
        
        if val is not None:
            return val
    except KeyringError:
        pass

        if default is not None:
            return default
        if required:
            raise SecretNotFound(f"Secret '{key}' not found in keyring for service '{service}' and no default provided.")
        return None