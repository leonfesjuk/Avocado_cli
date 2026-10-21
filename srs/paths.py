import os
import shutil
from pathlib import Path
from typing import Optional


def _env_path(name: str) -> Optional[Path]:
    value = os.getenv(name)
    return Path(value).expanduser() if value else None


def avocado_home() -> Optional[Path]:
    """Return ``AVOCADO_HOME`` when the user opted into a single-root layout."""
    return _env_path("AVOCADO_HOME")


def config_home() -> Path:
    root = avocado_home()
    if root is not None:
        return root
    return _env_path("XDG_CONFIG_HOME") or Path.home() / ".config"


def data_home() -> Path:
    root = avocado_home()
    if root is not None:
        return root
    return _env_path("XDG_DATA_HOME") or Path.home() / ".local" / "share"


def state_home() -> Path:
    root = avocado_home()
    if root is not None:
        return root
    return _env_path("XDG_STATE_HOME") or Path.home() / ".local" / "state"


def cache_home() -> Path:
    root = avocado_home()
    if root is not None:
        return root
    return _env_path("XDG_CACHE_HOME") or Path.home() / ".cache"


def user_config_path() -> Path:
    if avocado_home() is not None:
        return avocado_home() / "config.json"  # type: ignore[operator]
    return config_home() / "avocado" / "config.json"


def sessions_root() -> Path:
    if avocado_home() is not None:
        return avocado_home() / "sessions"  # type: ignore[operator]
    return data_home() / "avocado" / "sessions"


def state_dir() -> Path:
    if avocado_home() is not None:
        return avocado_home() / "state"  # type: ignore[operator]
    return state_home() / "avocado"


def cache_dir() -> Path:
    if avocado_home() is not None:
        return avocado_home() / "cache"  # type: ignore[operator]
    return cache_home() / "avocado"


def ensure_user_config(template: Path) -> Path:
    """Copy the packaged config to the XDG location on first use."""
    target = user_config_path()
    if target.exists():
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    shutil.copyfile(template, temporary)
    os.replace(temporary, target)
    return target
