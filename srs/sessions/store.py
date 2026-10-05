import json
import os
import re
import shutil
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from ..config.config_loader import CONFIG, get_sessions_config
from ..memory.plan import default_plan, utc_now


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-._")
    return slug or "session"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f"{path.name}.lock")
    with open(lock_path, "a+", encoding="utf-8") as handle:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):
            pass
        try:
            yield
        finally:
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except (ImportError, OSError):
                pass


@dataclass
class Session:
    """A single persisted conversation directory."""

    session_id: str
    name: str
    path: Path
    created_at: str
    updated_at: str
    model: str = ""

    @property
    def meta_path(self) -> Path:
        return self.path / "session.json"

    @property
    def plan_path(self) -> Path:
        return self.path / "plan.json"

    @property
    def memory_path(self) -> Path:
        return self.path / "memory.md"

    @property
    def memory_meta_path(self) -> Path:
        return self.path / "memory.meta.json"

    @property
    def run_path(self) -> Path:
        return self.path / "current_run.jsonl"

    @property
    def archive_dir(self) -> Path:
        return self.path / "archive"

    def load_meta(self) -> Dict[str, Any]:
        try:
            return json.loads(self.meta_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {
                "schema": 1,
                "session_id": self.session_id,
                "name": self.name,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "model": self.model,
            }

    def save_meta(self) -> None:
        self.updated_at = utc_now()
        payload = {
            "schema": 1,
            "session_id": self.session_id,
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "model": self.model,
        }
        _atomic_write(
            self.meta_path,
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        )

    def load_plan(self) -> Dict[str, Any]:
        try:
            payload = json.loads(self.plan_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return default_plan()
        if not isinstance(payload, dict):
            raise ValueError(f"Plan at {self.plan_path} must be a JSON object.")
        return payload

    def save_plan(self, plan: Dict[str, Any]) -> None:
        if not isinstance(plan, dict):
            raise TypeError("Plan must be a JSON object.")
        _atomic_write(
            self.plan_path,
            json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
        )

    def load_memory(self) -> str:
        try:
            return self.memory_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""

    def save_memory(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        _atomic_write(self.memory_path, text)
        payload = dict(metadata or {})
        payload["updated_at"] = utc_now()
        _atomic_write(
            self.memory_meta_path,
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        )

    def load_memory_meta(self) -> Dict[str, Any]:
        try:
            payload = json.loads(self.memory_meta_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def append_step(self, record: Dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str)
        with _locked(self.run_path):
            with open(self.run_path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def read_steps(self) -> List[Dict[str, Any]]:
        try:
            lines = self.run_path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return []

        steps: List[Dict[str, Any]] = []
        for line in lines:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                steps.append(record)
        return steps

    def raw_step_count(self) -> int:
        return len(self.read_steps())

    def clear_current_run(self) -> None:
        _atomic_write(self.run_path, "")

    def archive_current_run(self) -> Optional[Path]:
        if not self.run_path.exists() or self.run_path.stat().st_size == 0:
            return None

        self.archive_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archive_path = self.archive_dir / f"run-{stamp}-{uuid.uuid4().hex[:6]}.jsonl"
        os.replace(self.run_path, archive_path)
        _atomic_write(self.run_path, "")
        return archive_path

    def summary(self) -> Dict[str, Any]:
        return {
            "id": self.session_id,
            "name": self.name,
            "model": self.model,
            "steps": self.raw_step_count(),
            "memory_chars": len(self.load_memory()),
            "updated_at": self.updated_at,
            "path": str(self.path),
        }


class SessionStore:
    """Create, load and list session directories under a configured root."""

    def __init__(self, root: Optional[Path] = None) -> None:
        configured_root = get_sessions_config(CONFIG).get("root", "avocado_sessions")
        self.root = Path(root or configured_root).expanduser()
        if not self.root.is_absolute():
            self.root = Path.cwd() / self.root

    def _load(self, path: Path) -> Optional[Session]:
        meta_path = path / "session.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None
        if not isinstance(meta, dict):
            return None

        session_id = meta.get("session_id") or path.name
        return Session(
            session_id=session_id,
            name=meta.get("name") or path.name,
            path=path,
            created_at=meta.get("created_at") or utc_now(),
            updated_at=meta.get("updated_at") or utc_now(),
            model=meta.get("model") or "",
        )

    def create(
        self,
        name: Optional[str] = None,
        model_label: str = "",
    ) -> Session:
        self.root.mkdir(parents=True, exist_ok=True)
        session_id = uuid.uuid4().hex[:12]
        session_name = (name or "session").strip() or "session"
        path = self.root / f"{_slugify(session_name)}-{session_id}"
        path.mkdir(parents=True, exist_ok=False)

        now = utc_now()
        session = Session(
            session_id=session_id,
            name=session_name,
            path=path,
            created_at=now,
            updated_at=now,
            model=model_label,
        )
        session.save_meta()
        session.save_plan(default_plan())
        session.save_memory("", {"kind": "initial"})
        session.clear_current_run()
        return session

    def list(self) -> List[Session]:
        if not self.root.exists():
            return []
        sessions = []
        for path in self.root.iterdir():
            if not path.is_dir():
                continue
            session = self._load(path)
            if session is not None:
                sessions.append(session)
        sessions.sort(key=lambda item: item.updated_at, reverse=True)
        return sessions

    def latest(self) -> Optional[Session]:
        sessions = self.list()
        return sessions[0] if sessions else None

    def get(self, identifier: str) -> Session:
        target = identifier.strip()
        if not target:
            raise KeyError("Session identifier is empty.")

        sessions = self.list()
        for session in sessions:
            if session.session_id == target:
                return session
        for session in sessions:
            if session.session_id.startswith(target):
                return session
        for session in sessions:
            if session.name.casefold() == target.casefold():
                return session
        raise KeyError(f"Session '{identifier}' not found.")

    def delete(self, identifier: str) -> None:
        session = self.get(identifier)
        root = self.root.resolve()
        path = session.path.resolve()
        if path.parent != root:
            raise ValueError(f"Refusing to delete session outside {root}.")
        shutil.rmtree(path)
