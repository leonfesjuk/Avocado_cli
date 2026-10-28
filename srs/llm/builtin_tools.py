import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

from smolagents import Tool


class ReadFileTool(Tool):
    name = "read_file"
    description = "Read a UTF-8 text file, optionally limited to a line range."
    inputs = {
        "path": {
            "type": "string",
            "description": "Path to the file to read.",
        },
        "start_line": {
            "type": "integer",
            "description": "First line to read (1-based). Defaults to 1.",
            "nullable": True,
        },
        "end_line": {
            "type": "integer",
            "description": "Last line to read, inclusive. 0 means end of file.",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(self, path: str, start_line: int = 1, end_line: int = 0) -> str:
        file_path = Path(path).expanduser()
        try:
            lines = file_path.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            return f"read_file failed: {error}"

        start = max(1, int(start_line))
        end = len(lines) if int(end_line) <= 0 else min(len(lines), int(end_line))
        if start > end:
            return ""

        selected = lines[start - 1 : end]
        return "\n".join(
            f"{number:>6}\t{line}"
            for number, line in enumerate(selected, start=start)
        )


class WriteFileTool(Tool):
    name = "write_file"
    description = "Write UTF-8 text to a file, creating parent directories."
    inputs = {
        "path": {
            "type": "string",
            "description": "Path to the file to write.",
        },
        "content": {
            "type": "string",
            "description": "Full file content.",
        },
    }
    output_type = "string"

    def forward(self, path: str, content: str) -> str:
        file_path = Path(path).expanduser()
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
        except OSError as error:
            return f"write_file failed: {error}"
        return f"Wrote {len(content)} characters to {file_path}"


class ListDirTool(Tool):
    name = "list_dir"
    description = "List directory entries. Directories are shown with a trailing slash."
    inputs = {
        "path": {
            "type": "string",
            "description": "Directory to list. Defaults to the current directory.",
            "nullable": True,
        }
    }
    output_type = "string"

    def forward(self, path: str = ".") -> str:
        directory = Path(path).expanduser()
        try:
            entries = sorted(
                directory.iterdir(),
                key=lambda entry: (not entry.is_dir(), entry.name.lower()),
            )
        except OSError as error:
            return f"list_dir failed: {error}"

        if not entries:
            return "(empty)"
        return "\n".join(
            f"{entry.name}/" if entry.is_dir() else entry.name for entry in entries
        )


class EditFileTool(Tool):
    name = "edit_file"
    description = "Replace text in a UTF-8 file, optionally replacing every match."
    inputs = {
        "path": {
            "type": "string",
            "description": "Path to the file to edit.",
        },
        "old": {
            "type": "string",
            "description": "Exact text to replace.",
        },
        "new": {
            "type": "string",
            "description": "Replacement text.",
        },
        "replace_all": {
            "type": "boolean",
            "description": "Replace every occurrence. Defaults to false.",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(
        self,
        path: str,
        old: str,
        new: str,
        replace_all: bool = False,
    ) -> str:
        file_path = Path(path).expanduser()
        if not old:
            return "edit_file failed: 'old' must not be empty"
        try:
            content = file_path.read_text(encoding="utf-8")
        except OSError as error:
            return f"edit_file failed: {error}"

        count = content.count(old)
        if count == 0:
            return f"edit_file failed: text not found in {file_path}"
        if replace_all:
            updated = content.replace(old, new)
            replaced = count
        else:
            updated = content.replace(old, new, 1)
            replaced = 1
        try:
            file_path.write_text(updated, encoding="utf-8")
        except OSError as error:
            return f"edit_file failed: {error}"
        return f"Replaced {replaced} occurrence(s) in {file_path}"


def _protected_path(path: Path) -> bool:
    resolved = path.resolve()
    cwd = Path.cwd().resolve()
    return (
        resolved == Path("/")
        or resolved == Path.home().resolve()
        or resolved == cwd
        or resolved in cwd.parents
    )


class DeletePathTool(Tool):
    name = "delete_path"
    description = "Delete a file or directory. Refuses protected system paths."
    inputs = {
        "path": {
            "type": "string",
            "description": "File or directory to delete.",
        },
        "recursive": {
            "type": "boolean",
            "description": "Delete a non-empty directory recursively. Defaults to false.",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(self, path: str, recursive: bool = False) -> str:
        target = Path(path).expanduser()
        if _protected_path(target):
            return f"delete_path failed: refusing to delete protected path {target}"
        try:
            if target.is_dir() and not target.is_symlink():
                if recursive:
                    shutil.rmtree(target)
                else:
                    target.rmdir()
            elif target.exists() or target.is_symlink():
                target.unlink()
            else:
                return f"delete_path failed: {target} does not exist"
        except OSError as error:
            return f"delete_path failed: {error}"
        return f"Deleted {target}"


class MovePathTool(Tool):
    name = "move_path"
    description = "Move or rename a file or directory."
    inputs = {
        "source": {"type": "string", "description": "Source path."},
        "destination": {"type": "string", "description": "Destination path."},
    }
    output_type = "string"

    def forward(self, source: str, destination: str) -> str:
        src = Path(source).expanduser()
        dst = Path(destination).expanduser()
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
        except OSError as error:
            return f"move_path failed: {error}"
        return f"Moved {src} to {dst}"


class CopyPathTool(Tool):
    name = "copy_path"
    description = "Copy a file or directory."
    inputs = {
        "source": {"type": "string", "description": "Source path."},
        "destination": {"type": "string", "description": "Destination path."},
    }
    output_type = "string"

    def forward(self, source: str, destination: str) -> str:
        src = Path(source).expanduser()
        dst = Path(destination).expanduser()
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        except OSError as error:
            return f"copy_path failed: {error}"
        return f"Copied {src} to {dst}"


class MakeDirTool(Tool):
    name = "make_dir"
    description = "Create a directory, including missing parents."
    inputs = {
        "path": {"type": "string", "description": "Directory path to create."},
    }
    output_type = "string"

    def forward(self, path: str) -> str:
        directory = Path(path).expanduser()
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            return f"make_dir failed: {error}"
        return f"Created directory {directory}"


class HttpGetTool(Tool):
    name = "http_get"
    description = "Fetch a URL over HTTP(S) and return the response body as text."
    inputs = {
        "url": {
            "type": "string",
            "description": "URL to fetch.",
        },
        "timeout": {
            "type": "integer",
            "description": "Request timeout in seconds. Defaults to 20.",
            "nullable": True,
        },
    }
    output_type = "string"
    max_chars = 100_000

    def forward(self, url: str, timeout: int = 20) -> str:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Avocado-AI-CLI/0.1"},
        )
        try:
            with urllib.request.urlopen(request, timeout=int(timeout)) as response:
                raw = response.read(self.max_chars + 1)
        except (urllib.error.URLError, ValueError) as error:
            return f"http_get failed: {error}"

        text = raw.decode("utf-8", errors="replace")
        if len(text) > self.max_chars:
            text = text[: self.max_chars] + "\n[...truncated]"
        return text


class RunShellTool(Tool):
    name = "run_shell"
    description = (
        "Run a shell command and return stdout, stderr and the exit code. "
        "Use it only for commands you understand."
    )
    inputs = {
        "command": {
            "type": "string",
            "description": "Shell command to execute.",
        },
        "timeout": {
            "type": "integer",
            "description": "Command timeout in seconds. Defaults to 30.",
            "nullable": True,
        },
    }
    output_type = "string"
    max_chars = 20_000

    def forward(self, command: str, timeout: int = 30) -> str:
        try:
            completed = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=int(timeout),
            )
        except subprocess.TimeoutExpired:
            return f"{self.name} timed out after {timeout} seconds"
        except OSError as error:
            return f"{self.name} failed: {error}"

        output = completed.stdout
        if completed.stderr:
            output += ("\n" if output else "") + completed.stderr
        if not output:
            output = "(no output)"
        if len(output) > self.max_chars:
            output = output[: self.max_chars] + "\n[...truncated]"

        return f"exit_code: {completed.returncode}\n{output}"


class BashTool(RunShellTool):
    name = "bash"
    description = (
        "Run a bash command and return stdout, stderr and the exit code. "
        "Use it only for commands you understand."
    )


BUILTIN_TOOLS = {
    "read_file": ReadFileTool,
    "write_file": WriteFileTool,
    "edit_file": EditFileTool,
    "list_dir": ListDirTool,
    "delete_path": DeletePathTool,
    "move_path": MovePathTool,
    "copy_path": CopyPathTool,
    "make_dir": MakeDirTool,
    "http_get": HttpGetTool,
    "run_shell": RunShellTool,
    "bash": BashTool,
}
