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
            return f"run_shell timed out after {timeout} seconds"
        except OSError as error:
            return f"run_shell failed: {error}"

        output = completed.stdout
        if completed.stderr:
            output += ("\n" if output else "") + completed.stderr
        if not output:
            output = "(no output)"
        if len(output) > self.max_chars:
            output = output[: self.max_chars] + "\n[...truncated]"

        return f"exit_code: {completed.returncode}\n{output}"


BUILTIN_TOOLS = {
    "read_file": ReadFileTool,
    "write_file": WriteFileTool,
    "list_dir": ListDirTool,
    "http_get": HttpGetTool,
    "run_shell": RunShellTool,
}
