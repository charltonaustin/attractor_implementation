"""LocalExecutionEnvironment: filesystem and shell access for tool execution."""
from __future__ import annotations

import os
import platform
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_SENSITIVE_SUFFIXES = (
    "_API_KEY", "_SECRET", "_TOKEN", "_PASSWORD", "_CREDENTIAL",
)


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False


class LocalExecutionEnvironment:
    def __init__(self, working_dir: str | None = None) -> None:
        self._working_dir = Path(working_dir).resolve() if working_dir else Path.cwd()

    def working_directory(self) -> str:
        return str(self._working_dir)

    def platform(self) -> str:
        return platform.system()

    def os_version(self) -> str:
        return platform.version()

    def file_exists(self, path: str) -> bool:
        return Path(self._resolve(path)).exists()

    def read_file(self, path: str, offset: int | None = None, limit: int | None = None) -> str:
        resolved = self._resolve(path)
        text = Path(resolved).read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines(keepends=True)
        start = (offset - 1) if offset and offset > 0 else 0
        if limit:
            lines = lines[start : start + limit]
        elif offset:
            lines = lines[start:]
        return "".join(lines)

    def write_file(self, path: str, content: str) -> None:
        resolved = Path(self._resolve(path))
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")

    def edit_file(self, path: str, old_string: str, new_string: str) -> None:
        resolved = self._resolve(path)
        text = Path(resolved).read_text(encoding="utf-8", errors="replace")
        if old_string not in text:
            raise ValueError(f"old_string not found in {path}")
        updated = text.replace(old_string, new_string, 1)
        Path(resolved).write_text(updated, encoding="utf-8")

    def exec_command(
        self,
        command: str,
        timeout_ms: int = 30000,
        working_dir: str | None = None,
        env_vars: dict[str, str] | None = None,
    ) -> ExecResult:
        cwd = working_dir or str(self._working_dir)
        env = self._build_env(env_vars)
        timeout_s = timeout_ms / 1000.0

        try:
            proc = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=cwd,
                env=env,
                start_new_session=True,
            )
            try:
                stdout, stderr = proc.communicate(timeout=timeout_s)
                return ExecResult(
                    stdout=stdout.decode("utf-8", errors="replace"),
                    stderr=stderr.decode("utf-8", errors="replace"),
                    exit_code=proc.returncode,
                )
            except subprocess.TimeoutExpired:
                # SIGTERM then SIGKILL
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    proc.wait(timeout=2)
                except (subprocess.TimeoutExpired, ProcessLookupError):
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                stdout_bytes = proc.stdout.read() if proc.stdout else b""
                stderr_bytes = proc.stderr.read() if proc.stderr else b""
                return ExecResult(
                    stdout=stdout_bytes.decode("utf-8", errors="replace"),
                    stderr=stderr_bytes.decode("utf-8", errors="replace"),
                    exit_code=-1,
                    timed_out=True,
                )
        except Exception as exc:
            return ExecResult(stdout="", stderr=str(exc), exit_code=1)

    def grep(self, pattern: str, path: str, options: dict[str, Any] | None = None) -> str:
        opts = options or {}
        flags = ["-r"] if Path(self._resolve(path)).is_dir() else []
        if opts.get("case_insensitive"):
            flags.append("-i")
        cmd = ["grep", "-n"] + flags + [pattern, self._resolve(path)]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self._working_dir),
                timeout=10,
            )
            return result.stdout
        except subprocess.TimeoutExpired:
            return ""
        except FileNotFoundError:
            return ""

    def glob(self, pattern: str, path: str | None = None) -> list[str]:
        base = Path(self._resolve(path)) if path else self._working_dir
        return [str(p) for p in base.glob(pattern)]

    def is_git_repo(self) -> bool:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            cwd=str(self._working_dir),
        )
        return result.returncode == 0

    def git_branch(self) -> str:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(self._working_dir),
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return ""

    def _resolve(self, path: str) -> str:
        p = Path(path)
        if p.is_absolute():
            return str(p)
        return str(self._working_dir / p)

    def _build_env(self, extra: dict[str, str] | None) -> dict[str, str]:
        env = {
            k: v for k, v in os.environ.items()
            if not any(k.upper().endswith(s) for s in _SENSITIVE_SUFFIXES)
        }
        if extra:
            for k, v in extra.items():
                if not any(k.upper().endswith(s) for s in _SENSITIVE_SUFFIXES):
                    env[k] = v
        return env
