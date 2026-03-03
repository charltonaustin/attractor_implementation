"""ToolHandler: execute a shell command."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from attractor.model.context import Context
from attractor.model.types import Graph, Node, Outcome, StageStatus


def _parse_timeout_seconds(timeout_str: str) -> float | None:
    if not timeout_str:
        return None
    units = {"ms": 0.001, "s": 1, "m": 60, "h": 3600, "d": 86400}
    for suffix, mult in sorted(units.items(), key=lambda x: -len(x[0])):
        if timeout_str.endswith(suffix):
            try:
                return float(timeout_str[: -len(suffix)]) * mult
            except ValueError:
                pass
    try:
        return float(timeout_str)
    except ValueError:
        return None


class ToolHandler:
    def __init__(self, venv: str | None = None) -> None:
        self.venv = venv

    def _build_env(self) -> dict | None:
        if not self.venv:
            return None
        venv_bin = str(Path(self.venv).expanduser().resolve() / "bin")
        env = os.environ.copy()
        env["PATH"] = f"{venv_bin}{os.pathsep}{env.get('PATH', '')}"
        env["VIRTUAL_ENV"] = str(Path(self.venv).expanduser().resolve())
        env.pop("PYTHONHOME", None)
        return env

    async def execute(
        self, node: Node, context: Context, graph: Graph, logs_root: str
    ) -> Outcome:
        command = node.attrs.get("tool_command", "")
        if not command:
            return Outcome(
                status=StageStatus.FAIL,
                failure_reason="No tool_command specified on node",
            )

        timeout = _parse_timeout_seconds(node.timeout)

        stage_dir = Path(logs_root) / node.id
        stage_dir.mkdir(parents=True, exist_ok=True)

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._build_env(),
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                return Outcome(
                    status=StageStatus.FAIL,
                    failure_reason=f"Tool command timed out after {timeout}s",
                )

            stdout_text = stdout.decode()
            stderr_text = stderr.decode()

            (stage_dir / "stdout.txt").write_text(stdout_text, encoding="utf-8")
            if stderr_text:
                (stage_dir / "stderr.txt").write_text(stderr_text, encoding="utf-8")

            if proc.returncode != 0:
                return Outcome(
                    status=StageStatus.FAIL,
                    failure_reason=f"Command exited {proc.returncode}: {stderr_text[:200]}",
                )

            outcome = Outcome(
                status=StageStatus.SUCCESS,
                notes=f"Tool completed: {command}",
                context_updates={"tool.output": stdout_text[:500]},
            )
            _write_status(stage_dir, outcome)
            return outcome
        except Exception as exc:
            return Outcome(status=StageStatus.FAIL, failure_reason=str(exc))


def _write_status(stage_dir: Path, outcome: Outcome) -> None:
    data = {
        "status": outcome.status.value,
        "notes": outcome.notes,
        "failure_reason": outcome.failure_reason,
    }
    (stage_dir / "status.json").write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )
