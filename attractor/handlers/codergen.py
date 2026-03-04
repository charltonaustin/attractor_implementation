"""CodergenHandler: LLM task handler with claude CLI backend."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from attractor.model.context import Context
from attractor.model.types import Graph, Node, Outcome, StageStatus
from attractor.transforms.variable_expansion import expand_variables


class CodergenBackend:
    """Abstract interface for LLM backends."""

    async def run(self, node: Node, prompt: str, context: Context) -> str | Outcome:
        raise NotImplementedError


class ClaudeCliBackend(CodergenBackend):
    """Spawns the `claude` CLI in --print mode."""

    def __init__(self, extra_args: list[str] | None = None, workdir: str | None = None) -> None:
        self.extra_args = extra_args or []
        self.workdir = workdir

    async def run(self, node: Node, prompt: str, context: Context) -> str:
        cmd = ["claude", "--print"] + self.extra_args
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.workdir,
            )
            stdout, stderr = await proc.communicate(prompt.encode())
            if proc.returncode != 0:
                err = stderr.decode().strip()
                raise RuntimeError(f"claude CLI exited {proc.returncode}: {err}")
            return stdout.decode()
        except FileNotFoundError:
            raise RuntimeError(
                "claude CLI not found. Install it or use --backend simulate."
            )


def _write_status(stage_dir: Path, outcome: Outcome) -> None:
    data = {
        "status": outcome.status.value,
        "notes": outcome.notes,
        "failure_reason": outcome.failure_reason,
        "context_updates": outcome.context_updates,
    }
    (stage_dir / "status.json").write_text(
        json.dumps(data, indent=2, default=str), encoding="utf-8"
    )


class CodergenHandler:
    def __init__(self, backend: CodergenBackend | None = None) -> None:
        self.backend = backend

    async def execute(
        self, node: Node, context: Context, graph: Graph, logs_root: str
    ) -> Outcome:
        # 1. Build prompt, injecting prior stage responses from disk into context.
        # Always re-read from disk so loops get fresh outputs from re-run stages.
        logs_path = Path(logs_root)
        for response_file in logs_path.glob("*/response.md"):
            stage_id = response_file.parent.name
            key = f"{stage_id}_response"
            context.set(key, response_file.read_text(encoding="utf-8"))

        prompt = node.prompt or node.label
        prompt = expand_variables(prompt, graph, context)

        # 2. Write prompt to logs
        stage_dir = Path(logs_root) / node.id
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / "prompt.md").write_text(prompt, encoding="utf-8")

        # 3. Call backend
        if self.backend is not None:
            try:
                result = await self.backend.run(node, prompt, context)
                if isinstance(result, Outcome):
                    _write_status(stage_dir, result)
                    return result
                response_text = str(result)
            except Exception as exc:
                outcome = Outcome(
                    status=StageStatus.FAIL,
                    failure_reason=str(exc),
                )
                _write_status(stage_dir, outcome)
                return outcome
        else:
            response_text = f"[Simulated] Response for stage: {node.id}"

        # 4. Write response
        (stage_dir / "response.md").write_text(response_text, encoding="utf-8")

        # 5. Build and write outcome
        outcome = Outcome(
            status=StageStatus.SUCCESS,
            notes=f"Stage completed: {node.id}",
            context_updates={
                "last_stage": node.id,
                f"{node.id}_response": response_text,
            },
        )
        _write_status(stage_dir, outcome)
        return outcome
