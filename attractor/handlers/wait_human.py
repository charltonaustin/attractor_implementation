"""WaitForHumanHandler: blocks pipeline until a human selects an option."""
from __future__ import annotations

import re
from pathlib import Path

from attractor.interviewer.base import (
    Interviewer, Option, Question, QuestionType,
)
from attractor.model.context import Context
from attractor.model.types import Graph, Node, Outcome, StageStatus

_ACCEL_PATTERNS = [
    re.compile(r"^\[([A-Za-z0-9])\]\s*(.*)"),   # [K] Label
    re.compile(r"^([A-Za-z0-9])\)\s*(.*)"),       # K) Label
    re.compile(r"^([A-Za-z0-9])\s*-\s*(.*)"),     # K - Label
]


def parse_accelerator_key(label: str) -> str:
    """Extract accelerator key from an edge label."""
    for pattern in _ACCEL_PATTERNS:
        m = pattern.match(label.strip())
        if m:
            return m.group(1).upper()
    # Fallback: first character
    return label[0].upper() if label else "?"


class WaitForHumanHandler:
    def __init__(self, interviewer: Interviewer) -> None:
        self.interviewer = interviewer

    async def execute(
        self, node: Node, context: Context, graph: Graph, logs_root: str
    ) -> Outcome:
        if node.attrs.get("human.type") == "freeform":
            return await self._execute_freeform(node, context, graph, logs_root)

        edges = graph.outgoing_edges(node.id)
        if not edges:
            return Outcome(
                status=StageStatus.FAIL,
                failure_reason="No outgoing edges for human gate",
            )

        # Build choices from outgoing edges
        options = []
        for edge in edges:
            label = edge.label or edge.to_node
            key = parse_accelerator_key(label)
            options.append(Option(key=key, label=label))

        question = Question(
            text=node.label or "Select an option:",
            type=QuestionType.MULTIPLE_CHOICE,
            options=options,
            stage=node.id,
            timeout_seconds=float(
                node.attrs.get("human.timeout_seconds", 0) or 0
            ) or None,
        )

        answer = await self.interviewer.ask(question)

        if answer.is_timeout():
            default_choice = node.attrs.get("human.default_choice", "")
            if default_choice:
                selected_opt = next(
                    (o for o in options if o.key.upper() == default_choice.upper()),
                    options[0] if options else None,
                )
            else:
                return Outcome(
                    status=StageStatus.RETRY,
                    failure_reason="human gate timeout, no default",
                )
        elif answer.is_skipped():
            return Outcome(status=StageStatus.FAIL, failure_reason="human skipped interaction")
        else:
            selected_opt = answer.selected_option
            if selected_opt is None:
                # Try to match by value
                val = str(answer.value).upper()
                selected_opt = next(
                    (o for o in options if o.key.upper() == val), options[0] if options else None
                )

        if not selected_opt:
            return Outcome(status=StageStatus.FAIL, failure_reason="No option selected")

        # Find the edge target matching this option
        target = None
        for edge in edges:
            label = edge.label or edge.to_node
            key = parse_accelerator_key(label)
            if key.upper() == selected_opt.key.upper():
                target = edge.to_node
                break
        if target is None and edges:
            target = edges[0].to_node

        return Outcome(
            status=StageStatus.SUCCESS,
            suggested_next_ids=[target] if target else [],
            context_updates={
                "human.gate.selected": selected_opt.key,
                "human.gate.label": selected_opt.label,
            },
        )

    async def _execute_freeform(
        self, node: Node, context: Context, graph: Graph, logs_root: str
    ) -> Outcome:
        question = Question(
            text=node.label or "Enter your input:",
            type=QuestionType.FREEFORM,
            stage=node.id,
            timeout_seconds=float(
                node.attrs.get("human.timeout_seconds", 0) or 0
            ) or None,
        )

        answer = await self.interviewer.ask(question)

        if answer.is_timeout() or answer.is_skipped():
            return Outcome(status=StageStatus.FAIL, failure_reason="No input provided")

        response_text = answer.text or str(answer.value)

        stage_dir = Path(logs_root) / node.id
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / "response.md").write_text(response_text, encoding="utf-8")

        edges = graph.outgoing_edges(node.id)
        next_ids = [edges[0].to_node] if edges else []

        return Outcome(
            status=StageStatus.SUCCESS,
            suggested_next_ids=next_ids,
            context_updates={f"{node.id}_response": response_text},
        )
