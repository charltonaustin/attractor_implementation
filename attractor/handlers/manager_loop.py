"""ManagerLoopHandler: supervisor loop over a child pipeline."""
from __future__ import annotations

import asyncio

from attractor.condition.evaluator import evaluate_condition
from attractor.model.context import Context
from attractor.model.types import Graph, Node, Outcome, StageStatus


def _parse_duration_seconds(s: str) -> float:
    if not s:
        return 45.0
    units = {"ms": 0.001, "s": 1, "m": 60, "h": 3600, "d": 86400}
    for suffix, mult in sorted(units.items(), key=lambda x: -len(x[0])):
        if s.endswith(suffix):
            try:
                return float(s[: -len(suffix)]) * mult
            except ValueError:
                pass
    try:
        return float(s)
    except ValueError:
        return 45.0


class ManagerLoopHandler:
    async def execute(
        self, node: Node, context: Context, graph: Graph, logs_root: str
    ) -> Outcome:
        poll_interval = _parse_duration_seconds(
            node.attrs.get("manager.poll_interval", "45s")
        )
        max_cycles = int(node.attrs.get("manager.max_cycles", 1000))
        stop_condition = node.attrs.get("manager.stop_condition", "")
        actions = [
            a.strip()
            for a in node.attrs.get("manager.actions", "observe,wait").split(",")
        ]

        for cycle in range(1, max_cycles + 1):
            if "observe" in actions:
                # Observe child telemetry (stub: read context keys set by child)
                pass

            # Check child status
            child_status = context.get("context.stack.child.status", "")
            if child_status == "completed":
                child_outcome = context.get("context.stack.child.outcome", "")
                if child_outcome == "success":
                    return Outcome(status=StageStatus.SUCCESS, notes="Child completed successfully")
                return Outcome(
                    status=StageStatus.FAIL,
                    failure_reason=f"Child completed with outcome: {child_outcome}",
                )
            if child_status == "failed":
                return Outcome(status=StageStatus.FAIL, failure_reason="Child pipeline failed")

            # Evaluate custom stop condition
            if stop_condition:
                try:
                    if evaluate_condition(stop_condition, None, context):
                        return Outcome(
                            status=StageStatus.SUCCESS,
                            notes=f"Stop condition satisfied at cycle {cycle}",
                        )
                except Exception:
                    pass

            if "wait" in actions:
                await asyncio.sleep(poll_interval)

        return Outcome(
            status=StageStatus.FAIL,
            failure_reason=f"Manager loop exceeded max_cycles={max_cycles}",
        )
