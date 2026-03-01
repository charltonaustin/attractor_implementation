"""FanInHandler: consolidate results from a preceding parallel node."""
from __future__ import annotations

from attractor.model.context import Context
from attractor.model.types import Graph, Node, Outcome, StageStatus


_STATUS_RANK = {
    "success": 0,
    "partial_success": 1,
    "retry": 2,
    "fail": 3,
}


class FanInHandler:
    async def execute(
        self, node: Node, context: Context, graph: Graph, logs_root: str
    ) -> Outcome:
        results = context.get("parallel.results")
        if not results:
            return Outcome(
                status=StageStatus.FAIL,
                failure_reason="No parallel results to evaluate",
            )

        if not isinstance(results, list):
            results = []

        if not results:
            return Outcome(status=StageStatus.FAIL, failure_reason="Empty parallel results")

        # Heuristic selection: best by status rank
        def _rank(r: dict) -> tuple:
            status = r.get("status", "fail")
            return (_STATUS_RANK.get(status, 99), r.get("id", ""))

        best = sorted(results, key=_rank)[0]

        context_updates = {
            "parallel.fan_in.best_status": best.get("status", ""),
            "parallel.fan_in.best_notes": best.get("notes", ""),
        }

        return Outcome(
            status=StageStatus.SUCCESS,
            notes=f"Fan-in selected best candidate with status: {best.get('status')}",
            context_updates=context_updates,
        )
