"""StartHandler: no-op handler for pipeline entry point."""
from __future__ import annotations

from attractor.model.types import Graph, Node, Outcome, StageStatus
from attractor.model.context import Context


class StartHandler:
    async def execute(
        self, node: Node, context: Context, graph: Graph, logs_root: str
    ) -> Outcome:
        return Outcome(status=StageStatus.SUCCESS, notes=f"Pipeline started: {graph.id}")
