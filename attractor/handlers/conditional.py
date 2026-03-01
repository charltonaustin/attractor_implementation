"""ConditionalHandler: no-op routing node; actual routing by edge selector."""
from __future__ import annotations

from attractor.model.types import Graph, Node, Outcome, StageStatus
from attractor.model.context import Context


class ConditionalHandler:
    async def execute(
        self, node: Node, context: Context, graph: Graph, logs_root: str
    ) -> Outcome:
        return Outcome(
            status=StageStatus.SUCCESS,
            notes=f"Conditional node evaluated: {node.id}",
        )
