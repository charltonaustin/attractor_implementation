"""ConditionalHandler: routing node; passes through the previous stage's outcome."""
from __future__ import annotations

from attractor.model.types import Graph, Node, Outcome, StageStatus
from attractor.model.context import Context


class ConditionalHandler:
    async def execute(
        self, node: Node, context: Context, graph: Graph, logs_root: str
    ) -> Outcome:
        # Pass through the previous stage's outcome so edge conditions evaluate correctly.
        # The condition evaluator resolves "outcome" from the Outcome object, not context,
        # so this node must reflect the actual routing state.
        prev_value = context.get("outcome", StageStatus.SUCCESS.value)
        try:
            status = StageStatus(prev_value)
        except ValueError:
            status = StageStatus.SUCCESS
        return Outcome(
            status=status,
            notes=f"Conditional node evaluated: {node.id}",
        )
