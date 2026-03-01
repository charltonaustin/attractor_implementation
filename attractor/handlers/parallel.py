"""ParallelHandler: async fan-out to multiple branches."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from attractor.model.context import Context
from attractor.model.types import Graph, Node, Outcome, StageStatus


class ParallelHandler:
    def __init__(self, runner: Any = None) -> None:
        # runner is injected by the engine to enable recursive execution
        self._runner = runner

    def set_runner(self, runner: Any) -> None:
        self._runner = runner

    async def execute(
        self, node: Node, context: Context, graph: Graph, logs_root: str
    ) -> Outcome:
        branches = graph.outgoing_edges(node.id)
        if not branches:
            return Outcome(status=StageStatus.SUCCESS, notes="No parallel branches")

        join_policy = node.attrs.get("join_policy", "wait_all")
        error_policy = node.attrs.get("error_policy", "continue")
        max_parallel = int(node.attrs.get("max_parallel", 4))

        results: list[Outcome] = []
        sem = asyncio.Semaphore(max_parallel)

        async def run_branch(branch_edge: Any) -> Outcome:
            async with sem:
                branch_ctx = context.clone()
                if self._runner is not None:
                    return await self._runner.run_from(
                        branch_edge.to_node, branch_ctx, graph, logs_root
                    )
                return Outcome(
                    status=StageStatus.SUCCESS,
                    notes=f"Branch {branch_edge.to_node} (no runner configured)",
                )

        tasks = [asyncio.create_task(run_branch(b)) for b in branches]

        if error_policy == "fail_fast":
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_EXCEPTION
            )
            for t in pending:
                t.cancel()
            results = [t.result() for t in done if not t.cancelled() and not t.exception()]
        else:
            outcomes = await asyncio.gather(*tasks, return_exceptions=True)
            for o in outcomes:
                if isinstance(o, Exception):
                    results.append(Outcome(status=StageStatus.FAIL, failure_reason=str(o)))
                elif isinstance(o, Outcome):
                    results.append(o)

        success_count = sum(1 for r in results if r.is_success())
        fail_count = sum(1 for r in results if r.status == StageStatus.FAIL)

        # Store results for fan-in
        serialized = [
            {"status": r.status.value, "notes": r.notes, "failure_reason": r.failure_reason}
            for r in results
        ]
        context.set("parallel.results", serialized)
        context.set("parallel.success_count", success_count)
        context.set("parallel.fail_count", fail_count)

        if join_policy == "wait_all":
            status = StageStatus.SUCCESS if fail_count == 0 else StageStatus.PARTIAL_SUCCESS
        elif join_policy == "first_success":
            status = StageStatus.SUCCESS if success_count > 0 else StageStatus.FAIL
        else:
            status = StageStatus.SUCCESS if fail_count == 0 else StageStatus.PARTIAL_SUCCESS

        return Outcome(
            status=status,
            notes=f"Parallel: {success_count} succeeded, {fail_count} failed",
            context_updates={
                "parallel.results": serialized,
                "parallel.success_count": success_count,
                "parallel.fail_count": fail_count,
            },
        )
