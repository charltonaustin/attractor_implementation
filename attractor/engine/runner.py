"""PipelineRunner: main execution loop."""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from attractor.engine.edge_selector import select_edge
from attractor.engine.retry import build_retry_policy, execute_with_retry
from attractor.handlers.registry import HandlerRegistry, create_default_registry
from attractor.model.checkpoint import Checkpoint, load_checkpoint, save_checkpoint
from attractor.model.context import Context
from attractor.model.types import Graph, Node, Outcome, StageStatus
from attractor.transforms.variable_expansion import expand_variables
from attractor.transforms.stylesheet import apply_stylesheet_transform
from attractor.validation.validator import validate_or_raise


class PipelineError(Exception):
    pass


class PipelineRunner:
    """Main pipeline execution engine."""

    def __init__(
        self,
        registry: HandlerRegistry | None = None,
        logs_root: str = "runs",
        event_queue: asyncio.Queue | None = None,
        on_event: Callable[[Any], None] | None = None,
        resume: bool = False,
    ) -> None:
        self.registry = registry or create_default_registry()
        self.logs_root = logs_root
        self.event_queue = event_queue
        self.on_event = on_event
        self.resume = resume
        self._retry_counters: dict[str, int] = {}
        self._transforms: list[Callable[[Graph], None]] = [
            apply_stylesheet_transform,
        ]

    def register_transform(self, transform: Callable[[Graph], None]) -> None:
        self._transforms.append(transform)

    def _emit(self, event: Any) -> None:
        if self.event_queue:
            try:
                self.event_queue.put_nowait(event)
            except asyncio.QueueFull:
                pass
        if self.on_event:
            self.on_event(event)

    async def run(self, graph: Graph) -> Outcome:
        """Execute the full pipeline from start to exit."""
        from attractor.server.events import (
            PipelineStartedEvent, PipelineCompletedEvent, PipelineFailedEvent,
            StageStartedEvent, StageCompletedEvent, StageFailedEvent,
            CheckpointSavedEvent,
        )

        # Apply transforms
        for transform in self._transforms:
            transform(graph)

        # Validate
        validate_or_raise(graph)

        # Setup logs directory
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        logs_path = Path(self.logs_root) / run_id
        logs_path.mkdir(parents=True, exist_ok=True)

        # Write manifest
        manifest = {
            "pipeline_id": run_id,
            "graph_id": graph.id,
            "goal": graph.goal,
            "start_time": datetime.now(timezone.utc).isoformat(),
        }
        (logs_path / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        # Initialize context
        context = Context()
        context.set("graph.goal", graph.goal)
        context.set("graph.id", graph.id)

        # Load checkpoint if resuming
        checkpoint = load_checkpoint(logs_path) if self.resume else None
        completed_nodes: list[str] = []
        node_outcomes: dict[str, Outcome] = {}

        if checkpoint:
            context.update(checkpoint.context_snapshot)
            completed_nodes = list(checkpoint.completed_nodes)
            # Restore outcomes as stubs
            for nid, status_str in checkpoint.node_outcomes.items():
                node_outcomes[nid] = Outcome(status=StageStatus(status_str))

        # Find start node
        start_node = graph.find_start_node()
        if not start_node:
            raise PipelineError("No start node found")

        start_time = time.monotonic()
        self._emit(PipelineStartedEvent(name=graph.id, id=run_id))

        # If resuming, skip to the node after the last completed
        if checkpoint and checkpoint.current_node_id:
            current_node = self._find_resume_node(
                graph, checkpoint.current_node_id, completed_nodes
            )
        else:
            current_node = start_node

        last_outcome = Outcome(status=StageStatus.SUCCESS)

        # Main execution loop
        try:
            while True:
                node = graph.nodes.get(current_node.id)
                if not node:
                    raise PipelineError(f"Node '{current_node.id}' not found in graph")

                # Check for terminal node
                if node.is_terminal():
                    # Goal gate enforcement
                    gate_ok, failed_gate = self._check_goal_gates(graph, node_outcomes)
                    if not gate_ok and failed_gate:
                        retry_target = self._get_retry_target(failed_gate, graph)
                        if retry_target:
                            current_node = graph.nodes[retry_target]
                            continue
                        raise PipelineError(
                            f"Goal gate '{failed_gate.id}' unsatisfied and no retry target"
                        )
                    break  # Pipeline complete

                # Execute handler with retry
                handler = self.registry.resolve(node)
                if handler is None:
                    raise PipelineError(f"No handler for node '{node.id}'")

                retry_policy = build_retry_policy(node, graph)
                context.set("current_node", node.id)

                self._emit(StageStartedEvent(name=node.id, index=len(completed_nodes)))
                stage_start = time.monotonic()

                outcome = await execute_with_retry(
                    handler, node, context, graph, str(logs_path),
                    retry_policy, self._retry_counters, self.event_queue,
                )

                duration = time.monotonic() - stage_start

                # Record completion
                completed_nodes.append(node.id)
                node_outcomes[node.id] = outcome
                last_outcome = outcome

                if outcome.is_success():
                    self._emit(StageCompletedEvent(
                        name=node.id, index=len(completed_nodes), duration=duration
                    ))
                else:
                    self._emit(StageFailedEvent(
                        name=node.id, index=len(completed_nodes),
                        error=outcome.failure_reason, will_retry=False
                    ))

                # Apply context updates
                if outcome.context_updates:
                    context.update(outcome.context_updates)
                context.set("outcome", outcome.status.value)
                if outcome.preferred_label:
                    context.set("preferred_label", outcome.preferred_label)

                # Save checkpoint
                chk = Checkpoint(
                    context_snapshot=context.snapshot(),
                    current_node_id=node.id,
                    completed_nodes=list(completed_nodes),
                    node_outcomes={
                        nid: o.status.value for nid, o in node_outcomes.items()
                    },
                )
                save_checkpoint(chk, logs_path)
                self._emit(CheckpointSavedEvent(node_id=node.id))

                # Handle failure routing
                if outcome.status == StageStatus.FAIL:
                    # Try fail edge first
                    next_edge = select_edge(node, outcome, context, graph)
                    if next_edge is None:
                        # Try retry_target
                        if node.retry_target and node.retry_target in graph.nodes:
                            current_node = graph.nodes[node.retry_target]
                            continue
                        if node.fallback_retry_target and node.fallback_retry_target in graph.nodes:
                            current_node = graph.nodes[node.fallback_retry_target]
                            continue
                        raise PipelineError(
                            f"Stage '{node.id}' failed with no failure route: "
                            f"{outcome.failure_reason}"
                        )
                else:
                    next_edge = select_edge(node, outcome, context, graph)

                if next_edge is None:
                    break  # No outgoing edges; done

                # Handle loop_restart
                if next_edge.loop_restart:
                    # Re-launch (simplified: just continue from target)
                    current_node = graph.nodes[next_edge.to_node]
                    continue

                # Advance to next node
                if next_edge.to_node not in graph.nodes:
                    raise PipelineError(f"Edge target '{next_edge.to_node}' not found")
                current_node = graph.nodes[next_edge.to_node]

        except PipelineError:
            duration = time.monotonic() - start_time
            self._emit(PipelineFailedEvent(
                error=str(last_outcome.failure_reason), duration=duration
            ))
            raise

        duration = time.monotonic() - start_time
        self._emit(PipelineCompletedEvent(duration=duration, artifact_count=0))
        return last_outcome

    async def run_from(
        self, node_id: str, context: Context, graph: Graph, logs_root: str
    ) -> Outcome:
        """Execute the graph starting from a specific node (for parallel branches)."""
        if node_id not in graph.nodes:
            return Outcome(status=StageStatus.FAIL, failure_reason=f"Node '{node_id}' not found")

        current = graph.nodes[node_id]
        last_outcome = Outcome(status=StageStatus.SUCCESS)

        while True:
            if current.is_terminal():
                break

            handler = self.registry.resolve(current)
            if handler is None:
                return Outcome(
                    status=StageStatus.FAIL,
                    failure_reason=f"No handler for '{current.id}'",
                )

            retry_policy = build_retry_policy(current, graph)
            outcome = await execute_with_retry(
                handler, current, context, graph, logs_root,
                retry_policy, self._retry_counters, None,
            )
            last_outcome = outcome

            if outcome.context_updates:
                context.update(outcome.context_updates)
            context.set("outcome", outcome.status.value)

            next_edge = select_edge(current, outcome, context, graph)
            if next_edge is None:
                break

            if next_edge.to_node not in graph.nodes:
                break
            current = graph.nodes[next_edge.to_node]

        return last_outcome

    def _check_goal_gates(
        self, graph: Graph, node_outcomes: dict[str, Outcome]
    ) -> tuple[bool, Node | None]:
        for nid, outcome in node_outcomes.items():
            node = graph.nodes.get(nid)
            if node and node.goal_gate:
                if not outcome.is_success():
                    return False, node
        return True, None

    def _get_retry_target(self, node: Node, graph: Graph) -> str | None:
        for attr in (node.retry_target, node.fallback_retry_target,
                     graph.retry_target, graph.fallback_retry_target):
            if attr and attr in graph.nodes:
                return attr
        return None

    def _find_resume_node(
        self, graph: Graph, last_completed_id: str, completed_nodes: list[str]
    ) -> Node:
        """Find the node to resume from after the last completed node."""
        start = graph.find_start_node()
        if not start:
            raise PipelineError("No start node found for resume")
        # Simple approach: return start and let completed_nodes guide skipping
        # A more sophisticated implementation would track the actual next node
        if last_completed_id in graph.nodes:
            node = graph.nodes[last_completed_id]
            # Try to find the next node via edges
            from attractor.model.types import Outcome as _Outcome, StageStatus as _SS
            dummy_outcome = _Outcome(status=_SS.SUCCESS)
            dummy_ctx = Context()
            edge = select_edge(node, dummy_outcome, dummy_ctx, graph)
            if edge and edge.to_node in graph.nodes:
                return graph.nodes[edge.to_node]
        return start
