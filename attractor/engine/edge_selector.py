"""Five-step edge selection algorithm."""
from __future__ import annotations

import re

from attractor.condition.evaluator import evaluate_condition
from attractor.model.context import Context
from attractor.model.types import Edge, Graph, Node, Outcome


def _normalize_label(label: str) -> str:
    """Lowercase, trim whitespace, strip accelerator prefixes."""
    label = label.strip().lower()
    # Strip patterns like [Y] , Y) , Y -
    label = re.sub(r"^\[[a-z0-9]\]\s*", "", label)
    label = re.sub(r"^[a-z0-9]\)\s*", "", label)
    label = re.sub(r"^[a-z0-9]\s*-\s*", "", label)
    return label.strip()


def _best_by_weight_then_lexical(edges: list[Edge]) -> Edge:
    """Return the best edge by weight (desc) then target node ID (asc)."""
    return sorted(edges, key=lambda e: (-e.weight, e.to_node))[0]


def select_edge(
    node: Node,
    outcome: Outcome,
    context: Context,
    graph: Graph,
) -> Edge | None:
    """Select the next edge to follow after a node completes.

    Five-step priority order:
    1. Condition-matching edges
    2. Preferred label match
    3. Suggested next IDs
    4. Highest weight (unconditional edges)
    5. Lexical tiebreak
    """
    edges = graph.outgoing_edges(node.id)
    if not edges:
        return None

    # Step 1: Condition matching
    condition_matched: list[Edge] = []
    for edge in edges:
        if edge.condition:
            try:
                if evaluate_condition(edge.condition, outcome, context):
                    condition_matched.append(edge)
            except Exception:
                pass
    if condition_matched:
        return _best_by_weight_then_lexical(condition_matched)

    # Step 2: Preferred label match
    if outcome.preferred_label:
        norm_preferred = _normalize_label(outcome.preferred_label)
        for edge in edges:
            if _normalize_label(edge.label) == norm_preferred:
                return edge

    # Step 3: Suggested next IDs
    if outcome.suggested_next_ids:
        for suggested_id in outcome.suggested_next_ids:
            for edge in edges:
                if edge.to_node == suggested_id:
                    return edge

    # Step 4 & 5: Weight with lexical tiebreak (unconditional edges only)
    unconditional = [e for e in edges if not e.condition]
    if unconditional:
        return _best_by_weight_then_lexical(unconditional)

    # Fallback: any edge
    return _best_by_weight_then_lexical(edges)
