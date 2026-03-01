"""Stylesheet transform: apply model_stylesheet to graph nodes."""
from __future__ import annotations

from attractor.model.types import Graph
from attractor.stylesheet.applicator import apply_stylesheet_from_graph


def apply_stylesheet_transform(graph: Graph) -> None:
    """Apply the graph's model_stylesheet to all nodes."""
    apply_stylesheet_from_graph(graph)
