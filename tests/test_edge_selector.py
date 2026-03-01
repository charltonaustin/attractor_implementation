"""Tests for the edge selection algorithm."""
import pytest
from attractor.engine.edge_selector import select_edge
from attractor.model.types import Edge, Graph, Node, Outcome, StageStatus
from attractor.model.context import Context


def _make_graph(nodes: list[str], edges: list[tuple]) -> Graph:
    g = Graph(id="test")
    for nid in nodes:
        g.nodes[nid] = Node(id=nid)
    for e in edges:
        from_n, to_n = e[0], e[1]
        attrs = e[2] if len(e) > 2 else {}
        g.edges.append(Edge(from_node=from_n, to_node=to_n, attrs=attrs))
    return g


def test_unconditional_single_edge():
    g = _make_graph(["a", "b"], [("a", "b")])
    node = g.nodes["a"]
    outcome = Outcome(status=StageStatus.SUCCESS)
    ctx = Context()
    edge = select_edge(node, outcome, ctx, g)
    assert edge is not None
    assert edge.to_node == "b"


def test_condition_match():
    g = _make_graph(
        ["a", "b", "c"],
        [
            ("a", "b", {"condition": "outcome=success"}),
            ("a", "c", {"condition": "outcome=fail"}),
        ],
    )
    node = g.nodes["a"]
    outcome = Outcome(status=StageStatus.SUCCESS)
    ctx = Context()
    edge = select_edge(node, outcome, ctx, g)
    assert edge.to_node == "b"


def test_preferred_label():
    g = _make_graph(
        ["a", "b", "c"],
        [
            ("a", "b", {"label": "Yes"}),
            ("a", "c", {"label": "No"}),
        ],
    )
    node = g.nodes["a"]
    outcome = Outcome(status=StageStatus.SUCCESS, preferred_label="Yes")
    ctx = Context()
    edge = select_edge(node, outcome, ctx, g)
    assert edge.to_node == "b"


def test_suggested_next_ids():
    g = _make_graph(
        ["a", "b", "c"],
        [("a", "b"), ("a", "c")],
    )
    node = g.nodes["a"]
    outcome = Outcome(status=StageStatus.SUCCESS, suggested_next_ids=["c"])
    ctx = Context()
    edge = select_edge(node, outcome, ctx, g)
    assert edge.to_node == "c"


def test_weight_tiebreak():
    g = _make_graph(
        ["a", "b", "c"],
        [
            ("a", "b", {"weight": 1}),
            ("a", "c", {"weight": 5}),
        ],
    )
    node = g.nodes["a"]
    outcome = Outcome(status=StageStatus.SUCCESS)
    ctx = Context()
    edge = select_edge(node, outcome, ctx, g)
    assert edge.to_node == "c"  # higher weight wins


def test_lexical_tiebreak():
    g = _make_graph(
        ["a", "b", "c"],
        [("a", "c"), ("a", "b")],  # c listed first, b comes first lexically
    )
    node = g.nodes["a"]
    outcome = Outcome(status=StageStatus.SUCCESS)
    ctx = Context()
    edge = select_edge(node, outcome, ctx, g)
    assert edge.to_node == "b"  # b < c lexically


def test_no_edges():
    g = _make_graph(["a"], [])
    node = g.nodes["a"]
    outcome = Outcome(status=StageStatus.SUCCESS)
    ctx = Context()
    edge = select_edge(node, outcome, ctx, g)
    assert edge is None


def test_label_normalization_with_accelerator():
    """Accelerator prefix [A] should be stripped during label matching."""
    g = _make_graph(
        ["a", "b", "c"],
        [
            ("a", "b", {"label": "[A] Approve"}),
            ("a", "c", {"label": "[R] Reject"}),
        ],
    )
    node = g.nodes["a"]
    outcome = Outcome(status=StageStatus.SUCCESS, preferred_label="[A] Approve")
    ctx = Context()
    edge = select_edge(node, outcome, ctx, g)
    assert edge.to_node == "b"
