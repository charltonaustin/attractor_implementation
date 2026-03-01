"""Tests for the DOT parser."""
import pytest
from attractor.parser.dot_parser import parse, ParseError


def test_simple_graph():
    src = """
    digraph Simple {
        graph [goal="Test goal"]
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        start -> exit
    }
    """
    graph = parse(src)
    assert graph.id == "Simple"
    assert graph.goal == "Test goal"
    assert "start" in graph.nodes
    assert "exit" in graph.nodes
    assert len(graph.edges) == 1
    assert graph.edges[0].from_node == "start"
    assert graph.edges[0].to_node == "exit"


def test_chained_edges():
    src = """
    digraph Chain {
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        a [shape=box]
        b [shape=box]
        start -> a -> b -> exit
    }
    """
    graph = parse(src)
    assert len(graph.edges) == 3
    assert graph.edges[0].from_node == "start"
    assert graph.edges[0].to_node == "a"
    assert graph.edges[1].from_node == "a"
    assert graph.edges[1].to_node == "b"
    assert graph.edges[2].from_node == "b"
    assert graph.edges[2].to_node == "exit"


def test_edge_attributes():
    src = """
    digraph Attrs {
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        start -> exit [label="Yes", condition="outcome=success", weight=5]
    }
    """
    graph = parse(src)
    edge = graph.edges[0]
    assert edge.label == "Yes"
    assert edge.condition == "outcome=success"
    assert edge.weight == 5


def test_node_defaults():
    src = """
    digraph Defaults {
        node [shape=box, timeout="900s"]
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        task1 [label="Task 1"]
        start -> task1 -> exit
    }
    """
    graph = parse(src)
    assert graph.nodes["task1"].attrs.get("timeout") == "900s"
    # Explicit shape overrides default
    assert graph.nodes["start"].shape == "Mdiamond"


def test_subgraph_class_derivation():
    src = """
    digraph Sub {
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        subgraph cluster_loop {
            label = "Loop A"
            node [thread_id="loop-a"]
            task1 [shape=box]
        }
        start -> task1 -> exit
    }
    """
    graph = parse(src)
    assert "loop-a" in graph.nodes["task1"].attrs.get("class", "")


def test_comments_stripped():
    src = """
    // This is a line comment
    digraph Comments {
        /* block comment */
        start [shape=Mdiamond]  // inline comment
        exit  [shape=Msquare]
        start -> exit
    }
    """
    graph = parse(src)
    assert graph.id == "Comments"


def test_boolean_values():
    src = """
    digraph Bool {
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        task [shape=box, goal_gate=true, auto_status=false]
        start -> task -> exit
    }
    """
    graph = parse(src)
    assert graph.nodes["task"].attrs["goal_gate"] is True
    assert graph.nodes["task"].attrs["auto_status"] is False


def test_graph_level_attr():
    src = """
    digraph Attrs {
        rankdir=LR
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        start -> exit
    }
    """
    graph = parse(src)
    assert graph.attrs.get("rankdir") == "LR"
