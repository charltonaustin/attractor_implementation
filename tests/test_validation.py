"""Tests for graph validation rules."""
import pytest
from attractor.parser.dot_parser import parse
from attractor.validation.validator import validate
from attractor.model.types import DiagnosticLevel


def _parse(src: str):
    return parse(src)


def _errors(diags):
    return [d for d in diags if d.level == DiagnosticLevel.ERROR]


def _warnings(diags):
    return [d for d in diags if d.level == DiagnosticLevel.WARNING]


def test_valid_simple_graph():
    g = _parse("""
    digraph Simple {
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        start -> exit
    }
    """)
    diags = validate(g)
    errors = _errors(diags)
    assert errors == [], f"Unexpected errors: {errors}"


def test_missing_start_node():
    g = _parse("""
    digraph NoStart {
        exit [shape=Msquare]
        task [shape=box, label="Do something"]
        task -> exit
    }
    """)
    errors = _errors(validate(g))
    assert any(d.rule == "start_node" for d in errors)


def test_missing_exit_node():
    g = _parse("""
    digraph NoExit {
        start [shape=Mdiamond]
        task  [shape=box, label="Do something"]
        start -> task
    }
    """)
    errors = _errors(validate(g))
    assert any(d.rule == "terminal_node" for d in errors)


def test_unreachable_node():
    g = _parse("""
    digraph Unreachable {
        start   [shape=Mdiamond]
        exit    [shape=Msquare]
        orphan  [shape=box]
        start -> exit
    }
    """)
    errors = _errors(validate(g))
    assert any(d.rule == "reachability" and "orphan" in d.node_id for d in errors)


def test_start_no_incoming():
    g = _parse("""
    digraph StartIncoming {
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        task  [shape=box]
        start -> task -> exit
        task -> start
    }
    """)
    errors = _errors(validate(g))
    assert any(d.rule == "start_no_incoming" for d in errors)


def test_exit_no_outgoing():
    g = _parse("""
    digraph ExitOutgoing {
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        start -> exit -> start
    }
    """)
    errors = _errors(validate(g))
    assert any(d.rule == "exit_no_outgoing" for d in errors)


def test_goal_gate_warning():
    g = _parse("""
    digraph GoalGate {
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        task  [shape=box, goal_gate=true]
        start -> task -> exit
    }
    """)
    warnings = _warnings(validate(g))
    assert any(d.rule == "goal_gate_has_retry" for d in warnings)
