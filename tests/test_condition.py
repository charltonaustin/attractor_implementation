"""Tests for the condition expression evaluator."""
import pytest
from attractor.condition.evaluator import evaluate_condition
from attractor.model.types import Outcome, StageStatus
from attractor.model.context import Context


def _ctx(**kwargs):
    return Context(kwargs)


def _outcome(status="success", preferred_label=""):
    return Outcome(status=StageStatus(status), preferred_label=preferred_label)


def test_simple_equality():
    assert evaluate_condition("outcome=success", _outcome("success"), _ctx()) is True
    assert evaluate_condition("outcome=fail", _outcome("success"), _ctx()) is False


def test_inequality():
    assert evaluate_condition("outcome!=fail", _outcome("success"), _ctx()) is True
    assert evaluate_condition("outcome!=success", _outcome("success"), _ctx()) is False


def test_and_operator():
    ctx = _ctx(x="yes")
    o = _outcome("success")
    assert evaluate_condition("outcome=success && x=yes", o, ctx) is True
    assert evaluate_condition("outcome=success && x=no", o, ctx) is False


def test_or_operator():
    ctx = _ctx()
    o = _outcome("fail")
    assert evaluate_condition("outcome=success || outcome=fail", o, ctx) is True
    assert evaluate_condition("outcome=retry || outcome=partial_success", o, ctx) is False


def test_context_key_lookup():
    ctx = _ctx(some_key="hello")
    o = _outcome("success")
    assert evaluate_condition("some_key=hello", o, ctx) is True
    assert evaluate_condition("some_key=world", o, ctx) is False


def test_empty_condition():
    assert evaluate_condition("", _outcome("success"), _ctx()) is True
    assert evaluate_condition("  ", _outcome("success"), _ctx()) is True


def test_parentheses():
    ctx = _ctx(x="1")
    o = _outcome("success")
    assert evaluate_condition("(outcome=success && x=1)", o, ctx) is True
    assert evaluate_condition("(outcome=fail || x=1)", o, ctx) is True
