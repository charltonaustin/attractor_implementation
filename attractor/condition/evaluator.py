"""Parse and evaluate condition expressions for edge selection.

Grammar:
    condition  ::= expr (('&&' | '||') expr)*
    expr       ::= comparison | '(' condition ')'
    comparison ::= key ('=' | '!=' | '<' | '>' | '<=' | '>=') value
    key        ::= identifier ('.' identifier)*
    value      ::= string | identifier
"""
from __future__ import annotations

import re
from typing import Any


class ConditionError(Exception):
    pass


def _normalize(val: Any) -> str:
    """Normalize a value to lowercase string for comparison."""
    if val is None:
        return ""
    return str(val).lower().strip()


def evaluate_condition(condition: str, outcome: Any, context: Any) -> bool:
    """Evaluate a condition expression.

    Args:
        condition: The condition string, e.g. "outcome=success" or "outcome!=fail && x=y"
        outcome: The current Outcome object (has .status attribute)
        context: The current Context object (has .get() method)

    Returns:
        True if the condition passes, False otherwise.
    """
    if not condition or not condition.strip():
        return True
    tokens = _tokenize_condition(condition.strip())
    result, _ = _parse_or(tokens, 0, outcome, context)
    return result


def _resolve_key(key: str, outcome: Any, context: Any) -> str:
    """Resolve a key to its current value."""
    if key == "outcome":
        if outcome is None:
            return ""
        status = getattr(outcome, "status", outcome)
        return _normalize(status.value if hasattr(status, "value") else status)
    if key == "preferred_label":
        if outcome is None:
            return ""
        return _normalize(getattr(outcome, "preferred_label", ""))
    # Look up in context
    val = context.get(key) if context else None
    return _normalize(val)


def _tokenize_condition(condition: str) -> list[str]:
    """Split condition into tokens."""
    pattern = re.compile(
        r"""
        \|\|
        | &&
        | !=
        | <=
        | >=
        | [=<>()\s]+
        | [^\s!=<>()&|]+
        """,
        re.VERBOSE,
    )
    tokens = []
    for m in pattern.finditer(condition):
        tok = m.group(0).strip()
        if tok:
            tokens.append(tok)
    return tokens


def _parse_or(tokens: list[str], pos: int, outcome: Any, context: Any) -> tuple[bool, int]:
    left, pos = _parse_and(tokens, pos, outcome, context)
    while pos < len(tokens) and tokens[pos] == "||":
        pos += 1  # consume ||
        right, pos = _parse_and(tokens, pos, outcome, context)
        left = left or right
    return left, pos


def _parse_and(tokens: list[str], pos: int, outcome: Any, context: Any) -> tuple[bool, int]:
    left, pos = _parse_expr(tokens, pos, outcome, context)
    while pos < len(tokens) and tokens[pos] == "&&":
        pos += 1  # consume &&
        right, pos = _parse_expr(tokens, pos, outcome, context)
        left = left and right
    return left, pos


def _parse_expr(tokens: list[str], pos: int, outcome: Any, context: Any) -> tuple[bool, int]:
    if pos < len(tokens) and tokens[pos] == "(":
        pos += 1  # consume (
        result, pos = _parse_or(tokens, pos, outcome, context)
        if pos < len(tokens) and tokens[pos] == ")":
            pos += 1  # consume )
        return result, pos

    # Comparison: key op value
    if pos >= len(tokens):
        return True, pos

    key = tokens[pos]
    pos += 1

    if pos >= len(tokens):
        # Bare key treated as truthiness check
        return bool(_resolve_key(key, outcome, context)), pos

    op = tokens[pos] if pos < len(tokens) else ""
    if op not in ("=", "!=", "<", ">", "<=", ">="):
        # Bare key
        return bool(_resolve_key(key, outcome, context)), pos

    pos += 1  # consume operator
    if pos >= len(tokens):
        raise ConditionError(f"Expected value after operator '{op}'")
    value = tokens[pos].strip('"\'')
    pos += 1

    lhs = _resolve_key(key, outcome, context)
    rhs = _normalize(value)

    if op == "=":
        return lhs == rhs, pos
    if op == "!=":
        return lhs != rhs, pos
    if op == "<":
        try:
            return float(lhs) < float(rhs), pos
        except ValueError:
            return lhs < rhs, pos
    if op == ">":
        try:
            return float(lhs) > float(rhs), pos
        except ValueError:
            return lhs > rhs, pos
    if op == "<=":
        try:
            return float(lhs) <= float(rhs), pos
        except ValueError:
            return lhs <= rhs, pos
    if op == ">=":
        try:
            return float(lhs) >= float(rhs), pos
        except ValueError:
            return lhs >= rhs, pos

    return True, pos
