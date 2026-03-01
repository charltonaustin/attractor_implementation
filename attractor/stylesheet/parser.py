"""Parse model_stylesheet CSS-like syntax.

Syntax:
    selector { property: value; ... }

Selectors:
    .classname          - matches nodes with that class
    #node-id            - matches a specific node ID
    [shape=box]         - matches nodes with that shape
    *                   - matches all nodes
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class StyleRule:
    selector: str
    properties: dict[str, str] = field(default_factory=dict)


class StylesheetParseError(Exception):
    pass


def parse_stylesheet(css: str) -> list[StyleRule]:
    """Parse a CSS-like model stylesheet into a list of StyleRules."""
    rules: list[StyleRule] = []
    # Remove comments
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    css = re.sub(r"//[^\n]*", "", css)

    # Match: selector { ... }
    rule_pattern = re.compile(
        r"([^{]+)\{([^}]*)\}",
        re.DOTALL,
    )
    for m in rule_pattern.finditer(css):
        selector = m.group(1).strip()
        body = m.group(2).strip()
        if not selector:
            continue
        properties = _parse_properties(body)
        rules.append(StyleRule(selector=selector, properties=properties))

    return rules


def _parse_properties(body: str) -> dict[str, str]:
    """Parse property: value; declarations."""
    props: dict[str, str] = {}
    for part in body.split(";"):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            continue
        key, _, value = part.partition(":")
        props[key.strip()] = value.strip().strip('"\'')
    return props
