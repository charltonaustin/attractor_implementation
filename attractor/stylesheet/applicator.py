"""Apply stylesheet rules to graph nodes."""
from __future__ import annotations

from attractor.model.types import Graph, Node
from attractor.stylesheet.parser import StyleRule, parse_stylesheet


def _selector_matches(selector: str, node: Node) -> bool:
    """Check if a CSS-like selector matches a node."""
    selector = selector.strip()
    if selector == "*":
        return True
    # Class selector: .classname
    if selector.startswith("."):
        cls_name = selector[1:]
        node_classes = [c.strip() for c in node.css_class.split(",") if c.strip()]
        return cls_name in node_classes
    # ID selector: #node-id
    if selector.startswith("#"):
        return node.id == selector[1:]
    # Attribute selector: [shape=box]
    if selector.startswith("[") and selector.endswith("]"):
        inner = selector[1:-1]
        if "=" in inner:
            key, _, val = inner.partition("=")
            return str(node.attrs.get(key.strip(), "")) == val.strip().strip('"\'')
        else:
            return inner.strip() in node.attrs
    # Type selector: shape name or type name
    if selector == node.shape or selector == node.type:
        return True
    return False


def apply_stylesheet(graph: Graph, rules: list[StyleRule]) -> None:
    """Apply stylesheet rules to all nodes in the graph, in order.

    Later rules (lower in the stylesheet) can override earlier ones.
    Node-level explicit attributes take precedence over stylesheet values.
    """
    for node in graph.nodes.values():
        # Collect all matching rules (in order)
        for rule in rules:
            # Handle compound selectors separated by comma
            selectors = [s.strip() for s in rule.selector.split(",")]
            for sel in selectors:
                if _selector_matches(sel, node):
                    # Only apply if the node doesn't already have an explicit value
                    for prop, val in rule.properties.items():
                        # Stylesheet properties map to node attrs
                        # But only if not already explicitly set
                        if prop not in node.attrs:
                            node.attrs[prop] = val
                    break


def apply_stylesheet_from_graph(graph: Graph) -> None:
    """Parse the graph's model_stylesheet attribute and apply it."""
    css = graph.model_stylesheet
    if not css:
        return
    rules = parse_stylesheet(css)
    apply_stylesheet(graph, rules)
