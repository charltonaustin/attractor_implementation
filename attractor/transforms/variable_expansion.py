"""Variable expansion for node prompts."""
from __future__ import annotations

from attractor.model.types import Graph


def expand_variables(text: str, graph: Graph, context: "Context | None" = None) -> str:
    """Expand $goal and other variables in a prompt string.

    Currently supports:
        $goal  - graph-level goal attribute
    """
    if not text:
        return text

    result = text.replace("$goal", graph.goal)

    # Expand context variables $key -> context.get(key)
    if context:
        import re
        def _replace(m: re.Match) -> str:
            key = m.group(1)
            val = context.get(key)
            return str(val) if val is not None else m.group(0)
        result = re.sub(r"\$([A-Za-z_][A-Za-z0-9_.]*)", _replace, result)

    return result
