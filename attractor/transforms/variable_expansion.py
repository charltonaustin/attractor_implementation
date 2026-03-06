"""Variable expansion for node prompts."""
from __future__ import annotations

import re
from pathlib import Path

from attractor.model.types import Graph


def expand_variables(
    text: str,
    graph: Graph,
    context: "Context | None" = None,
    logs_path: str | Path | None = None,
) -> str:
    """Expand $goal and other variables in a prompt string.

    Supports:
        $goal              - graph-level goal attribute
        $<key>             - context variable
        $<stage_id>_response - reads <logs_path>/<stage_id>/response.md from disk
    """
    if not text:
        return text

    result = text.replace("$goal", graph.goal)

    def _replace(m: re.Match) -> str:
        key = m.group(1)

        # Check context first
        if context:
            val = context.get(key)
            if val is not None:
                return str(val)

        # Check for <stage_id>_response pattern and read from disk
        if logs_path and key.endswith("_response"):
            stage_id = key[: -len("_response")]
            response_file = Path(logs_path) / stage_id / "response.md"
            if response_file.exists():
                return response_file.read_text(encoding="utf-8")

        return m.group(0)

    result = re.sub(r"\$([A-Za-z_][A-Za-z0-9_.]*)", _replace, result)

    return result
