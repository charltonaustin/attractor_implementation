"""Graph validation entry point."""
from __future__ import annotations

from attractor.model.types import Diagnostic, DiagnosticLevel, Graph
from attractor.validation.rules import BUILT_IN_RULES, LintRule


class ValidationError(Exception):
    def __init__(self, diagnostics: list[Diagnostic]) -> None:
        self.diagnostics = diagnostics
        errors = [str(d) for d in diagnostics if d.level == DiagnosticLevel.ERROR]
        super().__init__("\n".join(errors))


def validate(graph: Graph, extra_rules: list[LintRule] | None = None) -> list[Diagnostic]:
    """Run all lint rules against the graph. Returns all diagnostics."""
    rules = list(BUILT_IN_RULES)
    if extra_rules:
        rules.extend(extra_rules)
    diags: list[Diagnostic] = []
    for rule in rules:
        diags.extend(rule.apply(graph))
    return diags


def validate_or_raise(
    graph: Graph, extra_rules: list[LintRule] | None = None
) -> list[Diagnostic]:
    """Run validation. Raises ValidationError if any errors are found."""
    diags = validate(graph, extra_rules)
    errors = [d for d in diags if d.level == DiagnosticLevel.ERROR]
    if errors:
        raise ValidationError(errors)
    return diags
