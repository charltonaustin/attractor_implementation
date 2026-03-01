"""Built-in lint rules for Attractor graphs."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque

from attractor.condition.evaluator import ConditionError, evaluate_condition
from attractor.model.types import Diagnostic, DiagnosticLevel, Graph
from attractor.stylesheet.parser import StylesheetParseError, parse_stylesheet

VALID_FIDELITY_MODES = {
    "full", "truncate", "compact",
    "summary:low", "summary:medium", "summary:high",
}

KNOWN_HANDLER_TYPES = {
    "start", "exit", "codergen", "wait.human", "conditional",
    "parallel", "parallel.fan_in", "tool", "stack.manager_loop",
}


class LintRule(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def apply(self, graph: Graph) -> list[Diagnostic]: ...


def _err(rule: str, message: str, node_id: str = "", edge: str = "") -> Diagnostic:
    return Diagnostic(DiagnosticLevel.ERROR, rule, message, node_id=node_id, edge=edge)


def _warn(rule: str, message: str, node_id: str = "", edge: str = "") -> Diagnostic:
    return Diagnostic(DiagnosticLevel.WARNING, rule, message, node_id=node_id, edge=edge)


class StartNodeRule(LintRule):
    name = "start_node"

    def apply(self, graph: Graph) -> list[Diagnostic]:
        starts = [n for n in graph.nodes.values() if n.is_start()]
        if len(starts) == 0:
            return [_err(self.name, "Pipeline must have exactly one start node (shape=Mdiamond)")]
        if len(starts) > 1:
            return [_err(self.name, f"Pipeline has {len(starts)} start nodes; expected exactly one")]
        return []


class TerminalNodeRule(LintRule):
    name = "terminal_node"

    def apply(self, graph: Graph) -> list[Diagnostic]:
        exits = [n for n in graph.nodes.values() if n.is_exit()]
        if len(exits) == 0:
            return [_err(self.name, "Pipeline must have at least one exit node (shape=Msquare)")]
        return []


class ReachabilityRule(LintRule):
    name = "reachability"

    def apply(self, graph: Graph) -> list[Diagnostic]:
        start = graph.find_start_node()
        if not start:
            return []  # StartNodeRule will catch this
        visited = set()
        queue = deque([start.id])
        while queue:
            nid = queue.popleft()
            if nid in visited:
                continue
            visited.add(nid)
            for edge in graph.outgoing_edges(nid):
                queue.append(edge.to_node)
        unreachable = [
            nid for nid in graph.nodes if nid not in visited
        ]
        return [
            _err(self.name, f"Node '{nid}' is not reachable from the start node", node_id=nid)
            for nid in unreachable
        ]


class EdgeTargetExistsRule(LintRule):
    name = "edge_target_exists"

    def apply(self, graph: Graph) -> list[Diagnostic]:
        diags = []
        for edge in graph.edges:
            if edge.to_node not in graph.nodes:
                diags.append(_err(
                    self.name,
                    f"Edge target '{edge.to_node}' does not exist",
                    edge=f"{edge.from_node}->{edge.to_node}",
                ))
            if edge.from_node not in graph.nodes:
                diags.append(_err(
                    self.name,
                    f"Edge source '{edge.from_node}' does not exist",
                    edge=f"{edge.from_node}->{edge.to_node}",
                ))
        return diags


class StartNoIncomingRule(LintRule):
    name = "start_no_incoming"

    def apply(self, graph: Graph) -> list[Diagnostic]:
        start = graph.find_start_node()
        if not start:
            return []
        incoming = graph.incoming_edges(start.id)
        if incoming:
            return [_err(
                self.name,
                f"Start node '{start.id}' must have no incoming edges",
                node_id=start.id,
            )]
        return []


class ExitNoOutgoingRule(LintRule):
    name = "exit_no_outgoing"

    def apply(self, graph: Graph) -> list[Diagnostic]:
        diags = []
        for node in graph.nodes.values():
            if node.is_exit():
                outgoing = graph.outgoing_edges(node.id)
                if outgoing:
                    diags.append(_err(
                        self.name,
                        f"Exit node '{node.id}' must have no outgoing edges",
                        node_id=node.id,
                    ))
        return diags


class ConditionSyntaxRule(LintRule):
    name = "condition_syntax"

    def apply(self, graph: Graph) -> list[Diagnostic]:
        diags = []
        for edge in graph.edges:
            cond = edge.condition
            if not cond:
                continue
            try:
                evaluate_condition(cond, None, _NullContext())
            except ConditionError as e:
                diags.append(_err(
                    self.name,
                    f"Invalid condition expression '{cond}': {e}",
                    edge=f"{edge.from_node}->{edge.to_node}",
                ))
            except Exception:
                pass  # Some errors are OK (undefined keys)
        return diags


class _NullContext:
    def get(self, key: str, default=None):
        return default


class StylesheetSyntaxRule(LintRule):
    name = "stylesheet_syntax"

    def apply(self, graph: Graph) -> list[Diagnostic]:
        css = graph.model_stylesheet
        if not css:
            return []
        try:
            parse_stylesheet(css)
        except StylesheetParseError as e:
            return [_err(self.name, f"Invalid model_stylesheet: {e}")]
        return []


class TypeKnownRule(LintRule):
    name = "type_known"

    def apply(self, graph: Graph) -> list[Diagnostic]:
        diags = []
        for node in graph.nodes.values():
            t = node.type
            if t and t not in KNOWN_HANDLER_TYPES:
                diags.append(_warn(
                    self.name,
                    f"Node '{node.id}' has unknown type '{t}'",
                    node_id=node.id,
                ))
        return diags


class FidelityValidRule(LintRule):
    name = "fidelity_valid"

    def apply(self, graph: Graph) -> list[Diagnostic]:
        diags = []
        for node in graph.nodes.values():
            f = node.fidelity
            if f and f not in VALID_FIDELITY_MODES:
                diags.append(_warn(
                    self.name,
                    f"Node '{node.id}' has invalid fidelity '{f}'",
                    node_id=node.id,
                ))
        return diags


class RetryTargetExistsRule(LintRule):
    name = "retry_target_exists"

    def apply(self, graph: Graph) -> list[Diagnostic]:
        diags = []
        for node in graph.nodes.values():
            for attr in ("retry_target", "fallback_retry_target"):
                target = node.attrs.get(attr, "")
                if target and target not in graph.nodes:
                    diags.append(_warn(
                        self.name,
                        f"Node '{node.id}' {attr}='{target}' does not exist",
                        node_id=node.id,
                    ))
        # Check graph-level
        for attr in ("retry_target", "fallback_retry_target"):
            target = graph.attrs.get(attr, "")
            if target and target not in graph.nodes:
                diags.append(_warn(
                    self.name,
                    f"Graph-level {attr}='{target}' does not exist",
                ))
        return diags


class GoalGateHasRetryRule(LintRule):
    name = "goal_gate_has_retry"

    def apply(self, graph: Graph) -> list[Diagnostic]:
        diags = []
        for node in graph.nodes.values():
            if node.goal_gate:
                has_retry = (
                    node.retry_target
                    or node.fallback_retry_target
                    or graph.retry_target
                    or graph.fallback_retry_target
                )
                if not has_retry:
                    diags.append(_warn(
                        self.name,
                        f"Node '{node.id}' has goal_gate=true but no retry_target",
                        node_id=node.id,
                    ))
        return diags


class PromptOnLlmNodesRule(LintRule):
    name = "prompt_on_llm_nodes"

    def apply(self, graph: Graph) -> list[Diagnostic]:
        from attractor.model.types import Node

        SHAPE_TO_TYPE = {
            "Mdiamond": "start",
            "Msquare": "exit",
            "box": "codergen",
            "hexagon": "wait.human",
            "diamond": "conditional",
            "component": "parallel",
            "tripleoctagon": "parallel.fan_in",
            "parallelogram": "tool",
            "house": "stack.manager_loop",
        }
        LLM_TYPES = {"codergen"}

        diags = []
        for node in graph.nodes.values():
            # Determine effective handler type
            effective = node.type
            if not effective:
                effective = SHAPE_TO_TYPE.get(node.shape, "codergen")
            if effective in LLM_TYPES:
                if not node.prompt and not node.label:
                    diags.append(_warn(
                        self.name,
                        f"LLM node '{node.id}' has no prompt or label",
                        node_id=node.id,
                    ))
        return diags


BUILT_IN_RULES: list[LintRule] = [
    StartNodeRule(),
    TerminalNodeRule(),
    ReachabilityRule(),
    EdgeTargetExistsRule(),
    StartNoIncomingRule(),
    ExitNoOutgoingRule(),
    ConditionSyntaxRule(),
    StylesheetSyntaxRule(),
    TypeKnownRule(),
    FidelityValidRule(),
    RetryTargetExistsRule(),
    GoalGateHasRetryRule(),
    PromptOnLlmNodesRule(),
]
