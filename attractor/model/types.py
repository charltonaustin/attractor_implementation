"""Core data model types for Attractor."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StageStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    RETRY = "retry"
    FAIL = "fail"


class DiagnosticLevel(str, Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass
class Outcome:
    status: StageStatus
    notes: str = ""
    failure_reason: str = ""
    context_updates: dict[str, Any] = field(default_factory=dict)
    preferred_label: str = ""
    suggested_next_ids: list[str] = field(default_factory=list)

    def is_success(self) -> bool:
        return self.status in (StageStatus.SUCCESS, StageStatus.PARTIAL_SUCCESS)


@dataclass
class Node:
    id: str
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.attrs.get("label", self.id)

    @property
    def shape(self) -> str:
        return self.attrs.get("shape", "box")

    @property
    def type(self) -> str:
        return self.attrs.get("type", "")

    @property
    def prompt(self) -> str:
        return self.attrs.get("prompt", "")

    @property
    def max_retries(self) -> int:
        return int(self.attrs.get("max_retries", 0))

    @property
    def goal_gate(self) -> bool:
        v = self.attrs.get("goal_gate", False)
        if isinstance(v, str):
            return v.lower() == "true"
        return bool(v)

    @property
    def retry_target(self) -> str:
        return self.attrs.get("retry_target", "")

    @property
    def fallback_retry_target(self) -> str:
        return self.attrs.get("fallback_retry_target", "")

    @property
    def allow_partial(self) -> bool:
        v = self.attrs.get("allow_partial", False)
        if isinstance(v, str):
            return v.lower() == "true"
        return bool(v)

    @property
    def auto_status(self) -> bool:
        v = self.attrs.get("auto_status", False)
        if isinstance(v, str):
            return v.lower() == "true"
        return bool(v)

    @property
    def timeout(self) -> str:
        return self.attrs.get("timeout", "")

    @property
    def fidelity(self) -> str:
        return self.attrs.get("fidelity", "")

    @property
    def thread_id(self) -> str:
        return self.attrs.get("thread_id", "")

    @property
    def llm_model(self) -> str:
        return self.attrs.get("llm_model", "")

    @property
    def llm_provider(self) -> str:
        return self.attrs.get("llm_provider", "")

    @property
    def reasoning_effort(self) -> str:
        return self.attrs.get("reasoning_effort", "high")

    @property
    def css_class(self) -> str:
        return self.attrs.get("class", "")

    def is_start(self) -> bool:
        return self.shape == "Mdiamond" or self.type == "start"

    def is_exit(self) -> bool:
        return self.shape == "Msquare" or self.type == "exit"

    def is_terminal(self) -> bool:
        return self.is_exit()


@dataclass
class Edge:
    from_node: str
    to_node: str
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.attrs.get("label", "")

    @property
    def condition(self) -> str:
        return self.attrs.get("condition", "")

    @property
    def weight(self) -> int:
        return int(self.attrs.get("weight", 0))

    @property
    def loop_restart(self) -> bool:
        v = self.attrs.get("loop_restart", False)
        if isinstance(v, str):
            return v.lower() == "true"
        return bool(v)

    @property
    def fidelity(self) -> str:
        return self.attrs.get("fidelity", "")

    @property
    def thread_id(self) -> str:
        return self.attrs.get("thread_id", "")


@dataclass
class Graph:
    id: str
    attrs: dict[str, Any] = field(default_factory=dict)
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)

    @property
    def goal(self) -> str:
        return self.attrs.get("goal", "")

    @property
    def label(self) -> str:
        return self.attrs.get("label", self.id)

    @property
    def model_stylesheet(self) -> str:
        return self.attrs.get("model_stylesheet", "")

    @property
    def default_max_retry(self) -> int:
        return int(self.attrs.get("default_max_retry", 50))

    @property
    def retry_target(self) -> str:
        return self.attrs.get("retry_target", "")

    @property
    def fallback_retry_target(self) -> str:
        return self.attrs.get("fallback_retry_target", "")

    @property
    def default_fidelity(self) -> str:
        return self.attrs.get("default_fidelity", "")

    def outgoing_edges(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges if e.from_node == node_id]

    def incoming_edges(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges if e.to_node == node_id]

    def find_start_node(self) -> Node | None:
        for node in self.nodes.values():
            if node.is_start():
                return node
        for nid in ("start", "Start"):
            if nid in self.nodes:
                return self.nodes[nid]
        return None

    def find_exit_node(self) -> Node | None:
        for node in self.nodes.values():
            if node.is_exit():
                return node
        for nid in ("exit", "Exit"):
            if nid in self.nodes:
                return self.nodes[nid]
        return None


@dataclass
class Diagnostic:
    level: DiagnosticLevel
    rule: str
    message: str
    node_id: str = ""
    edge: str = ""

    def __str__(self) -> str:
        loc = f" [node={self.node_id}]" if self.node_id else ""
        loc += f" [edge={self.edge}]" if self.edge else ""
        return f"[{self.level.value.upper()}] {self.rule}: {self.message}{loc}"
