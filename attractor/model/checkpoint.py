"""Checkpoint save/load for pipeline resume support."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class Checkpoint:
    context_snapshot: dict[str, Any]
    current_node_id: str
    completed_nodes: list[str]
    node_outcomes: dict[str, str]
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Checkpoint":
        return cls(
            context_snapshot=data.get("context_snapshot", {}),
            current_node_id=data.get("current_node_id", ""),
            completed_nodes=data.get("completed_nodes", []),
            node_outcomes=data.get("node_outcomes", {}),
            timestamp=data.get("timestamp", ""),
        )


CHECKPOINT_FILENAME = "checkpoint.json"


def save_checkpoint(checkpoint: Checkpoint, logs_root: str | Path) -> None:
    path = Path(logs_root) / CHECKPOINT_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(checkpoint.to_dict(), f, indent=2, default=str)


def load_checkpoint(logs_root: str | Path) -> Checkpoint | None:
    path = Path(logs_root) / CHECKPOINT_FILENAME
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return Checkpoint.from_dict(data)
