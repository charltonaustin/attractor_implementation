"""Thread-safe pipeline execution context."""
from __future__ import annotations

import asyncio
import copy
from typing import Any


class Context:
    """Thread-safe key-value store for pipeline execution state."""

    def __init__(self, initial: dict[str, Any] | None = None) -> None:
        self._data: dict[str, Any] = dict(initial or {})
        self._lock = asyncio.Lock()

    async def get_async(self, key: str, default: Any = None) -> Any:
        async with self._lock:
            return self._data.get(key, default)

    async def set_async(self, key: str, value: Any) -> None:
        async with self._lock:
            self._data[key] = value

    async def update_async(self, updates: dict[str, Any]) -> None:
        async with self._lock:
            self._data.update(updates)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def update(self, updates: dict[str, Any]) -> None:
        self._data.update(updates)

    def snapshot(self) -> dict[str, Any]:
        return dict(self._data)

    def clone(self) -> "Context":
        return Context(copy.deepcopy(self._data))

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        return f"Context({self._data!r})"
