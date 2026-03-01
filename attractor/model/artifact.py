"""ArtifactStore: memory + file-backed artifact storage."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any


class ArtifactStore:
    """Stores named artifacts either in memory or on disk."""

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self._memory: dict[str, Any] = {}
        self._base_dir = Path(base_dir) if base_dir else None
        self._lock = asyncio.Lock()

    async def put(self, key: str, value: Any, persist: bool = False) -> None:
        async with self._lock:
            self._memory[key] = value
            if persist and self._base_dir:
                self._base_dir.mkdir(parents=True, exist_ok=True)
                path = self._base_dir / key
                path.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(value, bytes):
                    path.write_bytes(value)
                else:
                    path.write_text(str(value), encoding="utf-8")

    async def get(self, key: str) -> Any | None:
        async with self._lock:
            if key in self._memory:
                return self._memory[key]
            if self._base_dir:
                path = self._base_dir / key
                if path.exists():
                    try:
                        return path.read_text(encoding="utf-8")
                    except Exception:
                        return path.read_bytes()
            return None

    async def list_keys(self) -> list[str]:
        async with self._lock:
            keys = set(self._memory.keys())
            if self._base_dir and self._base_dir.exists():
                for p in self._base_dir.rglob("*"):
                    if p.is_file():
                        keys.add(str(p.relative_to(self._base_dir)))
            return sorted(keys)

    def put_sync(self, key: str, value: Any) -> None:
        self._memory[key] = value

    def get_sync(self, key: str) -> Any | None:
        return self._memory.get(key)
