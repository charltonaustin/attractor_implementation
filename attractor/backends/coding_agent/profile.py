"""Provider profiles: system prompt builders and model configuration."""
from __future__ import annotations

import datetime
from abc import ABC, abstractmethod
from pathlib import Path

from attractor.backends.coding_agent.environment import LocalExecutionEnvironment
from attractor.backends.coding_agent.tools import ToolRegistry, build_core_tools


class ProviderProfile(ABC):
    model: str
    tool_registry: ToolRegistry
    supports_parallel_tool_calls: bool = True

    @abstractmethod
    def build_system_prompt(self, project_docs: str | None = None) -> str:
        ...

    def tools(self) -> list[dict]:
        return self.tool_registry.definitions()


class AnthropicProfile(ProviderProfile):
    supports_parallel_tool_calls = True

    def __init__(self, model: str, env: LocalExecutionEnvironment) -> None:
        self.model = model
        self._env = env
        self.tool_registry = build_core_tools(env)

    def build_system_prompt(self, project_docs: str | None = None) -> str:
        parts: list[str] = []

        parts.append(
            "You are an expert software engineer. You have access to tools that allow you to "
            "read and write files, execute shell commands, and search codebases. Use these tools "
            "to complete the task given to you. Think step by step, use tools as needed, and "
            "stop when the task is complete."
        )

        # Environment context
        env_lines = [
            f"Working directory: {self._env.working_directory()}",
            f"Is git repository: {self._env.is_git_repo()}",
        ]
        branch = self._env.git_branch()
        if branch:
            env_lines.append(f"Git branch: {branch}")
        env_lines.append(f"Platform: {self._env.platform()}")
        env_lines.append(f"Today's date: {datetime.date.today().isoformat()}")

        env_block = "<environment>\n" + "\n".join(env_lines) + "\n</environment>"
        parts.append(env_block)

        # CLAUDE.md if present
        claude_md = Path(self._env.working_directory()) / "CLAUDE.md"
        if claude_md.exists():
            try:
                content = claude_md.read_text(encoding="utf-8")
                parts.append(f"<project_instructions>\n{content}\n</project_instructions>")
            except Exception:
                pass

        # Additional project docs passed in
        if project_docs:
            parts.append(f"<project_docs>\n{project_docs}\n</project_docs>")

        return "\n\n".join(parts)
