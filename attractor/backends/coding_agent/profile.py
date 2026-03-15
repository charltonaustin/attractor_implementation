"""Provider profiles: system prompt builders, API callers, and message formatters."""
from __future__ import annotations

import asyncio
import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from attractor.backends.coding_agent.environment import LocalExecutionEnvironment
from attractor.backends.coding_agent.tools import ToolRegistry, build_core_tools

if TYPE_CHECKING:
    from attractor.backends.coding_agent.session import SessionConfig, Turn


@dataclass
class ToolCallData:
    id: str
    name: str
    input: dict


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCallData] = field(default_factory=list)
    stop_reason: str = "end_turn"


class ProviderProfile(ABC):
    model: str
    tool_registry: ToolRegistry
    supports_parallel_tool_calls: bool = True

    @abstractmethod
    def build_system_prompt(self, project_docs: str | None = None) -> str:
        ...

    @abstractmethod
    async def call_api(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict],
        config: "SessionConfig",
    ) -> LLMResponse:
        ...

    @abstractmethod
    def format_messages(self, history: list["Turn"]) -> list[dict]:
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

        claude_md = Path(self._env.working_directory()) / "CLAUDE.md"
        if claude_md.exists():
            try:
                content = claude_md.read_text(encoding="utf-8")
                parts.append(f"<project_instructions>\n{content}\n</project_instructions>")
            except Exception:
                pass

        if project_docs:
            parts.append(f"<project_docs>\n{project_docs}\n</project_docs>")

        return "\n\n".join(parts)

    async def call_api(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict],
        config: "SessionConfig",
    ) -> LLMResponse:
        import anthropic

        client = anthropic.Anthropic()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 8192,
            "system": system,
            "messages": messages,
            "tools": tools,
        }

        if self.supports_parallel_tool_calls:
            kwargs["tool_choice"] = {"type": "auto"}

        if config.reasoning_effort:
            kwargs["thinking"] = {"type": "adaptive"}

        response = await asyncio.to_thread(client.messages.create, **kwargs)

        text_parts: list[str] = []
        tool_calls: list[ToolCallData] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCallData(id=block.id, name=block.name, input=block.input))

        stop_reason = "tool_use" if tool_calls else "end_turn"
        return LLMResponse(text="\n".join(text_parts), tool_calls=tool_calls, stop_reason=stop_reason)

    def format_messages(self, history: list["Turn"]) -> list[dict]:
        from attractor.backends.coding_agent.session import _build_messages
        return _build_messages(history)


class OllamaProfile(ProviderProfile):
    supports_parallel_tool_calls = False

    def __init__(
        self,
        model: str,
        env: LocalExecutionEnvironment,
        host: str = "http://localhost:11434",
    ) -> None:
        self.model = model
        self._env = env
        self._host = host
        self.tool_registry = build_core_tools(env)

    def build_system_prompt(self, project_docs: str | None = None) -> str:
        parts: list[str] = []

        parts.append(
            "You are an expert software engineer. You have access to tools that allow you to "
            "read and write files, execute shell commands, and search codebases. Use these tools "
            "to complete the task given to you. Think step by step, use tools as needed, and "
            "stop when the task is complete."
        )

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

        claude_md = Path(self._env.working_directory()) / "CLAUDE.md"
        if claude_md.exists():
            try:
                content = claude_md.read_text(encoding="utf-8")
                parts.append(f"<project_instructions>\n{content}\n</project_instructions>")
            except Exception:
                pass

        if project_docs:
            parts.append(f"<project_docs>\n{project_docs}\n</project_docs>")

        return "\n\n".join(parts)

    def tools(self) -> list[dict]:
        """Return tools in Ollama's OpenAI-compatible function-calling format."""
        ollama_tools = []
        for defn in self.tool_registry.definitions():
            ollama_tools.append({
                "type": "function",
                "function": {
                    "name": defn["name"],
                    "description": defn.get("description", ""),
                    "parameters": defn.get("input_schema", {}),
                },
            })
        return ollama_tools

    async def call_api(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict],
        config: "SessionConfig",
    ) -> LLMResponse:
        import ollama

        client = ollama.Client(host=self._host)
        full_messages = [{"role": "system", "content": system}] + messages

        response = await asyncio.to_thread(
            client.chat,
            model=self.model,
            messages=full_messages,
            tools=tools,
        )

        text = response.message.content or ""
        tool_calls: list[ToolCallData] = []
        if response.message.tool_calls:
            for i, tc in enumerate(response.message.tool_calls):
                tool_calls.append(ToolCallData(
                    id=f"tc_{i}",
                    name=tc.function.name,
                    input=dict(tc.function.arguments),
                ))

        stop_reason = "tool_use" if tool_calls else "end_turn"
        return LLMResponse(text=text, tool_calls=tool_calls, stop_reason=stop_reason)

    def format_messages(self, history: list["Turn"]) -> list[dict]:
        """Build messages in OpenAI-compatible format for Ollama."""
        from attractor.backends.coding_agent.session import (
            UserTurn, AssistantTurn, ToolResultsTurn, SteeringTurn,
        )

        messages: list[dict] = []
        for turn in history:
            if isinstance(turn, UserTurn):
                messages.append({"role": "user", "content": turn.content})
            elif isinstance(turn, AssistantTurn):
                msg: dict[str, Any] = {"role": "assistant", "content": turn.content or ""}
                if turn.tool_calls:
                    msg["tool_calls"] = [
                        {"function": {"name": tc["name"], "arguments": tc["input"]}}
                        for tc in turn.tool_calls
                    ]
                messages.append(msg)
            elif isinstance(turn, ToolResultsTurn):
                for r in turn.results:
                    messages.append({
                        "role": "tool",
                        "content": r["content"],
                        "name": r["name"],
                    })
            elif isinstance(turn, SteeringTurn):
                messages.append({"role": "user", "content": turn.content})

        return messages
