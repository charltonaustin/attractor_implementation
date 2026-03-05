"""Agentic loop session: turn types, SessionConfig, and process_input."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from attractor.backends.coding_agent.environment import LocalExecutionEnvironment
from attractor.backends.coding_agent.profile import ProviderProfile


@dataclass
class UserTurn:
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class AssistantTurn:
    content: str
    tool_calls: list[dict] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ToolResultsTurn:
    results: list[dict]  # list of {tool_use_id, name, content}
    timestamp: float = field(default_factory=time.time)


@dataclass
class SteeringTurn:
    content: str
    timestamp: float = field(default_factory=time.time)


Turn = UserTurn | AssistantTurn | ToolResultsTurn | SteeringTurn


@dataclass
class SessionConfig:
    max_turns: int = 0                      # 0 = unlimited
    max_tool_rounds_per_input: int = 0      # 0 = unlimited
    reasoning_effort: str | None = None
    enable_loop_detection: bool = True
    loop_detection_window: int = 10


class Session:
    def __init__(
        self,
        provider_profile: ProviderProfile,
        execution_env: LocalExecutionEnvironment,
        config: SessionConfig | None = None,
    ) -> None:
        self._profile = provider_profile
        self._env = execution_env
        self._config = config or SessionConfig()
        self._history: list[Turn] = []
        self._turn_count = 0

    def get_final_response(self) -> str:
        for turn in reversed(self._history):
            if isinstance(turn, AssistantTurn):
                return turn.content
        return ""

    async def process_input(self, user_input: str) -> str:
        import anthropic

        self._history.append(UserTurn(content=user_input))

        client = anthropic.Anthropic()
        system_prompt = self._profile.build_system_prompt()
        tool_definitions = self._profile.tools()
        tool_round = 0

        while True:
            if self._config.max_turns > 0 and self._turn_count >= self._config.max_turns:
                break

            messages = _build_messages(self._history)

            kwargs: dict[str, Any] = {
                "model": self._profile.model,
                "max_tokens": 8192,
                "system": system_prompt,
                "messages": messages,
                "tools": tool_definitions,
            }

            if self._profile.supports_parallel_tool_calls:
                kwargs["tool_choice"] = {"type": "auto"}

            if self._config.reasoning_effort:
                # Map effort to budget tokens; skip if model doesn't support it
                budget_map = {"low": 1024, "medium": 4096, "high": 10000}
                budget = budget_map.get(self._config.reasoning_effort.lower())
                if budget:
                    kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}

            response = await asyncio.to_thread(
                client.messages.create, **kwargs
            )

            self._turn_count += 1

            # Extract text and tool calls from response
            text_parts: list[str] = []
            tool_use_blocks: list[Any] = []
            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_use_blocks.append(block)

            assistant_text = "\n".join(text_parts)
            tool_calls = [
                {"id": b.id, "name": b.name, "input": b.input}
                for b in tool_use_blocks
            ]
            self._history.append(AssistantTurn(content=assistant_text, tool_calls=tool_calls))

            # Natural completion: no tool calls
            if not tool_use_blocks or response.stop_reason == "end_turn":
                if not tool_use_blocks:
                    break

            # Check tool round limit
            if (
                self._config.max_tool_rounds_per_input > 0
                and tool_round >= self._config.max_tool_rounds_per_input
            ):
                break

            # Execute tool calls
            tool_results = await _execute_tools_parallel(tool_use_blocks, self._profile, self._env)
            self._history.append(ToolResultsTurn(results=tool_results))
            tool_round += 1

            # Loop detection
            if self._config.enable_loop_detection and _detect_loop(
                self._history, window=self._config.loop_detection_window
            ):
                self._history.append(
                    SteeringTurn(
                        content="[Loop detected: repeated identical tool calls. Stopping to avoid infinite loop.]"
                    )
                )
                break

        return self.get_final_response()


async def _execute_tools_parallel(
    tool_use_blocks: list[Any],
    profile: ProviderProfile,
    env: LocalExecutionEnvironment,
) -> list[dict]:
    async def run_one(block: Any) -> dict:
        tool = profile.tool_registry.get(block.name)
        if tool is None:
            content = f"Error: unknown tool '{block.name}'"
        else:
            try:
                content = await asyncio.to_thread(tool.executor, block.input, env)
            except Exception as exc:
                content = f"Error executing tool '{block.name}': {exc}"
        return {"tool_use_id": block.id, "name": block.name, "content": str(content)}

    return list(await asyncio.gather(*[run_one(b) for b in tool_use_blocks]))


def _build_messages(history: list[Turn]) -> list[dict]:
    """Convert turn history to Anthropic messages format with strict user/assistant alternation."""
    raw: list[dict] = []

    for turn in history:
        if isinstance(turn, UserTurn):
            raw.append({"role": "user", "content": turn.content})
        elif isinstance(turn, AssistantTurn):
            content: list[Any] = []
            if turn.content:
                content.append({"type": "text", "text": turn.content})
            for tc in turn.tool_calls:
                content.append({
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["name"],
                    "input": tc["input"],
                })
            raw.append({"role": "assistant", "content": content or [{"type": "text", "text": ""}]})
        elif isinstance(turn, ToolResultsTurn):
            tool_result_blocks = [
                {
                    "type": "tool_result",
                    "tool_use_id": r["tool_use_id"],
                    "content": r["content"],
                }
                for r in turn.results
            ]
            raw.append({"role": "user", "content": tool_result_blocks})
        elif isinstance(turn, SteeringTurn):
            raw.append({"role": "user", "content": turn.content})

    # Merge consecutive same-role messages
    merged: list[dict] = []
    for msg in raw:
        if merged and merged[-1]["role"] == msg["role"]:
            prev = merged[-1]
            # Normalize both to list form
            if isinstance(prev["content"], str):
                prev["content"] = [{"type": "text", "text": prev["content"]}]
            if isinstance(msg["content"], str):
                msg_content: list[Any] = [{"type": "text", "text": msg["content"]}]
            else:
                msg_content = msg["content"]
            prev["content"].extend(msg_content)
        else:
            # Deep-copy-ish: use list for content to allow future merges
            entry = {"role": msg["role"], "content": msg["content"]}
            merged.append(entry)

    return merged


def _detect_loop(history: list[Turn], window: int = 10) -> bool:
    """Detect repeated identical tool call patterns in recent history."""
    tool_round_signatures: list[frozenset] = []
    for turn in history:
        if isinstance(turn, AssistantTurn) and turn.tool_calls:
            sig = frozenset((tc["name"], str(tc["input"])) for tc in turn.tool_calls)
            tool_round_signatures.append(sig)

    recent = tool_round_signatures[-window:]
    if len(recent) < 3:
        return False

    # Check if last 3 signatures are identical
    return recent[-1] == recent[-2] == recent[-3]
