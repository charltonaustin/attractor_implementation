"""CodingAgentBackend: agentic loop backend implementing CodergenBackend."""
from __future__ import annotations

from attractor.handlers.codergen import CodergenBackend
from attractor.model.context import Context
from attractor.model.types import Node

from attractor.backends.coding_agent.environment import LocalExecutionEnvironment
from attractor.backends.coding_agent.profile import AnthropicProfile, OllamaProfile
from attractor.backends.coding_agent.session import Session, SessionConfig


class CodingAgentBackend(CodergenBackend):
    """Agentic loop backend using the Anthropic API with tool execution."""

    def __init__(
        self,
        model: str = "claude-opus-4-6",
        workdir: str | None = None,
        max_turns: int = 0,
        reasoning_effort: str | None = None,
    ) -> None:
        self.model = model
        self.workdir = workdir
        self.max_turns = max_turns
        self.reasoning_effort = reasoning_effort

    async def run(self, node: Node, prompt: str, context: Context) -> str:
        model = node.llm_model or self.model
        reasoning_effort = node.attrs.get("reasoning_effort") or self.reasoning_effort

        env = LocalExecutionEnvironment(working_dir=self.workdir)
        profile = AnthropicProfile(model=model, env=env)
        config = SessionConfig(
            max_turns=self.max_turns,
            reasoning_effort=reasoning_effort,
        )
        session = Session(provider_profile=profile, execution_env=env, config=config)
        await session.process_input(prompt)
        return session.get_final_response()


class OllamaCodingAgentBackend(CodergenBackend):
    """Agentic loop backend using a local Ollama server with tool execution."""

    def __init__(
        self,
        model: str = "deepseek-coder:33b",
        workdir: str | None = None,
        max_turns: int = 0,
        host: str = "http://localhost:11434",
    ) -> None:
        self.model = model
        self.workdir = workdir
        self.max_turns = max_turns
        self.host = host

    async def run(self, node: Node, prompt: str, context: Context) -> str:
        model = node.llm_model or self.model

        env = LocalExecutionEnvironment(working_dir=self.workdir)
        profile = OllamaProfile(model=model, env=env, host=self.host)
        config = SessionConfig(max_turns=self.max_turns)
        session = Session(provider_profile=profile, execution_env=env, config=config)
        await session.process_input(prompt)
        return session.get_final_response()
