"""Unit tests for the coding agent backend (mocked Anthropic client)."""
from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import pytest

from attractor.backends.coding_agent.environment import LocalExecutionEnvironment
from attractor.backends.coding_agent.tools import ToolRegistry, truncate_output, build_core_tools
from attractor.backends.coding_agent.session import (
    Session,
    SessionConfig,
    UserTurn,
    AssistantTurn,
    ToolResultsTurn,
    _build_messages,
    _detect_loop,
)
from attractor.backends.coding_agent.profile import AnthropicProfile


# ---------------------------------------------------------------------------
# truncate_output
# ---------------------------------------------------------------------------

def test_truncate_output_no_truncation():
    text = "hello world"
    assert truncate_output(text, 100) == text


def test_truncate_output_head_tail():
    text = "A" * 100 + "B" * 100
    result = truncate_output(text, 50, mode="head_tail")
    assert "[... output truncated ...]" in result
    assert result.startswith("A")
    assert result.endswith("B" * 25)


def test_truncate_output_tail():
    text = "A" * 100 + "B" * 100
    result = truncate_output(text, 50, mode="tail")
    assert "[... output truncated" in result
    assert result.endswith("B" * 50)


# ---------------------------------------------------------------------------
# LocalExecutionEnvironment
# ---------------------------------------------------------------------------

def test_env_read_write(tmp_path):
    env = LocalExecutionEnvironment(working_dir=str(tmp_path))
    env.write_file("test.txt", "hello\nworld\n")
    assert env.file_exists("test.txt")
    content = env.read_file("test.txt")
    assert content == "hello\nworld\n"


def test_env_edit_file(tmp_path):
    env = LocalExecutionEnvironment(working_dir=str(tmp_path))
    env.write_file("test.txt", "foo bar baz")
    env.edit_file("test.txt", "bar", "qux")
    assert env.read_file("test.txt") == "foo qux baz"


def test_env_edit_file_not_found(tmp_path):
    env = LocalExecutionEnvironment(working_dir=str(tmp_path))
    env.write_file("test.txt", "no match here")
    with pytest.raises(ValueError, match="not found"):
        env.edit_file("test.txt", "missing_string", "replacement")


def test_env_exec_command(tmp_path):
    env = LocalExecutionEnvironment(working_dir=str(tmp_path))
    result = env.exec_command("echo hello")
    assert result.exit_code == 0
    assert "hello" in result.stdout


def test_env_exec_command_timeout(tmp_path):
    env = LocalExecutionEnvironment(working_dir=str(tmp_path))
    result = env.exec_command("sleep 10", timeout_ms=100)
    assert result.timed_out


def test_env_glob(tmp_path):
    (tmp_path / "a.py").write_text("x")
    (tmp_path / "b.py").write_text("y")
    (tmp_path / "c.txt").write_text("z")
    env = LocalExecutionEnvironment(working_dir=str(tmp_path))
    results = env.glob("*.py")
    assert len(results) == 2
    assert all(r.endswith(".py") for r in results)


def test_env_read_with_offset_limit(tmp_path):
    env = LocalExecutionEnvironment(working_dir=str(tmp_path))
    env.write_file("lines.txt", "line1\nline2\nline3\nline4\nline5\n")
    content = env.read_file("lines.txt", offset=2, limit=2)
    assert "line2" in content
    assert "line3" in content
    assert "line4" not in content


# ---------------------------------------------------------------------------
# _build_messages
# ---------------------------------------------------------------------------

def test_build_messages_basic():
    history = [
        UserTurn(content="hello"),
        AssistantTurn(content="world"),
    ]
    msgs = _build_messages(history)
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"


def test_build_messages_merges_consecutive_user():
    history = [
        UserTurn(content="first"),
        ToolResultsTurn(results=[{"tool_use_id": "x", "name": "t", "content": "r"}]),
    ]
    msgs = _build_messages(history)
    # Both are "user" role — should be merged
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"


def test_build_messages_tool_calls():
    history = [
        UserTurn(content="do something"),
        AssistantTurn(
            content="",
            tool_calls=[{"id": "tc1", "name": "shell", "input": {"command": "ls"}}],
        ),
        ToolResultsTurn(results=[{"tool_use_id": "tc1", "name": "shell", "content": "file.txt"}]),
    ]
    msgs = _build_messages(history)
    assert len(msgs) == 3
    assert msgs[1]["role"] == "assistant"
    assert any(b.get("type") == "tool_use" for b in msgs[1]["content"])
    assert msgs[2]["role"] == "user"
    assert any(b.get("type") == "tool_result" for b in msgs[2]["content"])


# ---------------------------------------------------------------------------
# _detect_loop
# ---------------------------------------------------------------------------

def test_detect_loop_no_loop():
    history = [
        AssistantTurn(content="", tool_calls=[{"name": "shell", "input": {"command": "ls"}}]),
        AssistantTurn(content="", tool_calls=[{"name": "shell", "input": {"command": "pwd"}}]),
    ]
    assert not _detect_loop(history)


def test_detect_loop_detects():
    tc = [{"name": "shell", "input": {"command": "ls"}}]
    history = [
        AssistantTurn(content="", tool_calls=tc),
        AssistantTurn(content="", tool_calls=tc),
        AssistantTurn(content="", tool_calls=tc),
    ]
    assert _detect_loop(history)


# ---------------------------------------------------------------------------
# Session with mocked Anthropic client
# ---------------------------------------------------------------------------

def _make_text_response(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.content = [block]
    response.stop_reason = "end_turn"
    return response


def _make_tool_response(tool_name: str, tool_id: str, tool_input: dict, text: str = "") -> MagicMock:
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = text

    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.id = tool_id
    tool_block.name = tool_name
    tool_block.input = tool_input

    response = MagicMock()
    response.content = [text_block, tool_block]
    response.stop_reason = "tool_use"
    return response


@pytest.mark.asyncio
async def test_session_natural_completion(tmp_path):
    env = LocalExecutionEnvironment(working_dir=str(tmp_path))
    profile = AnthropicProfile(model="claude-opus-4-6", env=env)
    config = SessionConfig()

    text_response = _make_text_response("Task complete.")

    with patch("anthropic.Anthropic") as MockClient:
        instance = MockClient.return_value
        instance.messages.create.return_value = text_response

        with patch("attractor.backends.coding_agent.session.asyncio.to_thread") as mock_thread:
            mock_thread.return_value = text_response

            session = Session(provider_profile=profile, execution_env=env, config=config)
            result = await session.process_input("do something")

    assert result == "Task complete."


@pytest.mark.asyncio
async def test_session_tool_execution(tmp_path):
    env = LocalExecutionEnvironment(working_dir=str(tmp_path))
    profile = AnthropicProfile(model="claude-opus-4-6", env=env)
    config = SessionConfig()

    tool_response = _make_tool_response("shell", "tc1", {"command": "echo hello"})
    final_response = _make_text_response("Done after tool.")

    api_responses = [tool_response, final_response]
    call_idx = 0

    def fake_create(**kwargs):
        nonlocal call_idx
        r = api_responses[call_idx]
        call_idx += 1
        return r

    with patch("anthropic.Anthropic") as MockAnthropicClass:
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = fake_create
        MockAnthropicClass.return_value = mock_client

        session = Session(provider_profile=profile, execution_env=env, config=config)
        result = await session.process_input("run a command")

    assert result == "Done after tool."
    assert len(session._history) >= 3  # user, assistant w/ tool call, tool result, assistant final


@pytest.mark.asyncio
async def test_session_max_turns(tmp_path):
    env = LocalExecutionEnvironment(working_dir=str(tmp_path))
    profile = AnthropicProfile(model="claude-opus-4-6", env=env)
    config = SessionConfig(max_turns=1)

    text_response = _make_text_response("Stopped.")

    with patch("attractor.backends.coding_agent.session.asyncio.to_thread") as mock_thread:
        mock_thread.return_value = text_response

        session = Session(provider_profile=profile, execution_env=env, config=config)
        result = await session.process_input("do something")

    assert result == "Stopped."
    assert session._turn_count == 1


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------

def test_tool_registry_register_and_get():
    registry = ToolRegistry()
    registry.register(
        name="my_tool",
        description="does stuff",
        parameters={"type": "object", "properties": {}, "required": []},
        executor=lambda args, env: "result",
    )
    tool = registry.get("my_tool")
    assert tool is not None
    assert tool.definition["name"] == "my_tool"
    assert tool.executor({}, None) == "result"


def test_tool_registry_definitions():
    registry = ToolRegistry()
    registry.register("t1", "desc1", {"type": "object"}, lambda a, e: "")
    registry.register("t2", "desc2", {"type": "object"}, lambda a, e: "")
    defs = registry.definitions()
    assert len(defs) == 2
    names = {d["name"] for d in defs}
    assert names == {"t1", "t2"}


# ---------------------------------------------------------------------------
# Core tools integration (against real filesystem)
# ---------------------------------------------------------------------------

def test_core_tools_read_write(tmp_path):
    env = LocalExecutionEnvironment(working_dir=str(tmp_path))
    registry = build_core_tools(env)

    write_tool = registry.get("write_file")
    result = write_tool.executor({"path": "hello.txt", "content": "hello!"}, env)
    assert "Successfully wrote" in result

    read_tool = registry.get("read_file")
    content = read_tool.executor({"path": "hello.txt"}, env)
    assert content == "hello!"


def test_core_tools_shell(tmp_path):
    env = LocalExecutionEnvironment(working_dir=str(tmp_path))
    registry = build_core_tools(env)
    shell_tool = registry.get("shell")
    result = shell_tool.executor({"command": "echo hi"}, env)
    assert "hi" in result


def test_core_tools_glob(tmp_path):
    (tmp_path / "foo.py").write_text("")
    (tmp_path / "bar.py").write_text("")
    env = LocalExecutionEnvironment(working_dir=str(tmp_path))
    registry = build_core_tools(env)
    glob_tool = registry.get("glob")
    result = glob_tool.executor({"pattern": "*.py"}, env)
    assert "foo.py" in result
    assert "bar.py" in result
