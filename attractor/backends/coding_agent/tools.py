"""ToolRegistry and Anthropic core tools backed by LocalExecutionEnvironment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from attractor.backends.coding_agent.environment import LocalExecutionEnvironment


@dataclass
class RegisteredTool:
    definition: dict
    executor: Callable  # (arguments: dict, env: LocalExecutionEnvironment) -> str


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict,
        executor: Callable,
    ) -> None:
        self._tools[name] = RegisteredTool(
            definition={
                "name": name,
                "description": description,
                "input_schema": parameters,
            },
            executor=executor,
        )

    def get(self, name: str) -> RegisteredTool | None:
        return self._tools.get(name)

    def definitions(self) -> list[dict]:
        return [t.definition for t in self._tools.values()]


def truncate_output(output: str, max_chars: int, mode: str = "head_tail") -> str:
    """Truncate output to max_chars. mode: 'head_tail' or 'tail'."""
    if len(output) <= max_chars:
        return output

    if mode == "tail":
        truncated = output[-max_chars:]
        lines = truncated.splitlines()
        marker = f"[... output truncated, showing last {len(lines)} lines ...]\n"
        return marker + truncated

    # head_tail: show first half and last half
    half = max_chars // 2
    head = output[:half]
    tail = output[-half:]
    marker = "\n[... output truncated ...]\n"
    return head + marker + tail


def _truncate_lines(output: str, max_lines: int) -> str:
    lines = output.splitlines(keepends=True)
    if len(lines) <= max_lines:
        return output
    kept = lines[:max_lines]
    return "".join(kept) + f"\n[... truncated at {max_lines} lines ...]"


def build_core_tools(env: LocalExecutionEnvironment) -> ToolRegistry:
    registry = ToolRegistry()

    # read_file
    def exec_read_file(args: dict, _env: LocalExecutionEnvironment) -> str:
        path = args["path"]
        offset = args.get("offset")
        limit = args.get("limit")
        try:
            content = _env.read_file(path, offset=offset, limit=limit)
        except FileNotFoundError:
            return f"Error: file not found: {path}"
        except Exception as exc:
            return f"Error reading file: {exc}"
        return truncate_output(content, 50_000, mode="head_tail")

    registry.register(
        name="read_file",
        description="Read the contents of a file at the given path.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to read"},
                "offset": {"type": "integer", "description": "Line number to start reading from (1-indexed)"},
                "limit": {"type": "integer", "description": "Maximum number of lines to read"},
            },
            "required": ["path"],
        },
        executor=exec_read_file,
    )

    # write_file
    def exec_write_file(args: dict, _env: LocalExecutionEnvironment) -> str:
        path = args["path"]
        content = args["content"]
        try:
            _env.write_file(path, content)
        except Exception as exc:
            return f"Error writing file: {exc}"
        result = f"Successfully wrote {len(content)} characters to {path}"
        return truncate_output(result, 1_000, mode="tail")

    registry.register(
        name="write_file",
        description="Write content to a file, creating it or overwriting it.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to write"},
                "content": {"type": "string", "description": "Content to write to the file"},
            },
            "required": ["path", "content"],
        },
        executor=exec_write_file,
    )

    # edit_file
    def exec_edit_file(args: dict, _env: LocalExecutionEnvironment) -> str:
        path = args["path"]
        old_string = args["old_string"]
        new_string = args["new_string"]
        try:
            _env.edit_file(path, old_string, new_string)
        except ValueError as exc:
            return f"Error: {exc}"
        except Exception as exc:
            return f"Error editing file: {exc}"
        result = f"Successfully edited {path}"
        return truncate_output(result, 10_000, mode="tail")

    registry.register(
        name="edit_file",
        description="Replace an exact string in a file with a new string.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to edit"},
                "old_string": {"type": "string", "description": "Exact string to find and replace"},
                "new_string": {"type": "string", "description": "String to replace it with"},
            },
            "required": ["path", "old_string", "new_string"],
        },
        executor=exec_edit_file,
    )

    # shell
    def exec_shell(args: dict, _env: LocalExecutionEnvironment) -> str:
        command = args["command"]
        timeout_ms = args.get("timeout_ms", 30_000)
        result = _env.exec_command(command, timeout_ms=timeout_ms)
        output_parts = []
        if result.stdout:
            output_parts.append(result.stdout)
        if result.stderr:
            output_parts.append(f"[stderr]\n{result.stderr}")
        if result.timed_out:
            output_parts.append("[command timed out]")
        else:
            output_parts.append(f"[exit code: {result.exit_code}]")
        combined = "\n".join(output_parts)
        combined = _truncate_lines(combined, 256)
        return truncate_output(combined, 30_000, mode="head_tail")

    registry.register(
        name="shell",
        description="Execute a shell command and return its output.",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
                "timeout_ms": {"type": "integer", "description": "Timeout in milliseconds (default 30000)"},
            },
            "required": ["command"],
        },
        executor=exec_shell,
    )

    # grep
    def exec_grep(args: dict, _env: LocalExecutionEnvironment) -> str:
        pattern = args["pattern"]
        path = args["path"]
        case_insensitive = args.get("case_insensitive", False)
        options: dict[str, Any] = {}
        if case_insensitive:
            options["case_insensitive"] = True
        result = _env.grep(pattern, path, options=options)
        result = _truncate_lines(result, 200)
        return truncate_output(result, 20_000, mode="tail")

    registry.register(
        name="grep",
        description="Search for a pattern in a file or directory using grep.",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regular expression pattern to search for"},
                "path": {"type": "string", "description": "File or directory to search in"},
                "case_insensitive": {"type": "boolean", "description": "Case-insensitive search"},
            },
            "required": ["pattern", "path"],
        },
        executor=exec_grep,
    )

    # glob
    def exec_glob(args: dict, _env: LocalExecutionEnvironment) -> str:
        pattern = args["pattern"]
        path = args.get("path")
        results = _env.glob(pattern, path=path)
        output = "\n".join(results)
        output = _truncate_lines(output, 500)
        return truncate_output(output, 20_000, mode="tail")

    registry.register(
        name="glob",
        description="Find files matching a glob pattern.",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern to match files against"},
                "path": {"type": "string", "description": "Directory to search in (default: working directory)"},
            },
            "required": ["pattern"],
        },
        executor=exec_glob,
    )

    return registry
