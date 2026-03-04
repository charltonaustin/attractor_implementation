"""Attractor CLI: run, validate, serve commands."""
from __future__ import annotations

import asyncio
import sys

import click

from attractor.parser.dot_parser import parse_file, ParseError
from attractor.validation.validator import validate, DiagnosticLevel


@click.group()
@click.version_option()
def cli() -> None:
    """Attractor: DOT-based AI pipeline runner."""


@cli.command("run")
@click.argument("dotfile", type=click.Path(exists=True))
@click.option(
    "--backend",
    type=click.Choice(["simulate", "claude"]),
    default="simulate",
    show_default=True,
    help=(
        "LLM backend to use. 'simulate' runs a no-op backend useful for testing "
        "pipelines without calling an LLM. 'claude' invokes the claude CLI for each stage."
    ),
)
@click.option(
    "--logs-root",
    default="runs",
    show_default=True,
    help="Directory where run logs and checkpoints are stored.",
)
@click.option(
    "--interviewer",
    type=click.Choice(["auto", "console"]),
    default="console",
    show_default=True,
    help=(
        "How human-in-the-loop approval is handled. 'console' prompts interactively. "
        "'auto' approves all steps automatically."
    ),
)
@click.option(
    "--workdir",
    default=None,
    type=click.Path(exists=True, file_okay=False),
    help=(
        "Working directory passed to the LLM backend subprocess. "
        "Useful when the pipeline needs to operate on a specific project directory."
    ),
)
@click.option(
    "--venv",
    default=None,
    type=click.Path(exists=True, file_okay=False),
    help=(
        "Path to a virtual environment to activate when running tool commands "
        "(e.g. ~/dev/myproject/.venv)."
    ),
)
@click.option(
    "--dangerously-skip-permissions",
    "skip_permissions",
    is_flag=True,
    default=False,
    help=(
        "Pass --dangerously-skip-permissions to the claude CLI. "
        "Required for fully automated pipelines where interactive permission prompts would block execution."
    ),
)
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help="Resume from the last saved checkpoint instead of starting from the beginning.",
)
def run_command(
    dotfile: str,
    backend: str,
    logs_root: str,
    interviewer: str,
    workdir: str | None,
    venv: str | None,
    skip_permissions: bool,
    resume: bool,
) -> None:
    """Execute a pipeline defined in DOTFILE."""
    try:
        graph = parse_file(dotfile)
    except ParseError as e:
        click.echo(f"Parse error: {e}", err=True)
        sys.exit(1)
    except FileNotFoundError:
        click.echo(f"File not found: {dotfile}", err=True)
        sys.exit(1)

    # Build backend
    llm_backend = None
    if backend == "claude":
        from attractor.handlers.codergen import ClaudeCliBackend
        extra_args = ["--dangerously-skip-permissions"] if skip_permissions else []
        llm_backend = ClaudeCliBackend(extra_args=extra_args, workdir=workdir)

    # Build interviewer
    from attractor.interviewer.auto_approve import AutoApproveInterviewer
    from attractor.interviewer.console import ConsoleInterviewer

    if interviewer == "auto":
        iv = AutoApproveInterviewer()
    else:
        iv = ConsoleInterviewer()

    from attractor.handlers.registry import create_default_registry
    from attractor.engine.runner import PipelineRunner, PipelineError

    registry = create_default_registry(backend=llm_backend, interviewer=iv, venv=venv)
    runner = PipelineRunner(registry=registry, logs_root=logs_root, resume=resume)

    # Simple event printer
    def on_event(event: object) -> None:
        event_type = getattr(event, "event_type", "event")
        click.echo(f"[{event_type}] {_describe_event(event)}")

    runner.on_event = on_event

    click.echo(f"Running pipeline: {graph.id}")
    click.echo(f"  Goal: {graph.goal}")
    click.echo(f"  Backend: {backend}")
    click.echo("")

    try:
        outcome = asyncio.run(runner.run(graph))
        click.echo(f"\nPipeline completed with status: {outcome.status.value}")
        if outcome.notes:
            click.echo(f"  Notes: {outcome.notes}")
    except Exception as e:
        click.echo(f"\nPipeline failed: {e}", err=True)
        sys.exit(1)


@cli.command("validate")
@click.argument("dotfile", type=click.Path(exists=True))
@click.option("--strict", is_flag=True, help="Exit non-zero on warnings.")
def validate_command(dotfile: str, strict: bool) -> None:
    """Validate a pipeline DOT file."""
    try:
        graph = parse_file(dotfile)
    except ParseError as e:
        click.echo(f"Parse error: {e}", err=True)
        sys.exit(1)

    from attractor.validation.validator import validate as _validate

    diags = _validate(graph)

    if not diags:
        click.echo(f"✓ {dotfile}: No issues found")
        return

    errors = [d for d in diags if d.level == DiagnosticLevel.ERROR]
    warnings = [d for d in diags if d.level == DiagnosticLevel.WARNING]

    for d in diags:
        click.echo(str(d))

    click.echo(f"\n{len(errors)} error(s), {len(warnings)} warning(s)")

    if errors or (strict and warnings):
        sys.exit(1)


@cli.command("serve")
@click.option("--host", default="127.0.0.1", show_default=True, help="Bind host.")
@click.option("--port", default=8000, show_default=True, help="Bind port.")
@click.option("--reload", is_flag=True, help="Enable auto-reload.")
def serve_command(host: str, port: int, reload: bool) -> None:
    """Start the Attractor HTTP server."""
    try:
        import uvicorn
    except ImportError:
        click.echo("uvicorn is required: pip install uvicorn", err=True)
        sys.exit(1)

    click.echo(f"Starting Attractor server at http://{host}:{port}")
    uvicorn.run(
        "attractor.server.app:app",
        host=host,
        port=port,
        reload=reload,
    )


def _describe_event(event: object) -> str:
    """Human-readable event description."""
    et = getattr(event, "event_type", "")
    if et == "stage_started":
        return f"Starting stage: {getattr(event, 'name', '')}"
    if et == "stage_completed":
        duration = getattr(event, "duration", 0)
        return f"Completed stage: {getattr(event, 'name', '')} ({duration:.1f}s)"
    if et == "stage_failed":
        return f"Failed stage: {getattr(event, 'name', '')} - {getattr(event, 'error', '')}"
    if et == "stage_retrying":
        return (
            f"Retrying stage: {getattr(event, 'name', '')} "
            f"(attempt {getattr(event, 'attempt', '')})"
        )
    if et == "checkpoint_saved":
        return f"Checkpoint saved after: {getattr(event, 'node_id', '')}"
    if et == "pipeline_started":
        return f"Pipeline started: {getattr(event, 'name', '')}"
    if et == "pipeline_completed":
        duration = getattr(event, "duration", 0)
        return f"Pipeline completed in {duration:.1f}s"
    if et == "pipeline_failed":
        return f"Pipeline failed: {getattr(event, 'error', '')}"
    return str(event)


if __name__ == "__main__":
    cli()
