"""Tests for the pipeline runner."""
import pytest
import asyncio
from attractor.parser.dot_parser import parse
from attractor.engine.runner import PipelineRunner
from attractor.handlers.registry import create_default_registry
from attractor.model.types import StageStatus


@pytest.mark.asyncio
async def test_simple_pipeline():
    g = parse("""
    digraph Simple {
        graph [goal="Test"]
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        task  [shape=box, label="Do task"]
        start -> task -> exit
    }
    """)
    registry = create_default_registry()
    runner = PipelineRunner(registry=registry, logs_root="/tmp/attractor_test_runs")
    outcome = await runner.run(g)
    assert outcome.status == StageStatus.SUCCESS


@pytest.mark.asyncio
async def test_branching_pipeline_success_path():
    g = parse("""
    digraph Branch {
        graph [goal="Test branching"]
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        task  [shape=box]
        gate  [shape=diamond]
        start -> task -> gate
        gate -> exit [label="Yes", condition="outcome=success"]
        gate -> task [label="No", condition="outcome=fail"]
    }
    """)
    registry = create_default_registry()
    runner = PipelineRunner(registry=registry, logs_root="/tmp/attractor_test_runs")
    outcome = await runner.run(g)
    assert outcome.status in (StageStatus.SUCCESS, StageStatus.PARTIAL_SUCCESS)


@pytest.mark.asyncio
async def test_human_gate_auto_approve():
    g = parse("""
    digraph Human {
        graph [goal="Test human gate"]
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        gate  [shape=hexagon, label="Approve?"]
        done  [shape=box, label="Done"]
        start -> gate
        gate -> done [label="[A] Approve"]
        gate -> exit [label="[S] Skip"]
        done -> exit
    }
    """)
    from attractor.interviewer.auto_approve import AutoApproveInterviewer
    registry = create_default_registry(interviewer=AutoApproveInterviewer())
    runner = PipelineRunner(registry=registry, logs_root="/tmp/attractor_test_runs")
    outcome = await runner.run(g)
    assert outcome.status == StageStatus.SUCCESS


@pytest.mark.asyncio
async def test_events_emitted():
    events = []
    g = parse("""
    digraph Events {
        graph [goal="Test events"]
        start [shape=Mdiamond]
        exit  [shape=Msquare]
        start -> exit
    }
    """)
    runner = PipelineRunner(
        logs_root="/tmp/attractor_test_runs",
        on_event=events.append,
    )
    await runner.run(g)
    event_types = [getattr(e, "event_type", "") for e in events]
    assert "pipeline_started" in event_types
    assert "pipeline_completed" in event_types
    assert "stage_started" in event_types
    assert "checkpoint_saved" in event_types
