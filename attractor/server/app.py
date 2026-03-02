"""FastAPI HTTP server for Attractor."""
from __future__ import annotations

import asyncio
import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

from attractor.engine.runner import PipelineRunner
from attractor.handlers.codergen import ClaudeCliBackend
from attractor.handlers.registry import create_default_registry
from attractor.interviewer.auto_approve import AutoApproveInterviewer
from attractor.parser.dot_parser import parse, ParseError
from attractor.server.events import BaseEvent
from attractor.server.question_store import HttpInterviewer, QuestionStore
from attractor.validation.validator import validate

app = FastAPI(title="Attractor", version="0.1.0")

# In-memory pipeline store
_pipelines: dict[str, dict[str, Any]] = {}
_question_store = QuestionStore()


@app.post("/pipelines", status_code=201)
async def create_pipeline(
    request: Request,
    backend: str = Query(default="simulate", pattern="^(simulate|claude)$"),
) -> JSONResponse:
    """Submit a DOT source and start execution. Returns pipeline ID."""
    body = await request.body()
    dot_source = body.decode("utf-8")

    try:
        graph = parse(dot_source)
    except ParseError as e:
        raise HTTPException(status_code=422, detail=f"Parse error: {e}")

    pipeline_id = str(uuid.uuid4())
    event_queue: asyncio.Queue[BaseEvent] = asyncio.Queue(maxsize=1000)

    llm_backend = ClaudeCliBackend() if backend == "claude" else None
    interviewer = HttpInterviewer(pipeline_id, _question_store)
    registry = create_default_registry(backend=llm_backend, interviewer=interviewer)
    runner = PipelineRunner(
        registry=registry,
        logs_root=f"runs/{pipeline_id}",
        event_queue=event_queue,
    )

    _pipelines[pipeline_id] = {
        "id": pipeline_id,
        "graph_id": graph.id,
        "status": "running",
        "backend": backend,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "event_queue": event_queue,
        "runner": runner,
        "graph": graph,
    }

    # Start pipeline in background
    async def _run() -> None:
        try:
            await runner.run(graph)
            _pipelines[pipeline_id]["status"] = "completed"
        except Exception as e:
            _pipelines[pipeline_id]["status"] = "failed"
            _pipelines[pipeline_id]["error"] = str(e)

    asyncio.create_task(_run())

    return JSONResponse(
        {"id": pipeline_id, "status": "running", "backend": backend},
        status_code=201,
    )


@app.get("/pipelines/{pipeline_id}")
async def get_pipeline(pipeline_id: str) -> JSONResponse:
    """Get pipeline status and progress."""
    p = _pipelines.get(pipeline_id)
    if not p:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    return JSONResponse({
        "id": p["id"],
        "graph_id": p["graph_id"],
        "status": p["status"],
        "backend": p["backend"],
        "created_at": p["created_at"],
        "error": p.get("error", ""),
    })


@app.get("/pipelines/{pipeline_id}/events")
async def stream_events(pipeline_id: str) -> StreamingResponse:
    """SSE stream of pipeline events."""
    p = _pipelines.get(pipeline_id)
    if not p:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    event_queue: asyncio.Queue = p["event_queue"]

    async def _generate():
        while True:
            try:
                event = await asyncio.wait_for(event_queue.get(), timeout=30.0)
                if hasattr(event, "to_sse"):
                    yield event.to_sse()
                else:
                    yield f"data: {json.dumps(str(event))}\n\n"
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
            except Exception:
                break

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/pipelines/{pipeline_id}/cancel")
async def cancel_pipeline(pipeline_id: str) -> JSONResponse:
    """Cancel a running pipeline."""
    p = _pipelines.get(pipeline_id)
    if not p:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    _pipelines[pipeline_id]["status"] = "cancelled"
    return JSONResponse({"id": pipeline_id, "status": "cancelled"})


@app.get("/pipelines/{pipeline_id}/graph")
async def get_graph_svg(pipeline_id: str) -> Any:
    """Get rendered SVG graph visualization."""
    p = _pipelines.get(pipeline_id)
    if not p:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    # Find the .dot source file if available
    dot_source = getattr(p.get("graph"), "id", "unknown")

    # Try to render with graphviz
    try:
        # We need the original DOT source - store it at creation time
        from fastapi.responses import Response
        dot_text = p.get("dot_source", f'digraph {dot_source} {{}}')
        proc = subprocess.run(
            ["dot", "-Tsvg"],
            input=dot_text.encode(),
            capture_output=True,
            timeout=10,
        )
        if proc.returncode == 0:
            return Response(proc.stdout, media_type="image/svg+xml")
    except Exception:
        pass

    raise HTTPException(status_code=503, detail="Graphviz not available")


@app.get("/pipelines/{pipeline_id}/questions")
async def get_questions(pipeline_id: str) -> JSONResponse:
    """Get pending human interaction questions."""
    p = _pipelines.get(pipeline_id)
    if not p:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    pending = _question_store.list_pending(pipeline_id)
    return JSONResponse({
        "questions": [
            {
                "id": pq.id,
                "text": pq.question.text,
                "type": pq.question.type.value,
                "options": [
                    {"key": o.key, "label": o.label}
                    for o in pq.question.options
                ],
                "stage": pq.question.stage,
            }
            for pq in pending
        ]
    })


@app.post("/pipelines/{pipeline_id}/questions/{question_id}/answer")
async def answer_question(
    pipeline_id: str, question_id: str, request: Request
) -> JSONResponse:
    """Submit an answer to a pending question."""
    p = _pipelines.get(pipeline_id)
    if not p:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    body = await request.json()
    answer_value = body.get("answer", "")

    ok = await _question_store.submit_answer(question_id, answer_value)
    if not ok:
        raise HTTPException(status_code=404, detail="Question not found")

    return JSONResponse({"status": "answered", "question_id": question_id})


@app.get("/pipelines/{pipeline_id}/checkpoint")
async def get_checkpoint(pipeline_id: str) -> JSONResponse:
    """Get current checkpoint state."""
    checkpoint_path = Path(f"runs/{pipeline_id}/checkpoint.json")
    if not checkpoint_path.exists():
        raise HTTPException(status_code=404, detail="No checkpoint found")

    with open(checkpoint_path, encoding="utf-8") as f:
        data = json.load(f)
    return JSONResponse(data)


@app.get("/pipelines/{pipeline_id}/context")
async def get_context(pipeline_id: str) -> JSONResponse:
    """Get current context key-value store."""
    p = _pipelines.get(pipeline_id)
    if not p:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    runner: PipelineRunner = p.get("runner")
    # Read from checkpoint as approximation
    checkpoint_path = Path(f"runs/{pipeline_id}/checkpoint.json")
    if checkpoint_path.exists():
        with open(checkpoint_path, encoding="utf-8") as f:
            data = json.load(f)
        return JSONResponse(data.get("context_snapshot", {}))

    return JSONResponse({})
