"""Event types for pipeline observability and SSE streaming."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class BaseEvent:
    event_type: str
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = time.time()

    def to_sse(self) -> str:
        data = asdict(self)
        return f"event: {self.event_type}\ndata: {json.dumps(data, default=str)}\n\n"


@dataclass
class PipelineStartedEvent(BaseEvent):
    name: str = ""
    id: str = ""
    event_type: str = "pipeline_started"


@dataclass
class PipelineCompletedEvent(BaseEvent):
    duration: float = 0.0
    artifact_count: int = 0
    event_type: str = "pipeline_completed"


@dataclass
class PipelineFailedEvent(BaseEvent):
    error: str = ""
    duration: float = 0.0
    event_type: str = "pipeline_failed"


@dataclass
class StageStartedEvent(BaseEvent):
    name: str = ""
    index: int = 0
    event_type: str = "stage_started"


@dataclass
class StageCompletedEvent(BaseEvent):
    name: str = ""
    index: int = 0
    duration: float = 0.0
    event_type: str = "stage_completed"


@dataclass
class StageFailedEvent(BaseEvent):
    name: str = ""
    index: int = 0
    error: str = ""
    will_retry: bool = False
    event_type: str = "stage_failed"


@dataclass
class StageRetryingEvent(BaseEvent):
    name: str = ""
    index: int = 0
    attempt: int = 0
    delay: float = 0.0
    event_type: str = "stage_retrying"


@dataclass
class CheckpointSavedEvent(BaseEvent):
    node_id: str = ""
    event_type: str = "checkpoint_saved"


@dataclass
class InterviewStartedEvent(BaseEvent):
    question: str = ""
    stage: str = ""
    event_type: str = "interview_started"


@dataclass
class InterviewCompletedEvent(BaseEvent):
    question: str = ""
    answer: str = ""
    duration: float = 0.0
    event_type: str = "interview_completed"
