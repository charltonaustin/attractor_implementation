"""RecordingInterviewer: wraps another interviewer and records Q&A pairs."""
from __future__ import annotations

from attractor.interviewer.base import Answer, Interviewer, Question


class RecordingInterviewer(Interviewer):
    def __init__(self, inner: Interviewer) -> None:
        self.inner = inner
        self.recordings: list[tuple[Question, Answer]] = []

    async def ask(self, question: Question) -> Answer:
        answer = await self.inner.ask(question)
        self.recordings.append((question, answer))
        return answer

    async def inform(self, message: str, stage: str = "") -> None:
        await self.inner.inform(message, stage)
