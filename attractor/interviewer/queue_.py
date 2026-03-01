"""QueueInterviewer: reads answers from a pre-filled async queue."""
from __future__ import annotations

import asyncio

from attractor.interviewer.base import Answer, AnswerValue, Interviewer, Question


class QueueInterviewer(Interviewer):
    def __init__(self) -> None:
        self._queue: asyncio.Queue[Answer] = asyncio.Queue()

    def enqueue(self, answer: Answer) -> None:
        self._queue.put_nowait(answer)

    def enqueue_many(self, answers: list[Answer]) -> None:
        for a in answers:
            self._queue.put_nowait(a)

    async def ask(self, question: Question) -> Answer:
        if not self._queue.empty():
            return self._queue.get_nowait()
        return Answer(value=AnswerValue.SKIPPED)
