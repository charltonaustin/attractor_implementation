"""Pending question store for HTTP human gates."""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any

from attractor.interviewer.base import Answer, AnswerValue, Interviewer, Question


@dataclass
class PendingQuestion:
    id: str
    pipeline_id: str
    question: Question
    answer_queue: asyncio.Queue = field(default_factory=asyncio.Queue)


class QuestionStore:
    """Stores pending questions for HTTP pipelines."""

    def __init__(self) -> None:
        self._questions: dict[str, PendingQuestion] = {}

    def create(self, pipeline_id: str, question: Question) -> PendingQuestion:
        qid = str(uuid.uuid4())
        pq = PendingQuestion(id=qid, pipeline_id=pipeline_id, question=question)
        self._questions[qid] = pq
        return pq

    def get(self, qid: str) -> PendingQuestion | None:
        return self._questions.get(qid)

    def list_pending(self, pipeline_id: str) -> list[PendingQuestion]:
        return [
            pq for pq in self._questions.values()
            if pq.pipeline_id == pipeline_id
        ]

    def remove(self, qid: str) -> None:
        self._questions.pop(qid, None)

    async def submit_answer(self, qid: str, answer_value: str) -> bool:
        pq = self._questions.get(qid)
        if not pq:
            return False
        answer = Answer(value=answer_value)
        await pq.answer_queue.put(answer)
        return True


class HttpInterviewer(Interviewer):
    """Interviewer that registers questions in the QuestionStore and waits for HTTP answers."""

    def __init__(self, pipeline_id: str, store: QuestionStore) -> None:
        self._pipeline_id = pipeline_id
        self._store = store

    async def ask(self, question: Question) -> Answer:
        pq = self._store.create(self._pipeline_id, question)
        try:
            timeout = question.timeout_seconds
            if timeout:
                answer = await asyncio.wait_for(pq.answer_queue.get(), timeout=timeout)
            else:
                answer = await pq.answer_queue.get()
            return answer
        except asyncio.TimeoutError:
            if question.default:
                return question.default
            return Answer(value=AnswerValue.TIMEOUT)
        finally:
            self._store.remove(pq.id)
