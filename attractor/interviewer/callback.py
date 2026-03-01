"""CallbackInterviewer: delegates to a provided async/sync callback."""
from __future__ import annotations

import asyncio
from typing import Callable

from attractor.interviewer.base import Answer, Interviewer, Question


class CallbackInterviewer(Interviewer):
    def __init__(self, callback: Callable[[Question], Answer | "Awaitable[Answer]"]) -> None:
        self._callback = callback

    async def ask(self, question: Question) -> Answer:
        result = self._callback(question)
        if asyncio.iscoroutine(result):
            return await result
        return result
