"""ConsoleInterviewer: reads from stdin, prints to stdout."""
from __future__ import annotations

import asyncio
import sys

from attractor.interviewer.base import (
    Answer, AnswerValue, Interviewer, Option, Question, QuestionType,
)


class ConsoleInterviewer(Interviewer):
    def __init__(self, timeout: float | None = None) -> None:
        self.timeout = timeout

    async def ask(self, question: Question) -> Answer:
        timeout = question.timeout_seconds or self.timeout
        print(f"\n[?] {question.text}", flush=True)

        if question.type == QuestionType.MULTIPLE_CHOICE:
            for opt in question.options:
                print(f"  [{opt.key}] {opt.label}", flush=True)
            response = await self._read_line("Select: ", timeout)
            if response is None:
                if question.default:
                    return question.default
                return Answer(value=AnswerValue.TIMEOUT)
            return _find_option(response.strip(), question.options)

        if question.type in (QuestionType.YES_NO, QuestionType.CONFIRMATION):
            response = await self._read_line("[Y/N]: ", timeout)
            if response is None:
                if question.default:
                    return question.default
                return Answer(value=AnswerValue.TIMEOUT)
            if response.strip().lower() in ("y", "yes"):
                return Answer(value=AnswerValue.YES)
            return Answer(value=AnswerValue.NO)

        if question.type == QuestionType.FREEFORM:
            response = await self._read_line("> ", timeout)
            if response is None:
                if question.default:
                    return question.default
                return Answer(value=AnswerValue.TIMEOUT)
            return Answer(value=response.strip(), text=response.strip())

        return Answer(value=AnswerValue.SKIPPED)

    async def inform(self, message: str, stage: str = "") -> None:
        prefix = f"[{stage}] " if stage else ""
        print(f"{prefix}{message}", flush=True)

    async def _read_line(self, prompt: str, timeout: float | None) -> str | None:
        print(prompt, end="", flush=True)
        loop = asyncio.get_event_loop()
        try:
            if timeout is not None:
                line = await asyncio.wait_for(
                    loop.run_in_executor(None, sys.stdin.readline),
                    timeout=timeout,
                )
            else:
                line = await loop.run_in_executor(None, sys.stdin.readline)
            return line.rstrip("\n")
        except asyncio.TimeoutError:
            print()  # newline after timeout
            return None


def _find_option(response: str, options: list[Option]) -> Answer:
    r = response.strip().upper()
    for opt in options:
        if opt.key.upper() == r:
            return Answer(value=opt.key, selected_option=opt)
    # Try label prefix match
    r_lower = response.strip().lower()
    for opt in options:
        if opt.label.lower().startswith(r_lower):
            return Answer(value=opt.key, selected_option=opt)
    # Fallback: first option
    if options:
        return Answer(value=options[0].key, selected_option=options[0])
    return Answer(value=AnswerValue.SKIPPED)
