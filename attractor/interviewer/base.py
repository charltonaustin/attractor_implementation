"""Interviewer ABC and core types."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QuestionType(str, Enum):
    YES_NO = "yes_no"
    MULTIPLE_CHOICE = "multiple_choice"
    FREEFORM = "freeform"
    CONFIRMATION = "confirmation"


class AnswerValue(str, Enum):
    YES = "yes"
    NO = "no"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


@dataclass
class Option:
    key: str
    label: str


@dataclass
class Question:
    text: str
    type: QuestionType = QuestionType.MULTIPLE_CHOICE
    options: list[Option] = field(default_factory=list)
    default: "Answer | None" = None
    timeout_seconds: float | None = None
    stage: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Answer:
    value: str | AnswerValue = ""
    selected_option: Option | None = None
    text: str = ""

    def is_timeout(self) -> bool:
        return self.value == AnswerValue.TIMEOUT

    def is_skipped(self) -> bool:
        return self.value == AnswerValue.SKIPPED


class Interviewer(ABC):
    @abstractmethod
    async def ask(self, question: Question) -> Answer: ...

    async def ask_multiple(self, questions: list[Question]) -> list[Answer]:
        return [await self.ask(q) for q in questions]

    async def inform(self, message: str, stage: str = "") -> None:
        pass
