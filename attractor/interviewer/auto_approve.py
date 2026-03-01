"""AutoApproveInterviewer: always approves, selects first option."""
from __future__ import annotations

from attractor.interviewer.base import Answer, AnswerValue, Interviewer, Question, QuestionType


class AutoApproveInterviewer(Interviewer):
    async def ask(self, question: Question) -> Answer:
        if question.type in (QuestionType.YES_NO, QuestionType.CONFIRMATION):
            return Answer(value=AnswerValue.YES)
        if question.type == QuestionType.MULTIPLE_CHOICE and question.options:
            opt = question.options[0]
            return Answer(value=opt.key, selected_option=opt)
        return Answer(value="auto-approved", text="auto-approved")
