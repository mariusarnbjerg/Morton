"""
Loads questions from questions.json and serves them sequentially,
now supporting follow-up questions, completion_criteria, and confirmation_required.
"""

from __future__ import annotations
import json
from typing import Optional, List
from app.domain.models import Conversation, Question, FollowUpQuestion
from app.interfaces.question_flow import IQuestionFlow


class JsonQuestionFlow(IQuestionFlow):
    def __init__(self, questions_path: str):
        with open(questions_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        q_list = raw["questions"] if isinstance(raw, dict) and "questions" in raw else raw
        self._questions = [self._parse_question(q) for q in q_list]

    def _parse_question(self, data: dict) -> Question:
        """Parse a question dict, including any nested follow_up questions."""
        follow_ups = []
        for fu in data.get("follow_ups", []):
            follow_ups.append(FollowUpQuestion(
                id=fu["id"],
                text=fu["text"],
                trigger=fu["trigger"],
                type=fu.get("type", "free_text"),
                completion_criteria=fu.get("completion_criteria"),
                confirmation_required=fu.get("confirmation_required", False),
            ))

        return Question(
            id=data["id"],
            text=data["text"],
            type=data.get("type", "free_text"),
            required=data.get("required", True),
            help_prompt=data.get("help_prompt"),
            choices=data.get("choices"),
            validation=data.get("validation"),
            completion_criteria=data.get("completion_criteria"),
            confirmation_required=data.get("confirmation_required", False),
            follow_ups=follow_ups,
        )

    def get_all_questions(self) -> List[Question]:
        return list(self._questions)

    def get_question(self, conv: Conversation) -> Optional[Question]:
        if conv.question_index < 0 or conv.question_index >= len(self._questions):
            return None
        return self._questions[conv.question_index]

    def get_active_follow_up(self, conv: Conversation) -> Optional[FollowUpQuestion]:
        """Return the currently active follow-up question, if any."""
        if not conv.active_follow_up_id:
            return None
        q = self.get_question(conv)
        if not q:
            return None
        for fu in q.follow_ups:
            if fu.id == conv.active_follow_up_id:
                return fu
        return None

    def get_unanswered_required_ids(self, conv: Conversation) -> List[str]:
        """
        Return all question/follow-up ids that are required but not yet
        in conv.completed_question_ids. Used by the orchestrator to check
        whether the conversation is finished.
        """
        unanswered = []
        for q in self._questions:
            if not q.required:
                continue
            if q.id not in conv.completed_question_ids:
                unanswered.append(q.id)
            for fu in q.follow_ups:
                # Follow-ups are only "required" if their parent is answered
                # and the trigger was met — the LLM manages this, but we
                # surface the id here so the orchestrator can track it.
                if fu.id in conv.answers and fu.id not in conv.completed_question_ids:
                    unanswered.append(fu.id)
        return unanswered

    def advance(self, conv: Conversation) -> Conversation:
        """Move to the next top-level question and clear follow-up state."""
        conv.question_index += 1
        conv.active_follow_up_id = None
        conv.pending_confirmation_question_id = None
        return conv