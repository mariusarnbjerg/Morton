from __future__ import annotations
from typing import Optional, List
from abc import ABC, abstractmethod
from app.domain.models import Conversation, Question, FollowUpQuestion


class IQuestionFlow(ABC):

    @abstractmethod
    def get_question(self, conv: Conversation) -> Optional[Question]:
        """Return the current top-level question based on conversation progress."""
        pass

    @abstractmethod
    def get_all_questions(self) -> List[Question]:
        """Return all questions (used by orchestrator to build the LLM prompt checklist)."""
        pass

    @abstractmethod
    def get_active_follow_up(self, conv: Conversation) -> Optional[FollowUpQuestion]:
        """Return the currently active follow-up question, if any."""
        pass

    @abstractmethod
    def get_unanswered_required_ids(self, conv: Conversation) -> List[str]:
        """Return all required question/follow-up ids not yet in conv.completed_question_ids."""
        pass

    @abstractmethod
    def advance(self, conv: Conversation) -> Conversation:
        """Move to the next top-level question and clear follow-up state."""
        pass