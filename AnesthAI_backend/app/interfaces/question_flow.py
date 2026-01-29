from __future__ import annotations
from typing import Optional
from abc import ABC, abstractmethod
from app.domain.models import Conversation, Question

class IQuestionFlow(ABC):
    @abstractmethod
    def get_question(self, conv: Conversation) -> Optional[Question]:
        """Return current question based on conversation progress."""
        pass

    @abstractmethod
    def advance_with_answer(self, conv: Conversation, answer_text: str) -> Conversation:
        """Update conversation progress after receiving an answer."""
        pass
