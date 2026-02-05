from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional
from app.domain.models import Message

class IChatbot(ABC):
    @abstractmethod
    def answer(self, user_text: str, transcript: List[Message], context: Optional[str] = None) -> str:
        """Return assistant answer to user free-form question."""
        pass

