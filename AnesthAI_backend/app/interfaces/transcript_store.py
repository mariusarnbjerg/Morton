from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
from app.domain.models import Message

class ITranscriptStore(ABC):
    @abstractmethod
    def append(self, conversation_id: str, message: Message) -> None:
        pass

    @abstractmethod
    def get(self, conversation_id: str) -> List[Message]:
        pass