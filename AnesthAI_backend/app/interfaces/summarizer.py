from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from app.domain.models import Message

class ISummarizer(ABC):
    @abstractmethod
    def summarize(self, transcript: List[Message], template: Dict[str, Any]) -> Dict[str, Any]:
        """Return structured JSON matching template."""
        pass

