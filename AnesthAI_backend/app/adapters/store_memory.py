'''What: Store conversation turns in memory.
Why: This makes it easy to test now; later we replace it with SQLite/Postgres without touching orchestration logic.'''

from __future__ import annotations
from typing import Dict, List
from app.domain.models import Message
from app.interfaces.transcript_store import ITranscriptStore

class MemoryTranscriptStore(ITranscriptStore):
    def __init__(self):
        self._db: Dict[str, List[Message]] = {}

    def append(self, conversation_id: str, message: Message) -> None:
        self._db.setdefault(conversation_id, []).append(message)

    def get(self, conversation_id: str) -> List[Message]:
        return list(self._db.get(conversation_id, []))
