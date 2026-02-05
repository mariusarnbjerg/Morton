from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone


class Role(str, Enum):
    SYSTEM = "system"
    PATIENT = "patient"
    ASSISTANT = "assistant"


class Mode(str, Enum):
    ANSWER = "answer"   # patient answering standardized question
    CHAT = "chat"       # patient asking free-form question to chatbot


@dataclass
class Message:
    role: Role
    content: str
    ts: int = int(datetime.now(timezone.utc).timestamp())
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Question:
    id: str
    text: str
    type: str = "free_text"        # free_text | yesno | number | choice
    required: bool = True
    help_prompt: Optional[str] = None
    choices: Optional[List[str]] = None
    validation: Optional[Dict[str, Any]] = None


class ConversationState(str, Enum):
    FLOW_ASKING = "flow_asking"
    FLOW_WAITING_ANSWER = "flow_waiting_answer"
    CHAT_MODE = "chat_mode"
    SUMMARIZING = "summarizing"
    DONE = "done"


@dataclass
class Conversation:
    conversation_id: str
    state: ConversationState = ConversationState.FLOW_ASKING
    question_index: int = 0
    active_question_id: Optional[str] = None
    answers: Dict[str, str] = field(default_factory=dict)
