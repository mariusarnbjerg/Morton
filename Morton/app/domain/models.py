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
    """
    Kept for transcript metadata only — no longer drives orchestration routing.
    The LLM handles conversation and follow-up questions in a single call.
    """
    ANSWER = "answer"
    CHAT = "chat"


@dataclass
class Message:
    role: Role
    content: str
    ts: int = field(default_factory=lambda: int(datetime.now(timezone.utc).timestamp()))
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FollowUpQuestion:
    """
    A conditional follow-up question attached to a parent Question.

    The 'trigger' field is a natural language condition evaluated by the LLM
    against the patient's actual answer. If the condition is met, this follow-up
    is expected before the parent question is considered complete.

    Example trigger: "patient confirms taking prescription medications but has not yet listed them"
    """
    id: str
    text: str
    trigger: str                            # Natural language condition for LLM to evaluate
    type: str = "free_text"
    completion_criteria: Optional[str] = None


@dataclass
class Question:
    """
    A standardized question in the questionnaire flow.

    Layers of completion enforcement:
      - Layer A: Global prompt-level rules (always active, no config needed)
      - Layer B: completion_criteria — explicit per-question definition of "answered"
    """
    id: str
    text: str
    type: str = "free_text"                 # free_text | yesno | number | choice
    required: bool = True
    help_prompt: Optional[str] = None
    choices: Optional[List[str]] = None
    validation: Optional[Dict[str, Any]] = None
    completion_criteria: Optional[str] = None
    follow_ups: List[FollowUpQuestion] = field(default_factory=list)


class ConversationState(str, Enum):
    """
    IN_PROGRESS:           Normal flow — extracting answers, advancing questions.
    FREE_CHAT:             Patient didn't answer the current question — chatting freely
                           until they signal they're ready to continue.
    DONE:                  All required questions answered.
    """
    IN_PROGRESS = "in_progress"
    FREE_CHAT = "free_chat"
    DONE = "done"


FREE_CHAT_QUESTION = FollowUpQuestion(
    id="free_chat",
    text="Is there anything else on your mind, or are you ready to continue?",
    trigger="",
    completion_criteria=(
        "Patient has indicated they are ready to continue or have nothing more to add. "
        "This includes both affirmative signals ('yes', 'sure', 'let's go', 'ready') "
        "AND negative signals ('no', 'nothing else', 'nope', 'nothing more', 'I'm good'). "
        "Return is_complete=false only if the patient is clearly still engaging — "
        "asking a new question, sharing new information, or expressing a concern."
    ),
)

@dataclass
class Conversation:
    conversation_id: str
    state: ConversationState = ConversationState.IN_PROGRESS

    # Top-level question progress
    question_index: int = 0

    # Tracks the active follow-up question id, if any (None = on parent question)
    active_follow_up_id: Optional[str] = None

    # All collected answers keyed by question id (includes follow-up ids)
    answers: Dict[str, str] = field(default_factory=dict)

    # Tracks which question ids have been fully completed (including follow-ups)
    completed_question_ids: List[str] = field(default_factory=list)