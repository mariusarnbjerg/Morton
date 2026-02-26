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
    confirmation_required: bool = False


@dataclass
class Question:
    """
    A standardized question in the questionnaire flow.

    Layers of completion enforcement:
      - Layer A: Global prompt-level rules (always active, no config needed)
      - Layer B: completion_criteria — explicit per-question definition of "answered"
      - Layer C: confirmation_required — LLM generates a closing confirmation before marking done
    """
    id: str
    text: str
    type: str = "free_text"                 # free_text | yesno | number | choice
    required: bool = True
    help_prompt: Optional[str] = None
    choices: Optional[List[str]] = None
    validation: Optional[Dict[str, Any]] = None
    completion_criteria: Optional[str] = None       # Layer B
    confirmation_required: bool = False             # Layer C
    follow_ups: List[FollowUpQuestion] = field(default_factory=list)


class ConversationState(str, Enum):
    """
    Simplified state — the LLM now drives conversation naturally.
    No more FLOW_WAITING_ANSWER or CHAT_MODE routing.
    """
    IN_PROGRESS = "in_progress"
    AWAITING_CONFIRMATION = "awaiting_confirmation"  # Layer C: waiting for patient to confirm summary
    DONE = "done"


@dataclass
class Conversation:
    conversation_id: str
    state: ConversationState = ConversationState.IN_PROGRESS

    # Top-level question progress
    question_index: int = 0

    # Tracks the active follow-up question id, if any (None = on parent question)
    active_follow_up_id: Optional[str] = None

    # Tracks which question is currently awaiting Layer C confirmation
    pending_confirmation_question_id: Optional[str] = None

    # All collected answers keyed by question id (includes follow-up ids)
    answers: Dict[str, str] = field(default_factory=dict)

    # Tracks which question ids have been fully completed (including follow-ups + confirmation)
    completed_question_ids: List[str] = field(default_factory=list)