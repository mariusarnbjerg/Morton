"""
Pydantic models for API request/response validation.
"""
from pydantic import BaseModel, Field
from typing import Optional


# ============================================================================
# REQUEST models
# ============================================================================

class StartConversationRequest(BaseModel):
    conversation_id: str = Field(..., description="Unique ID for this conversation")


class MessageRequest(BaseModel):
    """Single request model for all patient input — answers and questions alike."""
    message: str = Field(..., description="Patient's message (answer or question)")


# ============================================================================
# RESPONSE models
# ============================================================================

class StartConversationResponse(BaseModel):
    conversation_id: str
    bot_text: str
    done: bool = False


class MessageResponse(BaseModel):
    bot_text: str
    done: bool = False
    current_question_id: Optional[str] = None
    acknowledged: bool = False


class ConversationStateResponse(BaseModel):
    conversation_id: str
    state: str
    answered_count: int
    done: bool
    current_question_id: Optional[str] = None