"""
Pydantic models for API request/response validation.
These define the "shape" of data going in and out of the API.
"""

from pydantic import BaseModel, Field
from typing import Optional

# ============================================================================
# REQUEST Models (what the frontend sends to us)
# ============================================================================

class StartConversationRequest(BaseModel):
    """Request to start a new conversation"""
    conversation_id: str = Field(..., description="Unique ID for this conversation")


class AnswerQuestionRequest(BaseModel):
    """Request to answer the current question"""
    answer: str = Field(..., description="Patient's answer")


class ChatRequest(BaseModel):
    """Request to ask the chatbot a question"""
    message: str = Field(..., description="Patient's free-form question")


# ============================================================================
# RESPONSE Models (what we send back to the frontend)
# ============================================================================

class StartConversationResponse(BaseModel):
    """Response when starting a conversation"""
    conversation_id: str
    question: str          # The first question
    question_id: str
    done: bool = False     # Is questionnaire complete?


class AnswerQuestionResponse(BaseModel):
    """Response after answering a question"""
    question: Optional[str] = None  # Next question (None if done)
    question_id: Optional[str] = None
    done: bool = False


class ChatResponse(BaseModel):
    """Response from the chatbot"""
    answer: str                    # Chatbot's answer
    current_question: str          # Which questionnaire question we're still on
    current_question_id: str