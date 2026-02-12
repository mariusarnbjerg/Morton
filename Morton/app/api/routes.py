"""
API routes for conversation management.
These are the actual endpoints the frontend will call.
"""

from fastapi import APIRouter, HTTPException

from app.api.models import (
    StartConversationRequest,
    StartConversationResponse,
    AnswerQuestionRequest,
    AnswerQuestionResponse,
    ChatRequest,
    ChatResponse,
)
from app.api.dependencies import get_conversation, get_orchestrator, delete_conversation
from app.domain.models import Mode


# Create a router (collection of related endpoints)
router = APIRouter()


# ============================================================================
# Endpoint 1: Start a new conversation
# ============================================================================

@router.post("/conversations/start", response_model=StartConversationResponse)
async def start_conversation(req: StartConversationRequest):
    """
    Start a new conversation (begin questionnaire).

    Example:
        POST /api/v1/conversations/start
        Body: {"conversation_id": "patient-123"}

        Returns: {
            "conversation_id": "patient-123",
            "question": "What is your name?",
            "question_id": "q1",
            "done": false
        }
    """
    # Get or create the conversation
    conv = get_conversation(req.conversation_id)

    # Get the orchestrator
    orch = get_orchestrator()

    # Start the conversation (asks first question)
    res = orch.start(conv)

    # Return the response
    return StartConversationResponse(
        conversation_id=conv.conversation_id,
        question=res.bot_text,
        question_id=conv.active_question_id or "",
        done=res.done
    )


# ============================================================================
# Endpoint 2: Answer a question
# ============================================================================

@router.post("/conversations/{conversation_id}/answer", response_model=AnswerQuestionResponse)
async def answer_question(conversation_id: str, req: AnswerQuestionRequest):
    """
    Answer the current question and get the next one.

    Example:
        POST /api/v1/conversations/patient-123/answer
        Body: {"answer": "John Doe"}

        Returns: {
            "question": "What is your age?",
            "question_id": "q2",
            "done": false
        }
    """
    # Get the conversation (will fail if doesn't exist)
    conv = get_conversation(conversation_id)

    # Get the orchestrator
    orch = get_orchestrator()

    # Process the answer
    res = orch.handle_user_message(conv, req.answer, mode=Mode.ANSWER)

    # Return the next question (or None if done)
    return AnswerQuestionResponse(
        question=res.bot_text if not res.done else None,
        question_id=conv.active_question_id,
        done=res.done
    )


# ============================================================================
# Endpoint 3: Chat with the bot
# ============================================================================

@router.post("/conversations/{conversation_id}/chat", response_model=ChatResponse)
async def chat(conversation_id: str, req: ChatRequest):
    """
    Ask the chatbot a free-form question.

    Example:
        POST /api/v1/conversations/patient-123/chat
        Body: {"message": "Why do you need my age?"}

        Returns: {
            "answer": "We need your age to assess anesthesia risks...",
            "current_question": "What is your age?",
            "current_question_id": "q2"
        }
    """
    # Get the conversation
    conv = get_conversation(conversation_id)

    # Get the orchestrator
    orch = get_orchestrator()

    # Process the chat message
    res = orch.handle_user_message(conv, req.message, mode=Mode.CHAT)

    # Parse the combined response
    # Remember: orchestrator returns "chatbot answer\n\n---\nBack to questionnaire:\nQuestion"
    parts = res.bot_text.split("---\nBack to the questionnaire:\n")

    return ChatResponse(
        answer=parts[0].strip() if len(parts) > 0 else res.bot_text,
        current_question=parts[1].strip() if len(parts) > 1 else "",
        current_question_id=conv.active_question_id or ""
    )


# ============================================================================
# Endpoint 4: Get conversation state (for debugging or UI)
# ============================================================================

@router.get("/conversations/{conversation_id}/state")
async def get_conversation_state(conversation_id: str):
    """
    Get the current state of a conversation.

    Useful for:
    - Showing progress (Question 3 of 10)
    - Debugging
    - Resuming interrupted conversations

    Example:
        GET /api/v1/conversations/patient-123/state

        Returns: {
            "conversation_id": "patient-123",
            "current_question": "What is your age?",
            "question_index": 1,
            "total_questions": 5,
            "answered_count": 1
        }
    """
    conv = get_conversation(conversation_id)
    orch = get_orchestrator()

    # Get current question
    q = orch.question_flow.get_question(conv)

    # Count total questions (a bit hacky, but works)
    total = 0
    while True:
        tmp_conv = type(conv)(conversation_id="tmp")
        tmp_conv.question_index = total
        if not orch.question_flow.get_question(tmp_conv):
            break
        total += 1

    return {
        "conversation_id": conv.conversation_id,
        "current_question": q.text if q else None,
        "current_question_id": q.id if q else None,
        "question_index": conv.question_index,
        "total_questions": total,
        "answered_count": len(conv.answers),
        "done": conv.question_index >= total
    }


# ============================================================================
# Endpoint 5: Get final summary
# ============================================================================

@router.get("/conversations/{conversation_id}/summary")
async def get_summary(conversation_id: str):
    """
    Get the final structured summary (after questionnaire is complete).

    Example:
        GET /api/v1/conversations/patient-123/summary

        Returns: {
            "patient": {...},
            "questionnaire_answers": [...],
            "patient_questions": [...],
            "red_flags": [...],
            "notes": "..."
        }
    """
    conv = get_conversation(conversation_id)
    orch = get_orchestrator()

    try:
        summary = orch.finalize(conv)
        return summary
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate summary: {str(e)}"
        )


# ============================================================================
# Endpoint 6: Delete conversation (cleanup)
# ============================================================================

@router.delete("/conversations/{conversation_id}")
async def delete_conversation_endpoint(conversation_id: str):
    """
    Delete a conversation from memory.

    Example:
        DELETE /api/v1/conversations/patient-123

        Returns: {
            "message": "Conversation deleted",
            "conversation_id": "patient-123"
        }
    """
    deleted = delete_conversation(conversation_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return {
        "message": "Conversation deleted",
        "conversation_id": conversation_id
    }


# ============================================================================
# Debug endpoint: List all conversations
# ============================================================================

@router.get("/conversations")
async def list_conversations():
    """
    List all active conversations (for debugging).

    Example:
        GET /api/v1/conversations

        Returns: {
            "count": 2,
            "conversations": [
                {"conversation_id": "patient-123", "question_index": 2, ...},
                {"conversation_id": "patient-456", "question_index": 0, ...}
            ]
        }
    """
    from app.api.dependencies import active_conversations

    return {
        "count": len(active_conversations),
        "conversations": [
            {
                "conversation_id": conv_id,
                "question_index": conv.question_index,
                "state": conv.state.value,
                "answered_count": len(conv.answers)
            }
            for conv_id, conv in active_conversations.items()
        ]
    }