"""
API routes for conversation management.
"""
from fastapi import APIRouter, HTTPException

from app.api.models import (
    StartConversationRequest,
    StartConversationResponse,
    MessageRequest,
    MessageResponse,
    ConversationStateResponse,
)
from app.api.dependencies import get_conversation, get_orchestrator, delete_conversation

router = APIRouter()


# ============================================================================
# Start a new conversation
# ============================================================================

@router.post("/conversations/start", response_model=StartConversationResponse)
async def start_conversation(req: StartConversationRequest):
    """
    Start a new conversation. Returns the opening message.
    """
    conv = get_conversation(req.conversation_id)
    orch = get_orchestrator()
    res  = orch.start(conv)

    return StartConversationResponse(
        conversation_id=conv.conversation_id,
        bot_text=res.bot_text,
        done=res.done,
    )


# ============================================================================
# Send a message (answer OR question — orchestrator handles both)
# ============================================================================

@router.post("/conversations/{conversation_id}/message", response_model=MessageResponse)
async def send_message(conversation_id: str, req: MessageRequest):
    """
    Send any patient message. The orchestrator detects whether it's an answer
    or a question and responds accordingly.
    """
    conv = get_conversation(conversation_id)
    orch = get_orchestrator()
    res  = orch.handle_user_message(conv, req.message)

    return MessageResponse(
        bot_text=res.bot_text,
        done=res.done,
    )


# ============================================================================
# Get conversation state
# ============================================================================

@router.get("/conversations/{conversation_id}/state", response_model=ConversationStateResponse)
async def get_conversation_state(conversation_id: str):
    """
    Get the current state of a conversation (for progress display).
    """
    conv = get_conversation(conversation_id)

    return ConversationStateResponse(
        conversation_id=conv.conversation_id,
        state=conv.state.value,
        answered_count=len(conv.answers),
        done=conv.state.value == "done",
    )


# ============================================================================
# Get final summary (after questionnaire complete)
# ============================================================================

@router.get("/conversations/{conversation_id}/summary")
async def get_summary(conversation_id: str):
    """
    Generate and return the structured clinical summary.
    Only valid once done=true.
    """
    conv = get_conversation(conversation_id)
    orch = get_orchestrator()

    try:
        return orch.finalize(conv)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {e}")


# ============================================================================
# Delete conversation
# ============================================================================

@router.delete("/conversations/{conversation_id}")
async def delete_conversation_endpoint(conversation_id: str):
    if not delete_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"message": "Conversation deleted", "conversation_id": conversation_id}


# ============================================================================
# Debug: list all conversations
# ============================================================================

@router.get("/conversations")
async def list_conversations():
    from app.api.dependencies import active_conversations
    return {
        "count": len(active_conversations),
        "conversations": [
            {
                "conversation_id": cid,
                "state": conv.state.value,
                "answered_count": len(conv.answers),
            }
            for cid, conv in active_conversations.items()
        ],
    }