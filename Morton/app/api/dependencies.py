"""
Dependency injection for FastAPI.
Creates and manages the orchestrator and active conversations.
"""
import os
from dotenv import load_dotenv

from typing import Dict
from app.domain.models import Conversation
from app.application.orchestrator_with_validation import ConversationOrchestrator
from app.adapters.question_flow_json import JsonQuestionFlow
from app.adapters.store_memory import MemoryTranscriptStore
from app.adapters.ollama_chatbot import OllamaChatbot
from app.adapters.ollama_summarizer import OllamaSummarizer

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# Configuration (you can change these)
# ============================================================================

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
QUESTIONS_PATH = os.getenv("QUESTIONS_PATH", "data/questions.json")
TEMPLATE_PATH = os.getenv("TEMPLATE_PATH", "data/summary_template.json")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")


# ============================================================================
# In-memory conversation storage
# (In production, this would be Redis or a database)
# ============================================================================

# Dictionary to store active conversations
# Key: conversation_id, Value: Conversation object
active_conversations: Dict[str, Conversation] = {}


def get_conversation(conversation_id: str) -> Conversation:
    """
    Get or create a conversation by ID.

    If the conversation doesn't exist, create a new one.
    If it exists, return it.

    Args:
        conversation_id: Unique identifier for the conversation

    Returns:
        Conversation object
    """
    if conversation_id not in active_conversations:
        # Create new conversation
        active_conversations[conversation_id] = Conversation(
            conversation_id=conversation_id
        )

    return active_conversations[conversation_id]


def delete_conversation(conversation_id: str) -> bool:
    """
    Delete a conversation from memory.

    Args:
        conversation_id: ID of conversation to delete

    Returns:
        True if deleted, False if not found
    """
    if conversation_id in active_conversations:
        del active_conversations[conversation_id]
        return True
    return False


# ============================================================================
# Orchestrator singleton
# ============================================================================

# We only need ONE orchestrator for all conversations
# Each conversation keeps its own state in the Conversation object
_orchestrator_instance = None


def get_orchestrator() -> ConversationOrchestrator:
    """
    Get the orchestrator instance (creates it if needed).

    This is a "singleton" - we only create it once and reuse it.

    Why? The orchestrator itself is stateless - all the state
    is in the Conversation object. So we can reuse the same
    orchestrator for all conversations.

    Returns:
        ConversationOrchestrator instance
    """
    global _orchestrator_instance

    if _orchestrator_instance is None:
        print("🔧 Creating orchestrator...")

        # Create all the adapters (same as main.py)
        store = MemoryTranscriptStore()
        qflow = JsonQuestionFlow(QUESTIONS_PATH)
        chatbot = OllamaChatbot(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL
        )
        summarizer = OllamaSummarizer(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL
        )

        # Wire up the orchestrator
        _orchestrator_instance = ConversationOrchestrator(
            question_flow=qflow,
            transcript_store=store,
            chatbot=chatbot,
            summarizer=summarizer,
            template_path=TEMPLATE_PATH
        )

        print("✅ Orchestrator created!")

    return _orchestrator_instance