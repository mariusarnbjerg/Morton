"""
Dependency injection for FastAPI.
Creates and manages the orchestrator and active conversations.
"""
import os
from dotenv import load_dotenv
from typing import Dict

from app.domain.models import Conversation
from app.application.orchestrator import ConversationOrchestrator
from app.adapters.question_flow_json import JsonQuestionFlow
from app.adapters.store_memory import MemoryTranscriptStore
from app.adapters.improved_ollama_summarizer import OllamaSummarizer

load_dotenv()

OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "llama3.1")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
QUESTIONS_PATH  = os.getenv("QUESTIONS_PATH",  "data/questions.json")
TEMPLATE_PATH   = os.getenv("TEMPLATE_PATH",   "data/summary_schema.json")

active_conversations: Dict[str, Conversation] = {}


def get_conversation(conversation_id: str) -> Conversation:
    if conversation_id not in active_conversations:
        active_conversations[conversation_id] = Conversation(
            conversation_id=conversation_id
        )
    return active_conversations[conversation_id]


def delete_conversation(conversation_id: str) -> bool:
    if conversation_id in active_conversations:
        del active_conversations[conversation_id]
        return True
    return False


_orchestrator_instance = None


def get_orchestrator() -> ConversationOrchestrator:
    global _orchestrator_instance

    if _orchestrator_instance is None:
        print("🔧 Creating orchestrator...")
        store      = MemoryTranscriptStore()
        qflow      = JsonQuestionFlow(QUESTIONS_PATH)
        summarizer = OllamaSummarizer(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)

        _orchestrator_instance = ConversationOrchestrator(
            question_flow=qflow,
            transcript_store=store,
            summarizer=summarizer,
            template_path=TEMPLATE_PATH,
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
        )
        print("✅ Orchestrator ready!")

    return _orchestrator_instance