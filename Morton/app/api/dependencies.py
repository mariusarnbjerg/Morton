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
from app.adapters.ollama_summarizer import OllamaSummarizer
from app.adapters.ollama_llm_client import OllamaLLMClient
from app.adapters.asa_predictor import ASAPredictor

load_dotenv()

OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "qwen3:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
QUESTIONS_PATH  = os.getenv("QUESTIONS_PATH",  "data/questions.json")
SUMMARY_SCHEMA_PATH   = os.getenv("SUMMARY_SCHEMA_PATH",   "data/summary_schema.json")
ASA_MODEL_PATH = os.getenv("ASA_MODEL_PATH", "ml/asa_model.joblib")

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
        llm_client = OllamaLLMClient(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
        asa_predictor = ASAPredictor(ASA_MODEL_PATH)

        _orchestrator_instance = ConversationOrchestrator(
            question_flow=qflow,
            transcript_store=store,
            llm_client=llm_client,
            summarizer=summarizer,
            summary_schema_path=SUMMARY_SCHEMA_PATH,
            asa_predictor=asa_predictor,
        )
        print("✅ Orchestrator ready!")

    return _orchestrator_instance