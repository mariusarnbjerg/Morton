'''What: Create an adapter that implements IChatbot and calls Ollama.
Why: Keeps your chatbot independent; later you can swap to RAG or another model without touching the orchestrator.'''

from __future__ import annotations
from typing import List, Optional
import requests

from app.interfaces.chatbot import IChatbot
from app.domain.models import Message, Role


class OllamaChatbot(IChatbot):
    """
    Minimal Ollama chat adapter using HTTP.
    Assumes Ollama is running locally: http://localhost:11434
    """
    def __init__(
        self,
        model: str = "llama3.1",
        base_url: str = "http://localhost:11434",
        system_prompt: Optional[str] = None,
        timeout_s: int = 60,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.system_prompt = system_prompt or (
            "You are a helpful clinical assistant answering a patient's free-form questions during a pre-anesthesia questionnaire. "
            "Important rules:\n"
            "1) Only answer the patient's current free-form question.\n"
            "2) Do NOT repeat, re-ask, or answer the standardized questionnaire questions unless the patient explicitly asks about them.\n"
            "3) Keep answers short, clear, and non-alarming.\n"
            "4) If asked for personalized medical advice or urgent symptoms, advise contacting the clinic.\n"
        )
        self.timeout_s = timeout_s

    def answer(self, user_text: str, transcript: List[Message], context: Optional[str] = None) -> str:
        chat_context = []

        for m in transcript[-30:]:
            mode = (m.meta or {}).get("mode")
            if m.role == Role.PATIENT and mode == "chat":
                if m.content.strip() == user_text.strip():
                    continue
                chat_context.append(("user", m.content))
            elif m.role == Role.ASSISTANT and mode == "chat":
                chat_context.append(("assistant", m.content))

        # Build messages
        messages = [{"role": "system", "content": self.system_prompt}]

        # Add questionnaire context (NOT dialogue)
        if context:
            messages.append({"role": "system", "content": f"Questionnaire context: {context}"})

        for role, content in chat_context[-10:]:
            messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": user_text})

        payload = {"model": self.model, "messages": messages, "stream": False}
        r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()["message"]["content"]

    #  --------------------OLD VERSION OF answer function -------------------------------
    # def answer(self, user_text: str, transcript: List[Message]) -> str:
    #     # Convert transcript (optional) into Ollama messages
    #     messages = [{"role": "system", "content": self.system_prompt}]
    #
    #     # Keep a short context window: last ~10 turns is usually enough for MVP
    #     for m in transcript[-10:]:
    #         role = "user" if m.role == Role.PATIENT else "assistant" if m.role == Role.ASSISTANT else "system"
    #         messages.append({"role": role, "content": m.content})
    #
    #     messages.append({"role": "user", "content": user_text})
    #
    #     payload = {"model": self.model, "messages": messages, "stream": False}
    #
    #     r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout_s)
    #     r.raise_for_status()
    #     data = r.json()
    #
    #     # Ollama returns: {"message":{"role":"assistant","content":"..."}, ...}
    #     return data["message"]["content"]
