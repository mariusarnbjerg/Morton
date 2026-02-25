"""
Streaming-capable Ollama chatbot adapter.

Supports both blocking (answer()) and streaming (answer_stream()) modes.
"""

from __future__ import annotations
from typing import List, Optional, Iterator
import requests

from app.interfaces.chatbot import IChatbot
from app.domain.models import Message, Role


class OllamaChatbotStreaming(IChatbot):
    """
    Ollama chat adapter with streaming support.

    Two modes:
    1. answer() - Blocking, returns complete response (backward compatible)
    2. answer_stream() - Yields tokens as they arrive
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

    def _build_messages(
            self,
            user_text: str,
            transcript: List[Message],
            context: Optional[str] = None
    ) -> List[dict]:
        """Build messages array for Ollama (shared by both modes)"""
        chat_context = []

        # Extract only CHAT mode messages
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

        # Add recent chat history
        for role, content in chat_context[-10:]:
            messages.append({"role": role, "content": content})

        # Add current user message
        messages.append({"role": "user", "content": user_text})

        return messages

    def answer(self, user_text: str, transcript: List[Message], context: Optional[str] = None) -> str:
        """
        Blocking mode - returns complete response.
        (Backward compatible with existing code)
        """
        messages = self._build_messages(user_text, transcript, context)

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False  # No streaming
        }

        r = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout_s
        )
        r.raise_for_status()
        return r.json()["message"]["content"]

    def answer_stream(
            self,
            user_text: str,
            transcript: List[Message],
            context: Optional[str] = None
    ) -> Iterator[str]:
        """
        Streaming mode - yields tokens as they arrive.

        Usage:
            for token in chatbot.answer_stream("Why do I need anesthesia?", transcript):
                print(token, end="", flush=True)

        Yields:
            str: Individual tokens/words from the LLM
        """
        messages = self._build_messages(user_text, transcript, context)

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True  # Enable streaming!
        }

        # Stream request
        with requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout_s,
                stream=True  # Key: stream=True on requests
        ) as r:
            r.raise_for_status()

            # Ollama returns newline-delimited JSON
            for line in r.iter_lines():
                if line:
                    import json
                    chunk = json.loads(line)

                    # Extract token from response
                    if "message" in chunk:
                        content = chunk["message"].get("content", "")
                        if content:
                            yield content

                    # Check if done
                    if chunk.get("done", False):
                        break


# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    # Test streaming
    chatbot = OllamaChatbotStreaming()

    print("Testing streaming mode:")
    print("USER: Why do I need to fast before surgery?")
    print("BOT: ", end="", flush=True)

    for token in chatbot.answer_stream(
            "Why do I need to fast before surgery?",
            transcript=[],
            context="Current question: Have you had anesthesia before?"
    ):
        print(token, end="", flush=True)

    print("\n")