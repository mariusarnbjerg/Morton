from __future__ import annotations
import json
import requests
from typing import Any, Dict, List

from app.interfaces.summarizer import ISummarizer
from app.domain.models import Message, Role

class OllamaSummarizer(ISummarizer):
    """
    Simplified summarizer using Ollama's native structured output support.

    Ollama's 'format' parameter constrains the model to output valid JSON
    matching the schema, eliminating most retry logic.
    """

    def __init__(
            self,
            model: str = "llama3.1",
            base_url: str = "http://localhost:11434",
            timeout_s: int = 120,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def _ollama_chat(self, messages: List[Dict[str, str]], schema: Dict[str, Any]) -> str:
        """
        Call Ollama with structured output enforcement.

        The 'format' parameter tells Ollama to constrain generation to match the schema.
        """

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": schema  # Native structured output support
        }

        r = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout_s
        )
        r.raise_for_status()
        return r.json()["message"]["content"]

    def summarize(self, transcript: List[Message], schema: Dict[str, Any]) -> Dict[str, Any]:

        # Convert transcript to compact text
        def fmt(m: Message) -> str:
            return f"{m.role.value.upper()}: {m.content}"

        transcript_text = "\n".join(fmt(m) for m in transcript)

        # System prompt - simpler now since Ollama enforces structure
        system_prompt = (
            "You are a clinical summarization assistant. "
            "Extract and summarize the relevant medical information from the transcript. "
            "The output will automatically be formatted as valid JSON matching the required schema."
        )

        user_prompt = (
            "Summarize the following medical questionnaire transcript.\n\n"
            "Transcript:\n"
            f"{transcript_text}\n"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Single call - no retry loop needed!
        raw = self._ollama_chat(messages, schema)

        # Ollama guarantees valid JSON matching the schema
        try:
            data = json.loads(raw)
            return data
        except json.JSONDecodeError as e:
            # This should rarely happen with native structured outputs
            # but keep as safeguard
            raise ValueError(
                f"Ollama returned invalid JSON despite schema constraint. "
                f"Error: {e}\nOutput: {raw}"
            )


# COMPARISON: Old approach with manual retries
"""
OLD VERSION (manual retry logic):
- System prompt explicitly describes schema
- Extract JSON from markdown fences
- Validate against schema
- If invalid, ask model to repair
- Retry up to 3 times
- ~80 lines of code

NEW VERSION (Ollama native):
- Pass schema directly to Ollama via 'format' param
- Ollama constrains generation to valid JSON
- Single call, no retries needed
- ~40 lines of code
- Higher success rate

TRADE-OFF:
- Ollama-specific (less portable)
- But much simpler and more reliable
"""