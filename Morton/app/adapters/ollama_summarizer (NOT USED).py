from __future__ import annotations
import json
import requests
from typing import Any, Dict, List, Optional

from jsonschema import validate, ValidationError

from app.interfaces.summarizer import ISummarizer
from app.domain.models import Message, Role
from app.adapters.template_schema import template_to_json_schema


def _extract_json(text: str) -> str:
    """
    Best-effort JSON extraction:
    - If model returns extra text, try to grab the first {...} block.
    """
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end+1]
    return text


class OllamaSummarizer(ISummarizer):
    def __init__(
        self,
        model: str = "llama3.1",
        base_url: str = "http://localhost:11434",
        timeout_s: int = 120,
        max_retries: int = 2,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    def _ollama_chat(self, messages: List[Dict[str, str]]) -> str:
        payload = {"model": self.model, "messages": messages, "stream": False}
        r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()["message"]["content"]

    def summarize(self, transcript: List[Message], template: Dict[str, Any]) -> Dict[str, Any]:
        schema = template_to_json_schema(template)

        # Convert transcript to compact text (you can refine this later)
        def fmt(m: Message) -> str:
            return f"{m.role.value.upper()}: {m.content}"

        transcript_text = "\n".join(fmt(m) for m in transcript)

        system_prompt = (
            "You are a clinical summarization assistant. "
            "Return ONLY valid JSON. No markdown, no commentary.\n"
            "The JSON MUST match the provided JSON Schema exactly:\n"
            f"{json.dumps(schema)}"
        )

        user_prompt = (
            "Summarize the following transcript into the JSON structure required by the schema.\n\n"
            "Transcript:\n"
            f"{transcript_text}\n"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        last_error: Optional[str] = None

        for attempt in range(self.max_retries + 1):
            raw = self._ollama_chat(messages)
            candidate = _extract_json(raw)

            try:
                data = json.loads(candidate)
                validate(instance=data, schema=schema)
                return data
            except (json.JSONDecodeError, ValidationError) as e:
                last_error = str(e)

                # Ask the model to repair its JSON according to schema
                repair_prompt = (
                    "Your previous output was invalid. Fix it.\n"
                    "Rules:\n"
                    "- Output ONLY JSON\n"
                    "- Must match schema exactly (no extra keys)\n"
                    "- Ensure valid JSON\n\n"
                    f"Validation/parse error:\n{last_error}\n\n"
                    f"Invalid output:\n{raw}\n"
                )
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": repair_prompt},
                ]

        raise ValueError(f"Could not produce valid summary JSON after retries. Last error: {last_error}")
