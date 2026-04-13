"""
Ollama adapter implementing ILLMClient.

Handles all prompt construction, structured output schemas, and HTTP
communication with the Ollama API. The orchestrator delegates all
language tasks here and never touches the LLM directly.
"""

from __future__ import annotations
import json
import requests
from typing import Any, Dict, List, Union

from app.interfaces.llm_client import ILLMClient
from app.domain.models import Message, Role, Question, FollowUpQuestion


# ---------------------------------------------------------------------------
# Structured output schemas
# ---------------------------------------------------------------------------

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "The answer extracted from the patient message. Empty string if not answered."
        },
        "is_complete": {
            "type": "boolean",
            "description": "True if the answer satisfies the completion criteria."
        },
        "needs_followup": {
            "type": "boolean",
            "description": (
                "True if the answer triggers a follow-up. "
                "For example: patient said yes to a yes/no question that requires detail."
            )
        }
    },
    "required": ["answer", "is_complete", "needs_followup"]
}

class OllamaLLMClient(ILLMClient):
    """
    Ollama-backed implementation of ILLMClient.

    All prompt engineering and Ollama-specific logic lives here.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        timeout_s: int = 120,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    # -------------------------------------------------------------------
    # ILLMClient implementation
    # -------------------------------------------------------------------

    def extract(
        self,
        focus: Union[Question, FollowUpQuestion],
        user_text: str,
    ) -> Dict[str, Any]:
        """Ask the LLM: what did the patient answer, and is it complete?"""
        criteria = (
            getattr(focus, "completion_criteria", None)
            or "Patient directly answered the question."
        )

        followup_hint = ""
        if hasattr(focus, "follow_ups") and focus.follow_ups:
            triggers = "\n".join(f"  - {fu.trigger}" for fu in focus.follow_ups)
            followup_hint = (
                f"\nIMPORTANT: Set needs_followup=true if the patient's answer meets ANY of these conditions:\n{triggers}\nNote: if the patient said 'yes' or confirmed the premise, that typically triggers a follow-up."
            )

        system = (
            "You are analyzing a patient's response to a single medical questionnaire question."
            "Your job: extract the answer and judge completeness based strictly on the completion criteria provided. "
            "Do not invent stricter requirements than the criteria states. "
            "For questions that ask whether something exists or has happened, "
            "a clear 'yes' or 'no' is a complete answer — even without details. "
            "Only mark 'yes' as incomplete when the completion criteria explicitly "
            "requires specific details (like names, dates, or descriptions). "
            "If the completion criteria says approximate answers are acceptable, accept them. "
            "If the patient says they don't know, and the criteria allows for that, mark as complete. "
            "IMPORTANT: A patient may both answer AND ask a question in the same message "
            "(e.g. 'I'm 25, why do you ask?', 'My name is Marius, why is this relevant?'). "
            "In this case, extract the answer normally and judge completeness as usual — "
            "do not discard the answer just because a question was also asked. "
            "Only set answer to empty string and is_complete to false if the patient "
            "provided NO answer at all — only a question or completely off-topic message."
        )

        user = (
            f'Question ({focus.id}): "{focus.text}"\n'
            f"Completion criteria: {criteria}"
            f"{followup_hint}\n\n"
            f'Patient said: "{user_text}"'
        )

        return self._ollama_structured(system, user, EXTRACT_SCHEMA)

    def generate_reply(
        self,
        transcript: List[Message],
        instruction: Dict[str, Any],
        patient_name: str = "",
    ) -> str:
        """Generate a natural reply given a precise instruction about what to do next."""
        action = instruction["action"]
        question_text = instruction.get("question_text")
        acknowledged = instruction.get("acknowledged_answer", "")
        current_question_text = instruction.get("current_question_text", "")

        # ------------------------------------------------------------------
        # Deterministic actions — no LLM call.
        # ------------------------------------------------------------------
        if action == "reentry":
            return ""

        if action in ("ask_next", "ask_followup"):
            full_message = instruction.get("full_message", "").strip()
            acknowledged = instruction.get("acknowledged_answer", "").strip()
            has_extra = full_message and full_message.lower() != acknowledged.lower()

            # Clean answer with nothing extra — skip the LLM entirely.
            if not has_extra:
                return ""

            current_question_text = instruction.get("current_question_text", "").strip()
            task = (
                f'You just asked the patient: "{current_question_text}". '
                f'They replied: "{full_message}". '
                f"Reply in one warm, natural sentence. "
                f"If they asked something back, answer it briefly and factually."
            )

        # ------------------------------------------------------------------
        # Reask — patient didn't answer, but is still engaged with the
        # current question. Respond to what they said; orchestrator will
        # re-append the current question.
        # ------------------------------------------------------------------
        elif action == "reask":
            full_message = instruction.get("full_message", "").strip()
            current_question_text = instruction.get("current_question_text", "").strip()
            task = (
                f'You just asked the patient: "{current_question_text}". '
                f'They replied: "{full_message}". '
                f"If their reply contains a question, answer that question briefly and "
                f"factually in one sentence — this is your top priority. "
                f"Otherwise, reply with one warm, natural sentence."
            )

        # ------------------------------------------------------------------
        # Free chat — patient's reply did not satisfy the current question.
        # ------------------------------------------------------------------
        elif action == "free_chat":
            full_message = (
                    instruction.get("full_message", "").strip()
                    or instruction.get("acknowledged_answer", "").strip()
            )
            pending_question = instruction.get("question_text", "").strip()
            task = (
                f'You just asked the patient: "{pending_question}". '
                f'They replied: "{full_message}". '
                f"Reply in one warm, natural sentence."
            )

        # ------------------------------------------------------------------
        # Done.
        # ------------------------------------------------------------------
        elif action == "done":
            task = (
                "All questions are complete. Thank the patient warmly in one or two "
                "sentences and let them know the consultation is finished. Do not "
                "make any assumptions about their procedure."
            )

        else:
            return ""

        # ------------------------------------------------------------------
        # Stable system prompt — sets persona, no decision logic.
        # ------------------------------------------------------------------
        system = (
                "You are a warm, professional clinical assistant conducting a "
                "pre-anesthesia consultation. Speak naturally, like a person, not a form. "
                "Reply in ONE brief sentence unless told otherwise. "
                "Never ask any questions — questions are appended separately. "
                "Never repeat the patient's words back to them. "
                "Never invent medical facts, personal details about yourself, or "
                "anything the patient did not explicitly say. "
                "You are NOT a medical expert. If the patient asks any medical, clinical, "
                "or pharmaceutical question — including about drugs, procedures, conditions, "
                "symptoms, or treatments — do not answer it. Say only that their medical "
                "team or pharmacist can answer that for them. "
                "If the patient asks to discuss something further with the medical team, "
                "or wants to flag a concern, reassure them that everything they share here "
                "is passed on to the team afterward — they don't need to repeat themselves. "
                "If the patient expresses fear or anxiety, acknowledge it with empathy."
                + (f" The patient's name is {patient_name}." if patient_name else "")
        )

        user = (
            f"{task}\n\n"
            f"Write only your reply. No labels, no JSON, no explanation, no questions."
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "think": True,
            "options": {
                "repeat_penalty": 1.2,
            },
        }
        r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()

    # -------------------------------------------------------------------
    # Ollama HTTP helper
    # -------------------------------------------------------------------

    def _ollama_structured(self, system: str, user: str, schema: Dict) -> Dict[str, Any]:
        """Call Ollama with structured output enforcement via the format parameter."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "stream": False,
            "format": schema,
            "think": False
        }
        r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        raw = r.json()["message"]["content"]
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Structured LLM call returned invalid JSON: {e}\nOutput: {raw}")