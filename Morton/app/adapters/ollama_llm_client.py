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

CONFIRM_SCHEMA = {
    "type": "object",
    "properties": {
        "confirmed": {
            "type": "boolean",
            "description": "True if the patient is agreeing with or confirming the summary (even if they also add more). False if they are disagreeing, correcting, or unclear."
        },
        "additional_info": {
            "type": "string",
            "description": "Any NEW information the patient added beyond just confirming. For example if asked to confirm gluten allergy and they say 'yes and also penicillin', this field should contain 'penicillin'. Empty string if nothing new was added."
        }
    },
    "required": ["confirmed", "additional_info"]
}

IS_QUESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "is_question": {
            "type": "boolean",
            "description": (
                "True if the patient is asking a question — either instead of answering, or in addition to answering. "
                "Examples that ARE questions: 'Why do you ask?', 'What does that mean?', 'Will I be awake?', 'Should I be worried?', 'I don't understand'. "
                "Examples that are NOT questions: 'Yes', 'No', 'My name is Marius', 'I'm 25', 'I had a heart attack'."
            )
        }
    },
    "required": ["is_question"]
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
        timeout_s: int = 60,
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
            "You are analyzing a patient's response to a single medical questionnaire question. "
            "Your job: extract the answer and judge completeness based strictly on the completion criteria provided. "
            "Do not invent stricter requirements than the criteria states. "
            "'yes' alone is incomplete for questions needing specific details, "
            "but 'no' or 'none' is complete for questions asking whether something exists. "
            "If the completion criteria says approximate answers are acceptable, accept them. "
            "If the patient says they don't know, and the criteria allows for that, mark as complete. "
            "IMPORTANT: If the patient is asking a question instead of answering "
            "(e.g. 'Why do you ask?', 'Why is that relevant?', 'What does that mean?'), "
            "set answer to empty string and is_complete to false. "
            "Do NOT infer or fabricate an answer from context when the patient has not provided one."
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
        patient_question = instruction.get("patient_question", "")

        if action == "ask_next":
            if acknowledged:
                task = (
                    f"The patient just provided information. React to it naturally in one brief warm sentence "
                    f"(do NOT quote or repeat the raw extracted value verbatim — rephrase naturally based on "
                    f"the conversation above), then ask this next question: \"{question_text}\""
                )
            else:
                task = f"Ask this question naturally: \"{question_text}\""

        elif action == "ask_followup":
            task = (
                f"The patient just answered. Acknowledge what they said naturally in one brief sentence "
                f"(based on the conversation above, not a raw value), "
                f"then ask this follow-up question: \"{question_text}\""
            )

        elif action == "confirm":
            answer = instruction.get("confirm_answer", acknowledged)
            task = (
                f"Briefly summarize the patient's answer (\"{answer}\") back to them "
                f"and ask them to confirm it is correct. Keep it short and natural."
            )

        elif action == "reask":
            if acknowledged:
                task = (
                    f"The patient said \"{acknowledged}\" but this didn't fully answer the question. "
                    f"Gently and warmly ask again: \"{question_text}\""
                )
            else:
                task = f"The patient didn't answer the question. Politely ask again: \"{question_text}\""

        elif action == "done":
            task = (
                "All questions are complete. Thank the patient warmly, "
                "let them know the consultation is finished, and wish them well."
            )
        else:
            task = f"Ask: \"{question_text}\""

        # If the patient asked a question, answer it first before the main task
        if patient_question:
            task = (
                f"The patient asked: \"{patient_question}\". "
                f"Answer it briefly and reassuringly in one sentence. Then: {task}"
            )

        # Last few turns for conversational context
        recent = [m for m in transcript if m.role in (Role.PATIENT, Role.ASSISTANT)][-6:]
        history = "\n".join(
            f"{'Patient' if m.role == Role.PATIENT else 'Assistant'}: {m.content}"
            for m in recent
        )

        system = (
            "You are a warm, professional clinical assistant conducting a pre-anesthesia consultation. "
            "Speak naturally and conversationally — not like a form or a checklist. "
            "Keep replies concise: one brief acknowledgment (if needed) and one question. "
            "Do not add unsolicited medical information or explanations. "
            "Do not invent personal details about yourself such as your name or age. "
            "If the patient expresses fear or anxiety, acknowledge it with empathy before moving on. "
            + (f"The patient's name is {patient_name}. " if patient_name else "")
        )

        user = (
            f"Recent conversation:\n{history}\n\n"
            f"Your task: {task}\n\n"
            f"Write only your reply to the patient. No labels, no JSON, no explanation."
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "stream": False
        }
        r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()

    def check_confirmation(
        self,
        user_text: str,
        last_bot_message: str,
        current_answer: str,
    ) -> Dict[str, Any]:
        """Determine whether the patient is confirming a summary."""
        system = (
            "You are determining whether a patient is confirming a summary read back to them, "
            "and whether they added any new information. "
            "Consider context carefully — 'no, I haven't' in response to 'You haven't been told X, right?' is a confirmation. "
            "If the patient confirms AND adds new info (e.g. 'yes, and also penicillin'), "
            "set confirmed=true and put the new info in additional_info."
        )
        user = (
            f'The assistant just said: "{last_bot_message}"\n'
            f'Current recorded answer: "{current_answer}"\n'
            f'The patient responded: "{user_text}"\n\n'
            f"Is the patient confirming? Did they add any new information?"
        )
        return self._ollama_structured(system, user, CONFIRM_SCHEMA)

    def detect_question(self, user_text: str) -> bool:
        """Classify whether the patient is asking a question."""
        system = (
            "You are classifying a single patient message. "
            "Return is_question=true if the patient is asking something — "
            "even if they also provided an answer. "
            "Return is_question=false if they are only answering (e.g. 'Yes', 'No', 'I'm 25', 'My name is Marius')."
        )
        user = f'Patient said: "{user_text}"'
        result = self._ollama_structured(system, user, IS_QUESTION_SCHEMA)
        return result.get("is_question", False)

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
            "format": schema
        }
        r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        raw = r.json()["message"]["content"]
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Structured LLM call returned invalid JSON: {e}\nOutput: {raw}")