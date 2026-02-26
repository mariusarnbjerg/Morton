"""
Conversational orchestrator - two focused LLM calls per turn.

Call 1 - EXTRACT (tiny structured schema):
  Given the current question and patient message, extract what was answered.

Call 2 - REPLY (free text):
  Given exactly what to do next, generate a natural reply.

Python manages all state. The LLM only handles language.

Key design rule for follow-ups:
  When a follow-up is activated, the PARENT question is immediately marked complete.
  The follow-up then becomes an independent question in its own right.
  This prevents the extractor from being asked about a question that is already answered.
"""

from __future__ import annotations
import json
import requests
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from app.domain.models import (
    Conversation, ConversationState, Message, Role, Question, FollowUpQuestion
)
from app.interfaces.question_flow import IQuestionFlow
from app.interfaces.transcript_store import ITranscriptStore
from app.interfaces.summarizer import ISummarizer


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


@dataclass
class OrchestratorResult:
    bot_text: str
    done: bool = False


class ConversationOrchestrator:
    def __init__(
        self,
        question_flow: IQuestionFlow,
        transcript_store: ITranscriptStore,
        summarizer: Optional[ISummarizer] = None,
        template_path: Optional[str] = None,
        model: str = "llama3.1",
        base_url: str = "http://localhost:11434",
        timeout_s: int = 60,
    ):
        self.question_flow = question_flow
        self.store = transcript_store
        self.summarizer = summarizer
        self.template_path = template_path
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def start(self, conv: Conversation) -> OrchestratorResult:
        """Deliver greeting and auto-complete q0 so the first patient turn starts on q1."""
        opening_q = self.question_flow.get_question(conv)
        if not opening_q:
            return OrchestratorResult(bot_text="No questions configured.", done=True)

        reply = opening_q.text
        conv.completed_question_ids.append(opening_q.id)
        self.question_flow.advance(conv)

        self.store.append(conv.conversation_id, Message(
            role=Role.ASSISTANT, content=reply, meta={"question_id": opening_q.id}
        ))
        return OrchestratorResult(bot_text=reply)

    def handle_user_message(self, conv: Conversation, user_text: str) -> OrchestratorResult:
        self.store.append(conv.conversation_id, Message(role=Role.PATIENT, content=user_text))

        # patient_question detection happens after extraction (see below)

        # Determine what we're currently trying to answer
        active_fu = self.question_flow.get_active_follow_up(conv)
        current_q = self.question_flow.get_question(conv)
        focus = active_fu if active_fu else current_q

        # Call 1: Extract
        # Skip extraction when awaiting confirmation — patient is confirming, not answering a question
        if conv.state == ConversationState.AWAITING_CONFIRMATION:
            extraction = None
        else:
            extraction = self._call_extract(focus, user_text) if focus else None
        print(f"[DEBUG] focus: {focus.id if focus else None} | state: {conv.state.value} | extraction: {extraction}")

        # Update state (pure Python) — returns instruction for reply
        instruction = self._update_state(conv, focus, active_fu, extraction, user_text)

        # Call 1b: Detect patient question — separate focused call, only when message is conversational
        # We only bother if the extraction didn't fully answer the question, or if there's no clear answer
        extracted_answer = (extraction or {}).get("answer", "").strip()
        looks_conversational = (
            not extracted_answer or
            any(w in user_text.lower() for w in ("why", "what", "how", "will", "should", "can you", "don't understand", "?"))
        )
        if looks_conversational:
            is_q_result = self._call_is_question(user_text)
            patient_question = user_text.strip() if is_q_result else ""
        else:
            patient_question = ""
        instruction["patient_question"] = patient_question

        print(f"[DEBUG] instruction: {instruction}")

        # Call 2: Reply
        transcript = self.store.get(conv.conversation_id)
        reply = self._call_reply(conv, transcript, instruction)
        print(f"[DEBUG] reply: {reply[:100]}\n")

        self.store.append(conv.conversation_id, Message(role=Role.ASSISTANT, content=reply))

        unanswered = self.question_flow.get_unanswered_required_ids(conv)
        if not unanswered and conv.state != ConversationState.AWAITING_CONFIRMATION:
            conv.state = ConversationState.DONE
            return OrchestratorResult(bot_text=reply, done=True)

        return OrchestratorResult(bot_text=reply, done=False)

    def finalize(self, conv: Conversation) -> dict:
        if not self.summarizer:
            raise RuntimeError("Summarizer not configured.")
        if not self.template_path:
            raise RuntimeError("template_path not configured.")
        with open(self.template_path, "r", encoding="utf-8") as f:
            template = json.load(f)
        transcript = self.store.get(conv.conversation_id)
        summary = self.summarizer.summarize(transcript=transcript, schema=template)
        summary["questionnaire_answers"] = self._build_questionnaire_answers(conv)
        summary["patient_questions"] = self._build_patient_questions(transcript)
        return summary

    # -----------------------------------------------------------------------
    # Call 1: Extract
    # -----------------------------------------------------------------------

    def _call_extract(self, focus, user_text: str) -> Dict[str, Any]:
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

    # -----------------------------------------------------------------------
    # State update — pure Python, no LLM
    # -----------------------------------------------------------------------

    def _update_state(
        self,
        conv: Conversation,
        focus,
        active_fu: Optional[FollowUpQuestion],
        extraction: Optional[Dict],
        user_text: str = ""
    ) -> Dict[str, Any]:
        """
        Apply extraction results to conversation state.
        Returns an instruction dict telling the reply call exactly what to do:
          action:      ask_next | ask_followup | reask | confirm | done
          question_text: the exact text to ask next (or None)
          question_id:   the id of the question being asked next
          acknowledged_answer: what the patient just answered (for natural transitions)
        """
        acknowledged = extraction.get("answer", "").strip() if extraction else ""
        is_complete = extraction.get("is_complete", False) if extraction else False
        needs_followup = extraction.get("needs_followup", False) if extraction else False

        # Store answer if we got something meaningful
        if acknowledged and focus:
            conv.answers[focus.id] = acknowledged

        # ------------------------------------------------------------------
        # Handle confirmation state first — patient is confirming a summary
        # ------------------------------------------------------------------
        if conv.state == ConversationState.AWAITING_CONFIRMATION:
            confirm_result = self._call_confirm(user_text, conv.pending_confirmation_question_id, conv)
            print(f"[DEBUG] confirm_result: {confirm_result}")

            if confirm_result.get("confirmed", False):
                confirmed_id = conv.pending_confirmation_question_id

                # If patient added new information while confirming, append it to the stored answer
                additional = confirm_result.get("additional_info", "").strip()
                if additional and confirmed_id:
                    existing = conv.answers.get(confirmed_id, "")
                    conv.answers[confirmed_id] = f"{existing}; {additional}" if existing else additional
                    print(f"[DEBUG] Patient added info during confirmation: {additional}")

                if confirmed_id and confirmed_id not in conv.completed_question_ids:
                    conv.completed_question_ids.append(confirmed_id)
                conv.pending_confirmation_question_id = None
                conv.state = ConversationState.IN_PROGRESS
                current_q = self.question_flow.get_question(conv)
                if current_q and current_q.id == confirmed_id:
                    self.question_flow.advance(conv)
                return self._next_question_instruction(conv, acknowledged)
            else:
                return {
                    "action": "reask",
                    "question_text": focus.text if focus else None,
                    "question_id": focus.id if focus else None,
                    "acknowledged_answer": ""
                }

        # ------------------------------------------------------------------
        # Normal flow
        # ------------------------------------------------------------------

        if not is_complete:
            return {
                "action": "reask",
                "question_text": focus.text if focus else None,
                "question_id": focus.id if focus else None,
                "acknowledged_answer": acknowledged
            }

        # Answer is complete — mark it
        if focus and focus.id not in conv.completed_question_ids:
            conv.completed_question_ids.append(focus.id)

        # ------------------------------------------------------------------
        # Was this a follow-up question?
        # ------------------------------------------------------------------
        if active_fu:
            conv.active_follow_up_id = None
            parent_q = self.question_flow.get_question(conv)

            if parent_q:
                remaining_fus = [
                    fu for fu in parent_q.follow_ups
                    if fu.id not in conv.completed_question_ids
                ]
                if remaining_fus:
                    conv.active_follow_up_id = remaining_fus[0].id
                    return {
                        "action": "ask_followup",
                        "question_text": remaining_fus[0].text,
                        "question_id": remaining_fus[0].id,
                        "acknowledged_answer": acknowledged
                    }

                if parent_q.confirmation_required and parent_q.id not in conv.completed_question_ids:
                    conv.pending_confirmation_question_id = parent_q.id
                    conv.state = ConversationState.AWAITING_CONFIRMATION
                    return {
                        "action": "confirm",
                        "question_text": None,
                        "question_id": parent_q.id,
                        "acknowledged_answer": acknowledged
                    }

                if parent_q.id not in conv.completed_question_ids:
                    conv.completed_question_ids.append(parent_q.id)
                self.question_flow.advance(conv)
            return self._next_question_instruction(conv, acknowledged)

        # ------------------------------------------------------------------
        # This was a top-level question
        # ------------------------------------------------------------------

        if needs_followup:
            current_q = self.question_flow.get_question(conv)
            if current_q and current_q.follow_ups:
                next_fu = next(
                    (fu for fu in current_q.follow_ups
                     if fu.id not in conv.completed_question_ids),
                    None
                )
                if next_fu:
                    if current_q.id not in conv.completed_question_ids:
                        conv.completed_question_ids.append(current_q.id)
                    conv.active_follow_up_id = next_fu.id
                    return {
                        "action": "ask_followup",
                        "question_text": next_fu.text,
                        "question_id": next_fu.id,
                        "acknowledged_answer": acknowledged
                    }

        current_q = self.question_flow.get_question(conv)
        if current_q and current_q.confirmation_required and current_q.id == focus.id:
            conv.pending_confirmation_question_id = current_q.id
            conv.state = ConversationState.AWAITING_CONFIRMATION
            return {
                "action": "confirm",
                "question_text": None,
                "question_id": current_q.id,
                "acknowledged_answer": acknowledged
            }

        self.question_flow.advance(conv)
        return self._next_question_instruction(conv, acknowledged)

    def _next_question_instruction(self, conv: Conversation, acknowledged: str) -> Dict[str, Any]:
        """Return an ask_next instruction for whatever question is up next, or done."""
        next_q = self.question_flow.get_question(conv)
        if not next_q:
            return {
                "action": "done",
                "question_text": None,
                "question_id": None,
                "acknowledged_answer": acknowledged
            }
        return {
            "action": "ask_next",
            "question_text": next_q.text,
            "question_id": next_q.id,
            "acknowledged_answer": acknowledged
        }

    # -----------------------------------------------------------------------
    # Call 2: Reply
    # -----------------------------------------------------------------------

    def _call_reply(
        self,
        conv: Conversation,
        transcript: List[Message],
        instruction: Dict[str, Any]
    ) -> str:
        """Generate a natural reply given a precise instruction about what to do next."""
        action = instruction["action"]
        question_text = instruction.get("question_text")
        acknowledged = instruction.get("acknowledged_answer", "")
        patient_question = instruction.get("patient_question", "")
        patient_name = conv.answers.get("q1", "")

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
            qid = instruction.get("question_id", "")
            answer = conv.answers.get(qid, acknowledged)
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

    # -----------------------------------------------------------------------
    # Shared Ollama helper
    # -----------------------------------------------------------------------

    def _call_is_question(self, user_text: str) -> bool:
        """
        Single-purpose call: is the patient asking a question?
        Only called when the message looks conversational.
        """
        system = (
            "You are classifying a single patient message. "
            "Return is_question=true if the patient is asking something — "
            "even if they also provided an answer. "
            "Return is_question=false if they are only answering (e.g. 'Yes', 'No', 'I'm 25', 'My name is Marius')."
        )
        user = f'Patient said: "{user_text}"'
        result = self._ollama_structured(system, user, IS_QUESTION_SCHEMA)
        return result.get("is_question", False)

    def _call_confirm(self, user_text: str, question_id: Optional[str], conv: Conversation) -> Dict[str, Any]:
        """
        Ask the LLM whether the patient is confirming the summary, and whether they added new info.
        Returns {"confirmed": bool, "additional_info": str}
        Handles both positive ("yes, correct") and negative ("no, I haven't") confirmations.
        """
        transcript = self.store.get(conv.conversation_id)
        last_bot = next(
            (m.content for m in reversed(transcript) if m.role == Role.ASSISTANT),
            ""
        )
        current_answer = conv.answers.get(question_id, "") if question_id else ""

        system = (
            "You are determining whether a patient is confirming a summary read back to them, "
            "and whether they added any new information. "
            "Consider context carefully — 'no, I haven't' in response to 'You haven't been told X, right?' is a confirmation. "
            "If the patient confirms AND adds new info (e.g. 'yes, and also penicillin'), "
            "set confirmed=true and put the new info in additional_info."
        )
        user = (
            f'The assistant just said: "{last_bot}"\n'
            f'Current recorded answer: "{current_answer}"\n'
            f'The patient responded: "{user_text}"\n\n'
            f"Is the patient confirming? Did they add any new information?"
        )
        result = self._ollama_structured(system, user, CONFIRM_SCHEMA)
        return result

    def _ollama_structured(self, system: str, user: str, schema: Dict) -> Dict[str, Any]:
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

    # -----------------------------------------------------------------------
    # Summary helpers
    # -----------------------------------------------------------------------

    def _build_questionnaire_answers(self, conv: Conversation) -> list:
        items = []
        for q in self.question_flow.get_all_questions():
            ans = conv.answers.get(q.id)
            if ans is not None:
                items.append({"question_id": q.id, "question": q.text, "answer": ans})
            for fu in q.follow_ups:
                fu_ans = conv.answers.get(fu.id)
                if fu_ans is not None:
                    items.append({"question_id": fu.id, "question": fu.text, "answer": fu_ans})
        return items

    def _build_patient_questions(self, transcript: List[Message]) -> list:
        out = []
        messages = [m for m in transcript if m.role in (Role.PATIENT, Role.ASSISTANT)]
        question_starters = ("why", "what", "how", "when", "where", "who", "can you", "could you")
        for i, m in enumerate(messages):
            if m.role == Role.PATIENT:
                text = m.content.strip()
                is_question = (
                    text.endswith("?") or
                    any(text.lower().startswith(w) for w in question_starters)
                )
                if is_question and i + 1 < len(messages) and messages[i + 1].role == Role.ASSISTANT:
                    out.append({"question": text, "answer": messages[i + 1].content})
        return out