"""
Conversational orchestrator - manages state, delegates language to ILLMClient.

The orchestrator coordinates the conversation flow:
  1. Receives patient messages
  2. Delegates language tasks to the LLM client (extract, reply, confirm, detect question)
  3. Updates conversation state based on structured results
  4. Returns the assistant's reply

Python manages all state. The LLM only handles language.

Key design rule for follow-ups:
  When a follow-up is activated, the PARENT question is immediately marked complete.
  The follow-up then becomes an independent question in its own right.
  This prevents the extractor from being asked about a question that is already answered.
"""

from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from app.domain.models import (
    Conversation, ConversationState, Message, Role, Question, FollowUpQuestion,
    FREE_CHAT_QUESTION
)
from app.interfaces.question_flow import IQuestionFlow
from app.interfaces.transcript_store import ITranscriptStore
from app.interfaces.summarizer import ISummarizer
from app.interfaces.llm_client import ILLMClient


@dataclass
class OrchestratorResult:
    bot_text: str
    done: bool = False


class ConversationOrchestrator:
    def __init__(
        self,
        question_flow: IQuestionFlow,
        transcript_store: ITranscriptStore,
        llm_client: ILLMClient,
        summarizer: Optional[ISummarizer] = None,
        summary_schema_path: Optional[str] = None,
    ):
        self.question_flow = question_flow
        self.store = transcript_store
        self.llm = llm_client
        self.summarizer = summarizer
        self.summary_schema_path = summary_schema_path

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def start(self, conv: Conversation) -> OrchestratorResult:
        """Deliver greeting and auto-complete q0 so the first patient turn starts on q1."""
        opening_q = self.question_flow.get_question(conv)
        if not opening_q:
            return OrchestratorResult(bot_text="No questions configured.", done=True)

        reply = opening_q.text

        self.store.append(conv.conversation_id, Message(
            role=Role.ASSISTANT, content=reply, meta={"question_id": opening_q.id}
        ))
        return OrchestratorResult(bot_text=reply)

    def handle_user_message(self, conv: Conversation, user_text: str) -> OrchestratorResult:
        self.store.append(conv.conversation_id, Message(role=Role.PATIENT, content=user_text))

        # Determine what we're currently trying to answer
        active_fu = self.question_flow.get_active_follow_up(conv)
        current_q = self.question_flow.get_question(conv)
        focus = active_fu if active_fu else current_q

        # Call 1: Extract
        # Skip extraction when awaiting confirmation — patient is confirming, not answering a question
        # Skip extraction when in free chat — patient is chatting freely, not answering a question
        if conv.state == ConversationState.AWAITING_CONFIRMATION:
            extraction = None
        elif conv.state == ConversationState.FREE_CHAT:
            extraction = None
        else:
            extraction = self.llm.extract(focus, user_text) if focus else None
        print(f"[DEBUG] focus: {focus.id if focus else None} | state: {conv.state.value} | extraction: {extraction}")

        # Update state (pure Python) — returns instruction for reply
        instruction = self._update_state(conv, focus, active_fu, extraction, user_text)

        # Call 2: Reply
        transcript = self.store.get(conv.conversation_id)
        patient_name = conv.answers.get("q1", "")
        reply = self.llm.generate_reply(transcript, instruction, patient_name)

        # For deterministic actions, append the question text in Python
        action = instruction.get("action")
        question_text = instruction.get("question_text")
        if action in ("ask_next", "ask_followup") and question_text:
            reply = f"{reply}\n\n{question_text}" if reply.strip() else question_text
        elif action == "reentry" and question_text:
            reply = question_text

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
        if not self.summary_schema_path:
            raise RuntimeError("summary_schema_path not configured.")
        with open(self.summary_schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        transcript = self.store.get(conv.conversation_id)
        summary = self.summarizer.summarize(transcript=transcript, schema=schema)
        summary["questionnaire_answers"] = self._build_questionnaire_answers(conv)
        summary["patient_questions"] = self._build_patient_questions(transcript)
        return summary

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
          action:      ask_next | ask_followup | free_chat | reentry | confirm | done
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
        # FREE_CHAT — patient is chatting freely, waiting to signal readiness
        # ------------------------------------------------------------------
        if conv.state == ConversationState.FREE_CHAT:
            free_chat_extraction = self.llm.extract(FREE_CHAT_QUESTION, user_text)
            print(f"[DEBUG] FREE_CHAT extraction: {free_chat_extraction}")
            if free_chat_extraction.get("is_complete", False):
                conv.state = ConversationState.IN_PROGRESS
                # q0 is a greeting — never re-ask it, just advance to q1
                if focus and focus.id == "q0":
                    if focus.id not in conv.completed_question_ids:
                        conv.completed_question_ids.append(focus.id)
                    self.question_flow.advance(conv)
                    return self._next_question_instruction(conv, "", user_text, focus.text if focus else "")
                return {
                    "action": "reentry",
                    "question_text": focus.text if focus else None,
                    "question_id": focus.id if focus else None,
                    "acknowledged_answer": ""
                }
            else:
                return {
                    "action": "free_chat",
                    "question_text": focus.text if focus else None,
                    "question_id": focus.id if focus else None,
                    "acknowledged_answer": ""
                }

        # ------------------------------------------------------------------
        # Handle confirmation state first — patient is confirming a summary
        # ------------------------------------------------------------------
        if conv.state == ConversationState.AWAITING_CONFIRMATION:
            # Gather context for the confirmation check
            transcript = self.store.get(conv.conversation_id)
            last_bot = next(
                (m.content for m in reversed(transcript) if m.role == Role.ASSISTANT),
                ""
            )
            current_answer = conv.answers.get(conv.pending_confirmation_question_id, "") if conv.pending_confirmation_question_id else ""

            confirm_result = self.llm.check_confirmation(user_text, last_bot, current_answer)
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
                return self._next_question_instruction(conv, acknowledged, user_text, focus.text if focus else "")
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
            conv.state = ConversationState.FREE_CHAT
            return {
                "action": "free_chat",
                "question_text": focus.text if focus else None,
                "question_id": focus.id if focus else None,
                "acknowledged_answer": acknowledged
            }

        # Answer is complete — mark it and ensure we're back in normal flow
        conv.state = ConversationState.IN_PROGRESS
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
                        "acknowledged_answer": acknowledged,
                        "full_message": user_text,
                        "current_question_text": focus.text if focus else ""
                    }
                if parent_q.confirmation_required and parent_q.id not in conv.completed_question_ids:
                    conv.pending_confirmation_question_id = parent_q.id
                    conv.state = ConversationState.AWAITING_CONFIRMATION
                    return {
                        "action": "confirm",
                        "question_text": None,
                        "question_id": parent_q.id,
                        "acknowledged_answer": acknowledged,
                        "confirm_answer": conv.answers.get(parent_q.id, acknowledged),
                    }

                if parent_q.id not in conv.completed_question_ids:
                    conv.completed_question_ids.append(parent_q.id)
                self.question_flow.advance(conv)
            return self._next_question_instruction(conv, acknowledged, user_text, focus.text if focus else "")

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
                        "acknowledged_answer": acknowledged,
                        "full_message": user_text
                    }

        current_q = self.question_flow.get_question(conv)
        if current_q and current_q.confirmation_required and current_q.id == focus.id:
            conv.pending_confirmation_question_id = current_q.id
            conv.state = ConversationState.AWAITING_CONFIRMATION
            return {
                "action": "confirm",
                "question_text": None,
                "question_id": current_q.id,
                "acknowledged_answer": acknowledged,
                "confirm_answer": conv.answers.get(current_q.id, acknowledged),
            }

        self.question_flow.advance(conv)
        return self._next_question_instruction(conv, acknowledged, user_text, focus.text if focus else "")

    def _next_question_instruction(self, conv: Conversation, acknowledged: str, full_message: str = "", current_question_text: str = "") -> Dict[str, Any]:
        next_q = self.question_flow.get_question(conv)
        if not next_q:
            return {
                "action": "done",
                "question_text": None,
                "question_id": None,
                "acknowledged_answer": acknowledged,
                "full_message": full_message,
                "current_question_text": current_question_text
            }
        return {
            "action": "ask_next",
            "question_text": next_q.text,
            "question_id": next_q.id,
            "acknowledged_answer": acknowledged,
            "full_message": full_message,
            "current_question_text": current_question_text
        }

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