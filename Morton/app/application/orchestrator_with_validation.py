"""
Updated orchestrator with answer validation.

Changes:
- Added answer validation before processing
- Auto-detects when user asks a question
- Prompts user to answer properly if input is invalid
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from app.domain.models import Conversation, ConversationState, Message, Role, Mode
from app.interfaces.chatbot import IChatbot
from app.interfaces.question_flow import IQuestionFlow
from app.interfaces.transcript_store import ITranscriptStore
from app.interfaces.summarizer import ISummarizer
from app.adapters.ollama_answer_validator import OllamaAnswerValidator
import json


@dataclass
class OrchestratorResult:
    bot_text: Optional[str] = None
    done: bool = False
    validation_failed: bool = False  # NEW: indicates validation failure


class ConversationOrchestrator:
    def __init__(
            self,
            question_flow: IQuestionFlow,
            transcript_store: ITranscriptStore,
            chatbot: Optional[IChatbot] = None,
            summarizer: Optional[ISummarizer] = None,
            template_path: Optional[str] = None,
            enable_validation: bool = True  # NEW: can disable for testing
    ):
        self.question_flow = question_flow
        self.store = transcript_store
        self.chatbot = chatbot
        self.summarizer = summarizer
        self.template_path = template_path
        self.enable_validation = enable_validation

        # NEW: Initialize validator if enabled
        if enable_validation:
            self.validator = OllamaAnswerValidator()
        else:
            self.validator = None

    def _ask_current_question(self, conv: Conversation) -> OrchestratorResult:
        q = self.question_flow.get_question(conv)
        if not q:
            conv.state = ConversationState.DONE
            self.store.append(conv.conversation_id, Message(role=Role.SYSTEM, content="Questionnaire complete."))
            return OrchestratorResult(bot_text="Questionnaire complete.", done=True)

        conv.state = ConversationState.FLOW_WAITING_ANSWER
        conv.active_question_id = q.id
        self.store.append(conv.conversation_id, Message(
            role=Role.SYSTEM,
            content=q.text,
            meta={"question_id": q.id, "channel": "questionnaire"}
        ))
        return OrchestratorResult(bot_text=q.text)

    def start(self, conv: Conversation) -> OrchestratorResult:
        return self._ask_current_question(conv)

    def handle_user_message(self, conv: Conversation, user_text: str, mode: Mode = Mode.ANSWER) -> OrchestratorResult:
        # Always store patient message
        self.store.append(conv.conversation_id,
                          Message(role=Role.PATIENT, content=user_text, meta={"mode": mode.value}))

        # If mode is explicitly CHAT, handle as before
        if mode == Mode.CHAT:
            return self._handle_chat_mode(conv, user_text)

        # NEW: If mode is ANSWER, validate first
        if mode == Mode.ANSWER and self.validator:
            return self._handle_answer_with_validation(conv, user_text)
        else:
            # Fallback: no validation (same as before)
            return self._handle_answer_without_validation(conv, user_text)

    def _handle_answer_with_validation(self, conv: Conversation, user_text: str) -> OrchestratorResult:
        """
        Handle answer with validation (NEW).

        Validates the answer, then:
        - valid_answer: Process normally
        - chatbot_question: Switch to CHAT mode
        - invalid_answer: Ask user to answer the question
        """
        current_q = self.question_flow.get_question(conv)
        if not current_q:
            # No current question, just process normally
            return self._handle_answer_without_validation(conv, user_text)

        # Validate the answer
        validation = self.validator.validate(
            question=current_q.text,
            user_input=user_text
        )

        print(f"🔍 Validation: {validation.classification} - {validation.explanation}")

        # Handle based on classification
        if validation.is_valid_answer():
            # Process as normal answer
            return self._handle_answer_without_validation(conv, user_text)

        elif validation.is_chatbot_question():
            # User is asking a question - switch to CHAT mode
            print(f"   → Switching to CHAT mode")
            return self._handle_chat_mode(conv, user_text)

        elif validation.is_invalid_answer():
            # Invalid answer - prompt user to answer properly
            print(f"   → Invalid answer, prompting user")

            # Create helpful response
            bot_response = (
                f"I noticed your response doesn't seem to answer the question. "
                f"The question is: '{current_q.text}'\n\n"
            )

            if validation.suggested_response:
                bot_response += validation.suggested_response
            else:
                bot_response += "Could you please provide an answer to this question?"

            # Store the validation failure message
            self.store.append(
                conv.conversation_id,
                Message(role=Role.ASSISTANT, content=bot_response, meta={"validation_failed": True})
            )

            # Re-ask the same question
            self.store.append(
                conv.conversation_id,
                Message(
                    role=Role.SYSTEM,
                    content=current_q.text,
                    meta={"question_id": current_q.id, "reask": True, "reason": "validation_failed"}
                )
            )

            combined = f"{bot_response}\n\n---\nPlease answer:\n{current_q.text}"

            return OrchestratorResult(
                bot_text=combined,
                done=False,
                validation_failed=True
            )

    def _handle_answer_without_validation(self, conv: Conversation, user_text: str) -> OrchestratorResult:
        """Original answer handling (no validation)"""
        if conv.state != ConversationState.FLOW_WAITING_ANSWER:
            conv.state = ConversationState.FLOW_WAITING_ANSWER

        qid = conv.active_question_id
        if qid:
            conv.answers[qid] = user_text

        # Advance to next question
        self.question_flow.advance_with_answer(conv, user_text)

        # Ask next (or finish)
        return self._ask_current_question(conv)

    def _handle_chat_mode(self, conv: Conversation, user_text: str) -> OrchestratorResult:
        """Handle CHAT mode (same as before)"""
        if not self.chatbot:
            self.store.append(conv.conversation_id, Message(role=Role.ASSISTANT, content="Chatbot not available."))
            return self._ask_current_question(conv)

        conv.state = ConversationState.CHAT_MODE
        transcript = self.store.get(conv.conversation_id)

        current_q = self.question_flow.get_question(conv)
        answered = self.build_questionnaire_answers(conv)

        answered_lines = "\n".join(
            f"- {x['question']} -> {x['answer']}" for x in answered
        ) or "None yet."

        ctx = (
            f"IMPORTANT CONTEXT:\n"
            f"The patient is currently being asked this standardized question:\n"
            f"  → \"{current_q.text}\" (question id: {current_q.id})\n\n"
            f"When the patient asks 'why do you need to know this?' or refers to 'this question', "
            f"they are referring to THIS CURRENT QUESTION: \"{current_q.text}\"\n\n"
            f"Previously answered standardized questions:\n{answered_lines}\n\n"
        )

        answer = self.chatbot.answer(user_text=user_text, transcript=transcript, context=ctx)

        self.store.append(
            conv.conversation_id,
            Message(role=Role.ASSISTANT, content=answer, meta={"mode": "chat"})
        )

        # Return to flow
        q = self.question_flow.get_question(conv)
        if not q:
            conv.state = ConversationState.DONE
            return OrchestratorResult(bot_text=answer + "\n\nConsultation complete.", done=True)

        conv.state = ConversationState.FLOW_WAITING_ANSWER
        conv.active_question_id = q.id

        self.store.append(
            conv.conversation_id,
            Message(role=Role.SYSTEM, content=q.text, meta={"question_id": q.id, "reask": True})
        )

        combined = f"{answer}\n\n---\nBack to the question:\n{q.text}"
        return OrchestratorResult(bot_text=combined, done=False)

    # Rest of the methods stay the same...

    def finalize(self, conv: Conversation) -> dict:
        if not self.summarizer:
            raise RuntimeError("Summarizer not configured.")
        if not self.template_path:
            raise RuntimeError("template_path not configured.")

        with open(self.template_path, "r", encoding="utf-8") as f:
            template = json.load(f)

        transcript = self.store.get(conv.conversation_id)
        summary = self.summarizer.summarize(transcript=transcript, schema=template)

        summary["questionnaire_answers"] = self.build_questionnaire_answers(conv)
        summary["patient_questions"] = self.build_patient_questions(transcript)

        return summary

    def build_questionnaire_answers(self, conv: Conversation) -> list[dict]:
        items = []
        idx = 0
        while True:
            tmp = type(conv)(conversation_id=conv.conversation_id)
            tmp.question_index = idx
            q = self.question_flow.get_question(tmp)
            if not q:
                break
            ans = conv.answers.get(q.id)
            if ans is not None:
                items.append({"question_id": q.id, "question": q.text, "answer": ans})
            idx += 1
        return items

    def build_patient_questions(self, transcript: list[Message]) -> list[dict]:
        out = []
        pending_q = None
        for m in transcript:
            mode = (m.meta or {}).get("mode")
            if m.role == Role.PATIENT and mode == "chat":
                pending_q = m.content
            elif m.role == Role.ASSISTANT and mode == "chat" and pending_q is not None:
                out.append({"question": pending_q, "answer": m.content})
                pending_q = None
        if pending_q is not None:
            out.append({"question": pending_q, "answer": ""})
        return out