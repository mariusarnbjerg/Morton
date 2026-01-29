'''What: The orchestrator decides what the system says next (question flow for now).
Why: This is the “single brain” that keeps your modules independent: question-flow doesn’t call chatbot, chatbot doesn’t call summarizer, etc.'''

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from app.domain.models import Conversation, ConversationState, Message, Role, Mode
from app.interfaces.chatbot import IChatbot
from app.interfaces.question_flow import IQuestionFlow
from app.interfaces.transcript_store import ITranscriptStore
from app.interfaces.summarizer import ISummarizer
import json

@dataclass
class OrchestratorResult:
    bot_text: Optional[str] = None
    done: bool = False


class ConversationOrchestrator:
    def __init__(self, question_flow: IQuestionFlow, transcript_store: ITranscriptStore, chatbot:Optional[IChatbot] = None, summarizer: Optional[ISummarizer] = None, template_path: Optional[str] = None):
        self.question_flow = question_flow
        self.store = transcript_store
        self.chatbot = chatbot
        self.summarizer = summarizer
        self.template_path = template_path


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

        # NEW: handle chat interruptions
        if mode == Mode.CHAT:
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
                f"Current standardized question: '{current_q.text}' (id={current_q.id}).\n"
                f"Standardized questions answered so far:\n{answered_lines}\n"
                "If the user asks what has been asked/answered so far, use this list."
            )

            answer = self.chatbot.answer(user_text=user_text, transcript=transcript, context=ctx)

            self.store.append(
                conv.conversation_id,
                Message(role=Role.ASSISTANT, content=answer, meta={"mode": "chat"})
            )

            # 2) Return to flow (same question_index), but return BOTH texts to the UI
            q = self.question_flow.get_question(conv)
            if not q:
                conv.state = ConversationState.DONE
                return OrchestratorResult(bot_text=answer + "\n\nQuestionnaire complete.", done=True)

            conv.state = ConversationState.FLOW_WAITING_ANSWER
            conv.active_question_id = q.id

            # Store that we are re-showing the current question
            self.store.append(
                conv.conversation_id,
                Message(role=Role.SYSTEM, content=q.text, meta={"question_id": q.id, "reask": True})
            )

            combined = f"{answer}\n\n---\nBack to the questionnaire:\n{q.text}"
            return OrchestratorResult(bot_text=combined, done=False)

        # ANSWER branch (standardized flow)
        if conv.state != ConversationState.FLOW_WAITING_ANSWER:
            conv.state = ConversationState.FLOW_WAITING_ANSWER

        qid = conv.active_question_id
        if qid:
            conv.answers[qid] = user_text

        # Advance to next question
        self.question_flow.advance_with_answer(conv, user_text)

        # Ask next (or finish)
        return self._ask_current_question(conv)

    def finalize(self, conv: Conversation) -> dict:
        if not self.summarizer:
            raise RuntimeError("Summarizer not configured.")
        if not self.template_path:
            raise RuntimeError("template_path not configured.")

        with open(self.template_path, "r", encoding="utf-8") as f:
            template = json.load(f)

        transcript = self.store.get(conv.conversation_id)
        summary = self.summarizer.summarize(transcript=transcript, template=template)

        # Deterministic questionnaire answers:
        summary["questionnaire_answers"] = self.build_questionnaire_answers(conv)

        summary["patient_questions"] = self.build_patient_questions(transcript)

        return summary

    def build_questionnaire_answers(self, conv: Conversation) -> list[dict]:
        '''What: Pair each question’s id/text with conv.answers[id].
        Why: 100% correct, no hallucination.'''
        items = []
        # iterate all questions in order using question_flow adapter data
        idx = 0
        while True:
            tmp = type(conv)(conversation_id=conv.conversation_id)  # make a lightweight clone
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
        # if last question had no answer, keep it with empty answer
        if pending_q is not None:
            out.append({"question": pending_q, "answer": ""})
        return out
