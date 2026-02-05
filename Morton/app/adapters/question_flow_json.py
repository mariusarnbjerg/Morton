'''What: Load questions from questions.json and serve them sequentially.
Why: This is the first of your “three big parts” (standardized questions), and it’s now isolated from UI and LLM concerns.'''

from __future__ import annotations
import json
from typing import Optional
from app.domain.models import Conversation, Question
from app.interfaces.question_flow import IQuestionFlow

class JsonQuestionFlow(IQuestionFlow):
    def __init__(self, questions_path: str):
        with open(questions_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        q_list = raw["questions"] if isinstance(raw, dict) and "questions" in raw else raw
        self._questions = [Question(**q) for q in q_list]

    def get_question(self, conv: Conversation) -> Optional[Question]:
        if conv.question_index < 0 or conv.question_index >= len(self._questions):
            return None
        return self._questions[conv.question_index]

    def advance_with_answer(self, conv: Conversation, answer_text: str) -> Conversation:
        # Step 1: always move forward; validation comes later.
        conv.question_index += 1
        conv.active_question_id = None
        return conv
