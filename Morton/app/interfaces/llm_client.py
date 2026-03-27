from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from app.domain.models import Message, Question, FollowUpQuestion


class ILLMClient(ABC):
    """
    Interface for all LLM language tasks used by the orchestrator.

    The orchestrator manages conversation state and logic.
    The LLM client handles all natural language processing:
      - Extracting answers from patient messages
      - Generating conversational replies
      - Checking whether a patient confirmed a summary
      - Detecting whether a patient is asking a question
    """

    @abstractmethod
    def extract(
        self,
        focus: Union[Question, FollowUpQuestion],
        user_text: str,
    ) -> Dict[str, Any]:
        """
        Extract a structured answer from the patient's message.

        Returns:
            {
                "answer": str,         # Extracted answer (empty if not answered)
                "is_complete": bool,   # Whether the answer satisfies completion criteria
                "needs_followup": bool # Whether a follow-up question should be triggered
            }
        """
        pass

    @abstractmethod
    def generate_reply(
        self,
        transcript: List[Message],
        instruction: Dict[str, Any],
        patient_name: str = "",
    ) -> str:
        """
        Generate a natural conversational reply given a precise instruction.

        Args:
            transcript: Recent conversation history for context.
            instruction: Dict with keys 'action', 'question_text', 'question_id',
                         'acknowledged_answer', 'patient_question'.
            patient_name: The patient's name if known.

        Returns:
            The assistant's reply as a string.
        """
        pass

    @abstractmethod
    def check_confirmation(
        self,
        user_text: str,
        last_bot_message: str,
        current_answer: str,
    ) -> Dict[str, Any]:
        """
        Determine whether the patient is confirming a summary read back to them.

        Returns:
            {
                "confirmed": bool,       # Whether the patient confirmed
                "additional_info": str   # Any new information added beyond confirming
            }
        """
        pass