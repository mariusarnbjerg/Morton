"""
Answer validation adapter using Ollama structured outputs.

Validates patient responses to ensure they're actually answering the question
or detects if they're asking a question to the chatbot.
"""

from __future__ import annotations
from typing import Literal, Optional
import requests
import json


class AnswerValidationResult:
    """Result of answer validation"""

    def __init__(
            self,
            classification: Literal["valid_answer", "chatbot_question", "invalid_answer"],
            explanation: str,
            suggested_response: Optional[str] = None
    ):
        self.classification = classification
        self.explanation = explanation
        self.suggested_response = suggested_response

    def is_valid_answer(self) -> bool:
        return self.classification == "valid_answer"

    def is_chatbot_question(self) -> bool:
        return self.classification == "chatbot_question"

    def is_invalid_answer(self) -> bool:
        return self.classification == "invalid_answer"

    def __repr__(self):
        return f"AnswerValidationResult(classification={self.classification}, explanation={self.explanation})"


class OllamaAnswerValidator:
    """
    Validates patient answers using Ollama with structured outputs.

    Uses constrained generation to ensure the model returns one of three classifications:
    - valid_answer: Patient answered the question
    - chatbot_question: Patient is asking a question (switch to CHAT mode)
    - invalid_answer: Patient didn't answer the question properly
    """

    def __init__(
            self,
            model: str = "llama3.1",
            base_url: str = "http://localhost:11434",
            timeout_s: int = 30,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

        # JSON Schema for structured output
        self.schema = {
            "type": "object",
            "properties": {
                "classification": {
                    "type": "string",
                    "enum": ["valid_answer", "chatbot_question", "invalid_answer"],
                    "description": (
                        "valid_answer: Patient provided a direct answer to the question. "
                        "chatbot_question: Patient is asking a question (e.g., starts with 'why', 'what', 'how', has '?'). "
                        "invalid_answer: Patient's response is off-topic, unclear, or doesn't answer the question."
                    )
                },
                "explanation": {
                    "type": "string",
                    "description": "Brief explanation of why this classification was chosen (1-2 sentences)"
                },
                "suggested_response": {
                    "type": "string",
                    "description": "For invalid_answer: a helpful prompt to guide the patient. For others: empty string."
                }
            },
            "required": ["classification", "explanation", "suggested_response"]
        }

    def validate(
            self,
            question: str,
            user_input: str,
            context: Optional[str] = None
    ) -> AnswerValidationResult:
        """
        Validate a user's input against the current question.

        Args:
            question: The question being asked
            user_input: What the patient typed
            context: Optional additional context (e.g., previous Q&A)

        Returns:
            AnswerValidationResult with classification and details
        """
        # Build the system prompt
        system_prompt = (
            "You are a validation assistant for a medical questionnaire. "
            "Your job is to classify patient responses into one of three categories:\n\n"
            "1. valid_answer: The patient directly answered the question asked. "
            "Even if brief or simple (like 'yes', 'no', 'John', '45'), it's valid if it answers the question.\n\n"
            "2. chatbot_question: The patient is asking a question instead of answering. "
            "Look for question words (why, what, how, when, where, who), question marks, "
            "or phrases like 'I don't understand', 'can you explain', etc.\n\n"
            "3. invalid_answer: The patient's response is off-topic, nonsensical, or doesn't answer the question. "
            "For example, answering 'yes' to 'What is your name?' or giving unrelated information.\n\n"
            "Be lenient - if there's any reasonable interpretation that the patient answered, classify as valid_answer."
        )

        # Build the user prompt
        user_prompt = f"""Question being asked: "{question}"

Patient's response: "{user_input}"

Classify this response."""

        if context:
            user_prompt += f"\n\nAdditional context:\n{context}"

        # Make the API call with structured output
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": self.schema  # Ollama structured output!
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout_s
            )
            response.raise_for_status()

            # Parse the response
            data = response.json()
            result = json.loads(data["message"]["content"])

            # Create and return result object
            return AnswerValidationResult(
                classification=result["classification"],
                explanation=result["explanation"],
                suggested_response=result.get("suggested_response", "")
            )

        except Exception as e:
            # Fallback: if validation fails, assume it's a valid answer
            # (fail open rather than fail closed)
            print(f"⚠️  Validation error: {e}")
            print(f"   Treating as valid answer (fail-safe)")
            return AnswerValidationResult(
                classification="valid_answer",
                explanation=f"Validation service error: {str(e)}",
                suggested_response=""
            )


# ============================================================================
# Example usage / testing
# ============================================================================

if __name__ == "__main__":
    validator = OllamaAnswerValidator()

    # Test cases
    test_cases = [
        {
            "question": "What is your name?",
            "inputs": [
                "John Doe",  # valid_answer
                "My name is Sarah",  # valid_answer
                "Why do you need my name?",  # chatbot_question
                "What will you do with this info?",  # chatbot_question
                "Yes",  # invalid_answer (doesn't answer "what is your name")
                "I like pizza",  # invalid_answer
            ]
        },
        {
            "question": "Do you have any allergies?",
            "inputs": [
                "No",  # valid_answer
                "Yes, penicillin",  # valid_answer
                "Pollen and dust",  # valid_answer
                "Why are you asking about allergies?",  # chatbot_question
                "What happens if I have allergies?",  # chatbot_question
                "John Doe",  # invalid_answer
            ]
        }
    ]

    for test in test_cases:
        question = test["question"]
        print(f"\n{'=' * 60}")
        print(f"Question: {question}")
        print('=' * 60)

        for user_input in test["inputs"]:
            result = validator.validate(question, user_input)

            emoji = {
                "valid_answer": "✅",
                "chatbot_question": "❓",
                "invalid_answer": "❌"
            }[result.classification]

            print(f"\n{emoji} Input: '{user_input}'")
            print(f"   Classification: {result.classification}")
            print(f"   Explanation: {result.explanation}")
            if result.suggested_response:
                print(f"   Suggested: {result.suggested_response}")