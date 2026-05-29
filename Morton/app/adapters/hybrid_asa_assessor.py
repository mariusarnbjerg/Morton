"""
Hybrid ASA Assessor — LLM makes a final ASA assessment informed by the ML prediction.

Takes the ML prediction (class + probabilities) and the full transcript,
asks the LLM to make its own ASA assessment, and explains whether it
agrees or disagrees with the ML model.
"""

from __future__ import annotations
from typing import Any, Dict, List

from app.domain.models import Message, Role


# =============================================================================
# Structured output schema for the hybrid assessment
# =============================================================================

HYBRID_ASA_SCHEMA = {
    "type": "object",
    "properties": {
        "asa_class": {
            "type": "string",
            "enum": ["ASA-I", "ASA-II", "ASA-III"],
            "description": "Your assessed ASA physical status classification."
        },
        "reasoning": {
            "type": "string",
            "description": "One to two sentences explaining your assessment and whether you agree or disagree with the ML prediction, and why."
        }
    },
    "required": ["asa_class", "reasoning"]
}


# =============================================================================
# Prompt construction
# =============================================================================

HYBRID_ASA_SYSTEM_PROMPT = (
    "You are a clinical decision support assistant. You are given a pre-anesthesia "
    "consultation transcript and a machine learning model's ASA prediction with "
    "class probabilities. Make your own ASA physical status assessment based on "
    "the full conversation. You may agree or disagree with the ML prediction. "
    "ASA-I: healthy patient. ASA-II: mild systemic disease. ASA-III: severe systemic disease. "
    "Be brief — one to two sentences of reasoning."
)


def build_hybrid_prompt(
    transcript: List[Message],
    ml_prediction: Dict[str, Any],
) -> str:
    """Build the user prompt with transcript and ML prediction context."""
    lines = []
    for m in transcript:
        if m.role in (Role.PATIENT, Role.ASSISTANT):
            speaker = "Patient" if m.role == Role.PATIENT else "Assistant"
            lines.append(f"{speaker}: {m.content}")
    transcript_text = "\n".join(lines)

    prob_str = ", ".join(
        f"{cls}: {prob:.0%}"
        for cls, prob in ml_prediction["probabilities"].items()
    )

    return (
        f"ML model prediction: {ml_prediction['asa_class']} "
        f"(confidence: {ml_prediction['confidence']:.0%})\n"
        f"Class probabilities: {prob_str}\n\n"
        f"Consultation transcript:\n{transcript_text}\n"
    )