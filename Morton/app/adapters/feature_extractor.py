"""
Feature extractor — extracts structured ML features from a chatbot transcript.

Uses the LLM with structured output to convert conversational patient language
into the exact feature format the trained ASA prediction model expects.
"""

from __future__ import annotations
from typing import Any, Dict, List

from app.domain.models import Message, Role


# =============================================================================
# Schema matching the 15 training features exactly
# =============================================================================

FEATURE_SCHEMA = {
    "type": "object",
    "properties": {
        "Age": {
            "type": "number",
            "description": "Patient's age in years."
        },
        "Sex": {
            "type": "string",
            "enum": ["Mand", "Kvinde"],
            "description": "Patient's sex. Use 'Mand' for male, 'Kvinde' for female."
        },
        "Height_cm": {
            "type": "number",
            "description": "Patient's height in centimeters."
        },
        "Weight_kg": {
            "type": "number",
            "description": "Patient's weight in kilograms."
        },
        "AlcoholConsumption_X": {
            "type": "number",
            "description": "Number of standard alcoholic drinks per week. Use 0 if patient denies drinking. Use -1 if not mentioned or unknown."
        },
        "NumberOfAllergies": {
            "type": "integer",
            "description": "Count of distinct allergies the patient mentioned (medications, latex, etc.). Use 0 if patient denies having allergies."
        },
        "NumberOfDiagnoses": {
            "type": "integer",
            "description": "Count of distinct medical conditions or diagnoses the patient mentioned (e.g. hypertension, diabetes, COPD, sleep apnea, heart disease). Count each separate condition once."
        },
        "SmokingStatus_Simplified": {
            "type": "string",
            "enum": ["Current smoker", "Never smoker", "Former smoker", "Unknown"],
            "description": "Patient's smoking status. 'Current smoker' if they currently smoke or use nicotine. 'Former smoker' if they used to but quit. 'Never smoker' if they never smoked. 'Unknown' if not mentioned."
        },
        "NeuroPsyk_flag": {
            "type": "integer",
            "enum": [0, 1],
            "description": "1 if the patient mentioned any neurological or psychiatric conditions (e.g. epilepsy, depression, anxiety disorder, neuropathy, stroke history). 0 if none mentioned or explicitly denied."
        },
        "Respiratorisk_flag": {
            "type": "integer",
            "enum": [0, 1],
            "description": "1 if the patient mentioned any respiratory conditions (e.g. asthma, COPD, sleep apnea, shortness of breath, breathing problems). 0 if none mentioned or explicitly denied."
        },
        "Kardiovaskulært_flag": {
            "type": "integer",
            "enum": [0, 1],
            "description": "1 if the patient mentioned any cardiovascular conditions (e.g. high blood pressure, heart disease, chest pain, stents, heart surgery, arrhythmia, blood thinners). 0 if none mentioned or explicitly denied."
        },
        "GI/Lever/Nyre_flag": {
            "type": "integer",
            "enum": [0, 1],
            "description": "1 if the patient mentioned any gastrointestinal, liver, or kidney conditions (e.g. acid reflux, heartburn, liver disease, kidney problems, dialysis). 0 if none mentioned or explicitly denied."
        },
        "Endo/Andet_flag": {
            "type": "integer",
            "enum": [0, 1],
            "description": "1 if the patient mentioned any endocrine or other systemic conditions (e.g. diabetes, thyroid disease, autoimmune conditions). 0 if none mentioned or explicitly denied."
        },
        "Bevægeapparat_flag": {
            "type": "integer",
            "enum": [0, 1],
            "description": "1 if the patient mentioned any musculoskeletal conditions (e.g. joint problems, back pain, arthritis, mobility issues, previous joint replacement). 0 if none mentioned or explicitly denied."
        }
    },
    "required": [
        "Age",
        "Sex",
        "Height_cm",
        "Weight_kg",
        "AlcoholConsumption_X",
        "NumberOfAllergies",
        "NumberOfDiagnoses",
        "SmokingStatus_Simplified",
        "NeuroPsyk_flag",
        "Respiratorisk_flag",
        "Kardiovaskulært_flag",
        "GI/Lever/Nyre_flag",
        "Endo/Andet_flag",
        "Bevægeapparat_flag"
    ]
}


# =============================================================================
# Extraction logic
# =============================================================================

FEATURE_EXTRACTOR_SYSTEM_PROMPT = (
    "You are a clinical data extraction assistant. Your task is to extract "
    "structured medical features from a patient conversation transcript. "
    "Extract only what the patient explicitly stated. Do not infer or assume "
    "conditions that were not mentioned. If a value was not discussed, use "
    "the appropriate default: 0 for counts and flags, 'Unknown' for smoking "
    "status, -1 for alcohol consumption."
)


def build_extraction_prompt(transcript: List[Message]) -> str:
    """Convert transcript to a text block for the extraction prompt."""
    lines = []
    for m in transcript:
        if m.role in (Role.PATIENT, Role.ASSISTANT):
            speaker = "Patient" if m.role == Role.PATIENT else "Assistant"
            lines.append(f"{speaker}: {m.content}")
    transcript_text = "\n".join(lines)

    return (
        "Extract the structured medical features from the following "
        "pre-anesthesia consultation transcript.\n\n"
        f"Transcript:\n{transcript_text}\n"
    )