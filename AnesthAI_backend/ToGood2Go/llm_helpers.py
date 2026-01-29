# llm_helpers.py

import json
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.1"  # or your favourite "thinking" model


def call_ollama(messages):
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "messages": messages,
            "stream": False,
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    # Ollama's /api/chat returns {"message": {"content": "..."} , ...}
    return data["message"]["content"]


def extract_slot_values(user_text: str, current_slots: dict) -> dict:
    """
    Ask the LLM to update the structured anamnesis slots
    based on the patient's latest answer AND what we already know.
    """

    system_msg = {
        "role": "system",
        "content": (
            "You are an assistant that extracts structured medical anamnesis "
            "information from free text.\n"
            "You must ONLY output valid JSON.\n"
            "Keys must be: "
            + ", ".join(current_slots.keys())
            + ".\n"
            "If you don't find information for a key, set it to null."
        ),
    }

    # Give the model what we already know, plus the new answer
    user_msg = {
        "role": "user",
        "content": (
            "Current known information (JSON):\n"
            f"{json.dumps(current_slots, ensure_ascii=False)}\n\n"
            "New patient answer:\n"
            f"{user_text}\n\n"
            "Update the JSON with any new or more precise information you can extract.\n"
            "Only respond with JSON, no explanation."
        ),
    }

    raw = call_ollama([system_msg, user_msg])

    # be safe: try to parse JSON, handle weird outputs
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # fallback: no updates
        print("WARNING: LLM did not return valid JSON:", raw)
        return {}

    # ensure only known keys
    return {k: data.get(k) for k in current_slots.keys()}

def summarize_anamnesis(slots: dict) -> str:
    system_msg = {
        "role": "system",
        "content": (
            "You are an anesthesiologist preparing a preoperative summary. "
            "Summarize the patient's medical history in clear, clinical language."
        ),
    }

    user_msg = {
        "role": "user",
        "content": (
            "Here is the structured anamnesis information as JSON:\n"
            f"{json.dumps(slots, ensure_ascii=False, indent=2)}\n\n"
            "Please write a concise summary suitable for an anesthetic pre-assessment note. "
            "Include: chief complaint, relevant comorbidities, past surgeries, medications, and allergies."
        ),
    }

    return call_ollama([system_msg, user_msg])
