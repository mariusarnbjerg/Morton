# # https://www.youtube.com/watch?v=cy6EAp4iNN4
# from fastapi import FastAPI
# import ollama
#
# app = FastAPI()
#
# # define an endpoint that we can access
# @app.post("/generate")
# def generate(prompt: str):
#     response = ollama.chat(model="llama3.1", messages=[{"role":"user", "content": prompt}])
#     return {"response": response["message"]["content"]}

# main.py

from fastapi import FastAPI
from pydantic import BaseModel
import requests

app = FastAPI()

# ------------------------------------------------------------
# 1. Data models
# ------------------------------------------------------------

class ChatRequest(BaseModel):
    conversation_id: str
    message: str


class ConversationState:
    """Holder styr på messages + hvilke sektioner der er dækket."""
    def __init__(self):
        self.messages = []
        self.completed = set()   # fx {"basic_info", "medication"}
        self.answers = {}        # parsed information from user answers


# ------------------------------------------------------------
# 2. "Database" i memory (skift dette til rigtig DB senere)
# ------------------------------------------------------------

conversations = {}  # {conversation_id: ConversationState()}


# ------------------------------------------------------------
# 3. Anamnese-definition
# ------------------------------------------------------------

ANAMNESE_FLOW = [
    ("basic_info", "Basale oplysninger (vægt, højde)"),
    ("medical_history", "Tidligere sygdomme og operationer"),
    ("medication", "Aktuel medicin"),
    ("allergies", "Allergier og reaktioner på bedøvelse"),
    ("lifestyle", "Rygning, alkohol, fysisk form"),
]


def get_next_topic(state: ConversationState):
    """Returnér ID + label for den næste ikke-færdige sektion."""
    for section_id, label in ANAMNESE_FLOW:
        if section_id not in state.completed:
            return section_id, label
    return None, None  # færdig med hele anamnesen


# ------------------------------------------------------------
# 4. Helper: System-prompt bygning
# ------------------------------------------------------------

def build_system_prompt(next_id, next_label, answers):
    if next_id is None:
        return (
            "Du er en erfaren anæstesilæge. Du har nu indsamlet alle nødvendige "
            "oplysninger for en præoperativ anæstesi-anamnese. "
            "Lav nu en venlig afrunding og spørg om patienten har yderligere spørgsmål."
        )

    return f"""
Du er anæstesilæge og laver en præoperativ anæstesivurdering.
Samtalen skal være rolig, empatisk og i helt naturligt sprog.

Du arbejder struktureret og skal nu dække sektionen:
**{next_label}**

REGLER:
- Stil kun 1–2 relevante spørgsmål ad gangen.
- Reager kort på patientens seneste svar.
- Undgå at nævne at du følger en tjekliste.

Tidligere indsamlede informationer: {answers}
"""


# ------------------------------------------------------------
# 5. Helper: (Optional) Analyse af patientens svar
# ------------------------------------------------------------
# Simpel placeholder — du kan udvide med LLM-baseret informationsudtræk.

def update_state_from_user_input(state: ConversationState, text: str, next_topic_id: str):
    """Simpel logik: Hvis brugeren siger 'jeg tager ingen medicin' → marker sektionen færdig."""
    low = text.lower()

    if next_topic_id == "basic_info":
        if "cm" in low or "kg" in low or "højde" in low or "vægt" in low:
            state.answers["basic_info"] = text
            state.completed.add("basic_info")

    elif next_topic_id == "medical_history":
        if "ingen" in low or "har aldrig" in low or "operation" in low:
            state.answers["medical_history"] = text
            # du kan vælge at kræve mere detaljeret info

    elif next_topic_id == "medication":
        if "ingen medicin" in low or "tager ikke medicin" in low:
            state.answers["medication"] = "Ingen medicin"
            state.completed.add("medication")

    elif next_topic_id == "allergies":
        if "ingen allergi" in low or "ingen reaktion" in low:
            state.answers["allergies"] = text
            state.completed.add("allergies")

    elif next_topic_id == "lifestyle":
        if "ikke ryger" in low or "løber" in low or "motion" in low:
            state.answers["lifestyle"] = text
            # her kan du vælge at kræve mere info

    return state


# ------------------------------------------------------------
# 6. Ollama call
# ------------------------------------------------------------

def ask_ollama(messages):
    url = "http://localhost:11434/v1/chat/completions"
    payload = {
        "model": "gpt-oss:20b",   # Skift til din model i Ollama
        "messages": messages,
        "stream": False
    }
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# ------------------------------------------------------------
# 7. FastAPI endpoint
# ------------------------------------------------------------

@app.post("/chat")
def chat(req: ChatRequest):

    # Opret ny samtale hvis nødvendig
    if req.conversation_id not in conversations:
        conversations[req.conversation_id] = ConversationState()

    conv = conversations[req.conversation_id]

    # Tilføj user-meddelelse
    conv.messages.append({"role": "user", "content": req.message})

    # Find næste sektion
    next_topic_id, next_topic_label = get_next_topic(conv)

    # Opdater state baseret på brugerinput
    conv = update_state_from_user_input(conv, req.message, next_topic_id)

    # (Efter opdatering kan sektionen være "færdig")
    next_topic_id, next_topic_label = get_next_topic(conv)

    # Byg systemprompt
    system_prompt = build_system_prompt(next_topic_id, next_topic_label, conv.answers)

    # Forbered Ollama-kald
    ollama_messages = [
        {"role": "system", "content": system_prompt},
        *conv.messages
    ]

    # Få svar fra modellen
    reply = ask_ollama(ollama_messages)

    # Gem assistent-svar i historikken
    conv.messages.append({"role": "assistant", "content": reply})

    return {
        "response": reply,
        "next_topic": next_topic_id,
        "completed": list(conv.completed),
        "answers": conv.answers
    }
