from ToGood2Go.utils.session_store import SESSION_STATE

ANAMNESIS_SLOTS = [
    "patient demographics",
    "previous anesthesia experience",
    "medications",
    "allergies",
    "cardiovascular history",
    "respiratory history",
    "diabetes or endocrine issues",
    "lifestyle factors (smoking, alcohol)",
    "functional capacity (METs)",
]

def update_session(session_id, message):
    if session_id not in SESSION_STATE:
        SESSION_STATE[session_id] = {"messages": [], "filled_slots": []}

    SESSION_STATE[session_id]["messages"].append(message)


def get_next_question(session_id):
    session = SESSION_STATE.get(session_id, {"filled_slots": []})

    for slot in ANAMNESIS_SLOTS:
        if slot not in session["filled_slots"]:
            return f"Please ask about: {slot}", slot

    return "All slots filled.", None
