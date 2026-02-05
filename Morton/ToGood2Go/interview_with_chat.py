import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import ollama
import requests


# ---- Config ----
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "jobautomation/OpenEuroLLM-Danish"  # ændr til din model, fx "mistral", "llama3.2", osv.


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, obj: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_questions(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["questions"]


def ollama_chat(messages):
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False
    }

    r = requests.post(OLLAMA_URL, json=payload, timeout=120)

    if not r.ok:
        # Print præcis fejl fra Ollama (typisk JSON med "error")
        raise RuntimeError(f"Ollama HTTP {r.status_code}: {r.text}")

    data = r.json()
    return data.message.content



def build_chat_system_prompt() -> str:
    # Keep it simple. You can tighten this later (medical disclaimers, tone, etc.)
    return (
        "Du er en hjælpsom pre-anæstesi chatbot. Svar kort og klart på dansk. "
        "Hvis du er usikker, så sig det. Giv ikke farlige råd. "
        "Opfordr til at kontakte sundhedspersonale ved alvorlige symptomer."
    )


def main():
    questions_path = Path("../data/questions.json")
    out_dir = Path("runs")
    out_dir.mkdir(exist_ok=True)

    session_id = str(uuid.uuid4())
    transcript_path = out_dir / f"{session_id}.jsonl"
    output_path = out_dir / f"{session_id}.json"

    questions = load_questions(questions_path)

    # In-memory structures (we also log everything to JSONL)
    answers: Dict[str, str] = {}
    chat_messages: List[Dict[str, str]] = [{"role": "system", "content": build_chat_system_prompt()}]

    append_jsonl(transcript_path, {
        "ts": now_iso(),
        "type": "session_started",
        "sessionId": session_id,
        "meta": {
            "questionsFile": str(questions_path),
            "ollamaModel": OLLAMA_MODEL,
            "ollamaUrl": OLLAMA_URL
        }
    })

    print("\n--- Interview start ---")
    print(f"Session: {session_id}")
    print("Kommandoer: /chat, /next, /quit\n")

    def do_chat():
        nonlocal chat_messages
        print("\n--- Chat med chatbot ---")
        print("Skriv dit spørgsmål. (Tom linje for at gå tilbage)\n")

        while True:
            user_q = input("DIG: ").strip()
            if user_q == "":
                print("--- Tilbage til spørgeflow ---\n")
                return
            if user_q.lower() in ("/quit",):
                raise SystemExit

            # Log patient spørgsmål
            append_jsonl(transcript_path, {
                "ts": now_iso(),
                "type": "patient_chat_question",
                "sessionId": session_id,
                "text": user_q
            })

            chat_messages.append({"role": "user", "content": user_q})

            try:
                bot_a = ollama_chat(chat_messages)
            except Exception as e:
                bot_a = f"[Fejl ved kald til Ollama: {e}]"

            # Log bot svar
            append_jsonl(transcript_path, {
                "ts": now_iso(),
                "type": "bot_chat_answer",
                "sessionId": session_id,
                "text": bot_a
            })

            chat_messages.append({"role": "assistant", "content": bot_a})

            print(f"\nBOT: {bot_a}\n")

    # Main question loop
    for q in questions:
        qid = q["id"]
        qtext = q["text"]

        append_jsonl(transcript_path, {
            "ts": now_iso(),
            "type": "standard_question_shown",
            "sessionId": session_id,
            "questionId": qid,
            "text": qtext
        })

        print(f"SPØRGSMÅL ({qid}): {qtext}")
        print("Svar eller skriv /chat, /next, /quit")
        user_input = input("SVAR: ").strip()

        # command handling
        while user_input.lower() in ("/chat",):
            # go to chat, then come back to same question
            append_jsonl(transcript_path, {
                "ts": now_iso(),
                "type": "command_used",
                "sessionId": session_id,
                "command": "/chat",
                "context": {"questionId": qid}
            })
            do_chat()
            print(f"SPØRGSMÅL ({qid}): {qtext}")
            print("Svar eller skriv /chat, /next, /quit")
            user_input = input("SVAR: ").strip()

        if user_input.lower() == "/quit":
            append_jsonl(transcript_path, {
                "ts": now_iso(),
                "type": "session_aborted",
                "sessionId": session_id,
                "reason": "user_quit"
            })
            print("\nAfsluttet.")
            return

        if user_input.lower() == "/next":
            append_jsonl(transcript_path, {
                "ts": now_iso(),
                "type": "standard_answer_skipped",
                "sessionId": session_id,
                "questionId": qid,
                "text": ""
            })
            print("(Sprunget over)\n")
            continue

        # normal answer
        answers[qid] = user_input
        append_jsonl(transcript_path, {
            "ts": now_iso(),
            "type": "standard_answer_given",
            "sessionId": session_id,
            "questionId": qid,
            "text": user_input
        })
        print()

    # After standard questions: allow final chat
    print("--- Alle standardspørgsmål er besvaret ---")
    print("Vil du stille flere spørgsmål til chatbotten? Skriv /chat eller tryk Enter for at afslutte.")
    final = input("> ").strip()
    if final.lower() == "/chat":
        append_jsonl(transcript_path, {
            "ts": now_iso(),
            "type": "command_used",
            "sessionId": session_id,
            "command": "/chat",
            "context": {"phase": "after_questions"}
        })
        try:
            do_chat()
        except SystemExit:
            append_jsonl(transcript_path, {
                "ts": now_iso(),
                "type": "session_aborted",
                "sessionId": session_id,
                "reason": "user_quit_in_chat"
            })
            print("\nAfsluttet.")
            return

    append_jsonl(transcript_path, {
        "ts": now_iso(),
        "type": "session_finished",
        "sessionId": session_id
    })

    # Save a combined output snapshot
    output = {
        "sessionId": session_id,
        "createdAt": now_iso(),
        "questions": questions,
        "answers": answers,
        "ollama": {"model": OLLAMA_MODEL, "url": OLLAMA_URL},
        "transcriptJsonl": str(transcript_path)
    }
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n--- Færdig ---")
    print(f"Transcript: {transcript_path}")
    print(f"Output:     {output_path}")


if __name__ == "__main__":
    main()
