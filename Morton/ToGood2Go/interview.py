import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
import ollama

def chat(transcript_path: Path, session_id:str):
    model = 'jobautomation/OpenEuroLLM-Danish:latest' # "llama3.1"
    history = []

    while True:
        prompt = input("DIG: ").strip()
        if prompt.lower() in ("exit", "quit", "/quit"):
            print("Farvel")
            break

        user_msg = {"role": "user", "content": prompt}
        history.append(user_msg)

        append_jsonl(transcript_path, {
            "ts": now_iso(),
            "type": "chat",
            "sessionId": session_id,
            "questionId": "chat",
            "text": user_msg
        })

        print("Bot: ", end="", flush=True)
        bot_message_content = ""

        response = ollama.chat(model=model, messages=history, stream=True)

        for chunk in response:
            # Handle both styles: object chunks vs dict chunks
            if hasattr(chunk, "message"):
                token = chunk.message.content or ""
            else:
                token = chunk.get("message", {}).get("content", "") or chunk.get("response", "")

            bot_message_content += token
            print(token, end="", flush=True)

        print()
        history.append({"role": "assistant", "content": bot_message_content})

# Get the current timestamp in UTC
def now_iso() -> int:
    return int(datetime.now(timezone.utc).timestamp())

# Append info to the json log
def append_jsonl(path: Path, obj: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# Load questions from Json and return a list of dictionary objects
def load_questions(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["questions"]

def main():

    # Path to json with questions
    questions_path = Path("../data/questions.json")

    # Directory where json log and summary are placed
    out_dir = Path("runs")
    out_dir.mkdir(exist_ok=True)

    # Compute a unique session ID
    session_id = str(uuid.uuid4())
    transcript_path = out_dir / f"{session_id}.jsonl"
    summary_path = out_dir / f"{session_id}.json"

    questions = load_questions(questions_path)

    # Start-event
    append_jsonl(transcript_path, {
        "ts": now_iso(),
        "type": "session_started",
        "sessionId": session_id,
        "meta": {"questionsFile": str(questions_path)}
    })

    # Dictionary to hold the answers
    answers = {}

    print("\n--- Interview start ---")
    print(f"Session: {session_id}")
    print("Tip: Skriv '/quit' for at stoppe.\n")

    for q in questions:
        qid = q["id"]
        qtext = q["text"]

        # Log at spørgsmålet blev vist
        append_jsonl(transcript_path, {
            "ts": now_iso(),
            "type": "standard_question_shown",
            "sessionId": session_id,
            "questionId": qid,
            "text": qtext
        })

        print(f"SPØRGSMÅL ({qid}): {qtext}")
        answer = input("SVAR: ").strip()

        # command handling
        while answer.lower() in ("/chat",):
            # go to chat, then come back to same question
            append_jsonl(transcript_path, {
                "ts": now_iso(),
                "type": "command_used",
                "sessionId": session_id,
                "command": "/chat",
                "context": {"questionId": qid}
            })
            chat(transcript_path, session_id)
            print(f"SPØRGSMÅL ({qid}): {qtext}")
            print("Svar eller skriv /chat, /next, /quit")
            answer = input("SVAR: ").strip()

        if answer.lower() == "/quit":
            append_jsonl(transcript_path, {
                "ts": now_iso(),
                "type": "session_aborted",
                "sessionId": session_id,
                "reason": "user_quit"
            })
            print("\nAfsluttet.")
            return

        answers[qid] = answer

        # Log svaret
        append_jsonl(transcript_path, {
            "ts": now_iso(),
            "type": "standard_answer_given",
            "sessionId": session_id,
            "questionId": qid,
            "text": answer
        })

        print()  # luft

    # Slut-event
    append_jsonl(transcript_path, {
        "ts": now_iso(),
        "type": "session_finished",
        "sessionId": session_id
    })

    # Gem en samlet “pakke” (nem at bruge til opsummering senere)
    output = {
        "sessionId": session_id,
        "createdAt": now_iso(),
        "questions": questions,
        "answers": answers,
        "transcriptJsonl": str(transcript_path)
    }
    summary_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print("--- Interview done ---")
    print(f"Transcript: {transcript_path}")
    print(f"Output:      {summary_path}")

if __name__ == "__main__":
    main()
