import json

from app.domain.models import Conversation, Mode
from app.adapters.question_flow_json import JsonQuestionFlow
from app.adapters.store_memory import MemoryTranscriptStore
from app.adapters.ollama_chatbot import OllamaChatbot
# from app.adapters.ollama_summarizer import OllamaSummarizer
from app.adapters.improved_ollama_summarizer import OllamaSummarizer
from app.application.orchestrator import ConversationOrchestrator

model = "llama3.1" # "jobautomation/OpenEuroLLM-Danish:latest"

def main():
    store = MemoryTranscriptStore()
    qflow = JsonQuestionFlow("data/questions.json")

    # NEW: Wire in Ollama chatbot
    chatbot = OllamaChatbot(model=model, base_url="http://localhost:11434")
    summarizer = OllamaSummarizer(model=model, base_url="http://localhost:11434")

    orch = ConversationOrchestrator(
        qflow,
        store,
        chatbot=chatbot,
        summarizer=summarizer,
        template_path="data/summary_schema.json",
    )

    conv = Conversation(conversation_id="test-1")
    res = orch.start(conv)
    print("\nTip: Prefix with '?' to ask the chatbot while staying in the flow.")
    print("Example: ?What does fasting mean before surgery?\n")

    print("BOT:", res.bot_text)

    while True:
        user = input("YOU: ").strip()
        if user.lower() in {"quit", "exit"}:
            break

        if user.startswith("?"):
            res = orch.handle_user_message(conv, user[1:].strip(), mode=Mode.CHAT)
        else:
            res = orch.handle_user_message(conv, user, mode=Mode.ANSWER)

        if res is None:
            raise RuntimeError("handle_user_message returned None (missing return).")

        print("BOT:", res.bot_text)

        if res.done:
            print("\nGenerating structured summary...\n")
            summary = orch.finalize(conv)
            print(json.dumps(summary, indent=2, ensure_ascii=False))

            # Optionally save
            with open("summary_output (1).json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            break

    # print("\n--- Transcript ---")
    # for m in store.get(conv.conversation_id):
    #     print(f"{m.ts} [{m.role}] {m.content} {m.meta}")


if __name__ == "__main__":
    main()