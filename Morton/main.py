import json
from app.domain.models import Conversation
from app.adapters.question_flow_json import JsonQuestionFlow
from app.adapters.store_memory import MemoryTranscriptStore
from app.adapters.ollama_summarizer import OllamaSummarizer
from app.adapters.ollama_llm_client import OllamaLLMClient
from app.application.orchestrator import ConversationOrchestrator


MODEL = "llama3.1"
BASE_URL = "http://localhost:11434"


def main():
    store = MemoryTranscriptStore()
    qflow = JsonQuestionFlow("data/questions.json")
    summarizer = OllamaSummarizer(model=MODEL, base_url=BASE_URL)
    llm_client = OllamaLLMClient(model=MODEL, base_url=BASE_URL)

    orch = ConversationOrchestrator(
        question_flow=qflow,
        transcript_store=store,
        summarizer=summarizer,
        summary_schema_path="data/summary_schema.json",
        llm_client=llm_client,
    )

    conv = Conversation(conversation_id="test-1")
    res = orch.start(conv)
    print(f"\nBOT: {res.bot_text}\n")

    while True:
        user = input("YOU: ").strip()
        if not user:
            continue
        if user.lower() in {"quit", "exit"}:
            break

        res = orch.handle_user_message(conv, user)

        if res is None:
            raise RuntimeError("handle_user_message returned None.")

        print(f"\nBOT: {res.bot_text}\n")

        if res.done:
            print("\nGenerating structured summary...\n")
            summary = orch.finalize(conv)
            print(json.dumps(summary, indent=2, ensure_ascii=False))

            with open("summary_output.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            print("\nSummary saved to summary_output.json")
            break


if __name__ == "__main__":
    main()