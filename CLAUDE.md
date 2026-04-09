# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Morton (branded as **AnæstesiCare**) is an AI-assisted pre-anesthesia interview system. A Python/FastAPI backend drives a structured clinical questionnaire via a local LLM (Ollama), and a React/TypeScript frontend provides both a patient-facing chatbot and a doctor-facing patient management UI.

## Prerequisites

Ollama must be running locally with the custom `morton` model:

```bash
ollama pull qwen3:8b
ollama create morton -f Morton/Modelfile   # adjust path as needed
```

## Running the System

**Backend** (from the `Morton/` directory):
```bash
pip install -r requirements.txt
python -m app.api.server          # API at http://localhost:8000, docs at /docs
# Or with explicit uvicorn:
uvicorn app.api.server:app --host 0.0.0.0 --port 8000 --reload
```

**CLI mode** (no frontend, interactive terminal):
```bash
cd Morton && python main.py
```

**Frontend** (from the `WebPage/` directory):
```bash
npm install
npm run dev     # Dev server at http://localhost:5173
npm run build   # Production bundle
```

The frontend calls the backend at `http://127.0.0.1:8000` by default (override with `VITE_API_BASE` env var).

## Architecture

The backend follows **hexagonal (ports & adapters) architecture**:

```
FastAPI (app/api/)
  └── ConversationOrchestrator (app/application/orchestrator.py)   ← owns all state
        ├── IQuestionFlow  → QuestionFlowJSON (loads data/questions.json)
        ├── ITranscriptStore → StoreMemory (in-memory per-conversation)
        ├── ISummarizer    → OllamaSummarizer
        └── ILLMClient     → OllamaLLMClient (HTTP to localhost:11434)
```

Key principle: **the orchestrator, not the LLM, owns the conversation state machine**. The LLM only handles natural language (extract answers, generate replies, check confirmations). All logic about what question comes next lives in Python.

### Conversation State Machine

`ConversationState` in [Morton/app/domain/models.py](Morton/app/domain/models.py):
- `IN_PROGRESS` → `AWAITING_CONFIRMATION` → `IN_PROGRESS` (cycle per question if `confirmation_required`)
- `IN_PROGRESS` → `FREE_CHAT` (when patient goes off-topic)
- `IN_PROGRESS` → `DONE` (all required questions complete)

### Data files (Morton/data/)

- `questions.json` — full questionnaire: question text, type (`free_text`/`yesno`/`number`/`choice`), `completion_criteria`, optional `follow_ups` with `trigger` conditions
- `summary_schema.json` — JSON Schema for the structured clinical output (ASA classification, allergies, etc.)
- `summary_template.json` — template passed to the summarizer

### Frontend structure (WebPage/src/)

- `app/App.tsx` — top-level role switcher (doctor vs. patient mode)
- `api/client.ts` — all HTTP calls to the backend; normalizes response key variations
- `app/components/PatientChatbot.tsx` — patient questionnaire UI
- `app/components/PatientSearch.tsx`, `CalendarView.tsx` — doctor-mode views
- `app/components/ui/` — Radix UI primitive wrappers (~30 components)

### API endpoints (all under `/api/v1/conversations`)

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/start` | Create a new conversation |
| POST | `/{id}/chat` | Send a message, get LLM reply |
| POST | `/{id}/answer` | Submit a structured answer |
| GET | `/{id}/state` | Get current conversation state |
| GET | `/{id}/summary` | Retrieve the final clinical summary |

## Environment Variables

Create a `.env` file in `Morton/` if needed:

```
OLLAMA_MODEL=morton
OLLAMA_BASE_URL=http://localhost:11434
CORS_ORIGINS=http://localhost:5173,http://localhost:3000
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=true
```

## Extending the Questionnaire

Add questions to `Morton/data/questions.json` — no code changes needed. Each question supports `follow_ups` (triggered by LLM evaluation of patient responses) and multi-layer `completion_criteria`.

## Adding a New LLM Adapter

Implement `ILLMClient` from [Morton/app/interfaces/llm_client.py](Morton/app/interfaces/llm_client.py) and wire it up in [Morton/app/api/dependencies.py](Morton/app/api/dependencies.py).
