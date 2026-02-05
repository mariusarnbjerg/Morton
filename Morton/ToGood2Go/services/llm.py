import requests

OLLAMA_URL = "http://localhost:11434/api/chat"

def call_ollama(system_prompt: str, user_message: str, model: str = "llama3.1"):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    }

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()

    data = response.json()
    return data["message"]["content"]
