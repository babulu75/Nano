import requests
from config import LOCAL_MODEL, OLLAMA_URL


def think_local(prompt: str) -> str:

    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": LOCAL_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False
        },
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    return data["message"]["content"]