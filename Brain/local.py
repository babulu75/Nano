import requests
from typing import List, Dict, Optional
from config import LOCAL_MODEL, OLLAMA_URL


def think_local(prompt: str, history: Optional[List[Dict[str, str]]] = None) -> str:
    """
    Send chat messages to Ollama with conversation history.
    Thinking is disabled so thinking-capable models return visible answer text.
    """
    messages = []
    if history:
        messages.extend(history)

    # Append current user prompt
    messages.append({
        "role": "user",
        "content": prompt,
    })

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": LOCAL_MODEL,
                "messages": messages,
                "stream": False,
                "think": False,
            },
            timeout=120,
        )

        response.raise_for_status()
        data = response.json()
        content = data.get("message", {}).get("content", "")
        if not isinstance(content, str) or not content.strip():
            return (
                "[Nano] Ollama returned an empty answer. "
                "Please try again; if it keeps happening, restart Ollama and check the model."
            )
        return content.strip()

    except requests.exceptions.Timeout:
        return (
            f"[Nano] Ollama did not respond within 120 seconds.\n"
            f"  The model '{LOCAL_MODEL}' may still be loading — try again in a moment."
        )

    except requests.exceptions.ConnectionError:
        return (
            "[Nano] Cannot reach Ollama. Is it running?\n"
            f"  Start it with: ollama serve\n"
            f"  Then make sure the model is pulled: ollama pull {LOCAL_MODEL}"
        )

    except requests.exceptions.HTTPError as e:
        return f"[Nano] Ollama returned an error: {e}"

    except Exception as e:
        return f"[Nano] Unexpected error: {e}"


class LocalBrain:

    def generate(self, prompt: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        return think_local(prompt, history=history)
