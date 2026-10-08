import os
import json
import time
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict, field


@dataclass
class Message:
    role: str  # "user", "assistant", or "system"
    content: str
    model: Optional[str] = None  # e.g., "gemini", "ollama"
    timestamp: float = field(default_factory=time.time)


class ConversationMemory:
    """
    Conversation memory layer for the local Ollama brain.

    Features:
    - Maintains complete chronological turns.
    - Sliding window (max_turns) to keep context within model limits.
    - Formats history for Ollama (/api/chat).
    - Optional JSON persistence to preserve context across restarts.
    """

    def __init__(
        self,
        max_turns: int = 20,
        persist_path: Optional[str] = None,
    ):
        self.max_turns = max_turns
        self.persist_path = persist_path
        self._messages: List[Message] = []

        if self.persist_path and os.path.exists(self.persist_path):
            self.load()

    def add_user_message(self, content: str) -> None:
        self._messages.append(
            Message(role="user", content=content)
        )
        self._trim()
        self._auto_save()

    def add_assistant_message(self, content: str, model: Optional[str] = None) -> None:
        self._messages.append(
            Message(role="assistant", content=content, model=model)
        )
        self._trim()
        self._auto_save()

    def add_system_message(self, content: str) -> None:
        self._messages.append(
            Message(role="system", content=content)
        )
        self._trim()
        self._auto_save()

    def get_messages(self) -> List[Dict]:
        """Return raw list of messages as dicts."""
        return [asdict(m) for m in self._messages]

    def get_ollama_history(self) -> List[Dict[str, str]]:
        """
        Format history for Ollama /api/chat:
        [{"role": "user" | "assistant" | "system", "content": "..."}]
        """
        history = []
        for message in self._messages:
            if message.role == "assistant" and not message.content.strip():
                # A failed/blank assistant turn should not poison future prompts.
                if history and history[-1]["role"] == "user":
                    history.pop()
                continue
            history.append({"role": message.role, "content": message.content})
        return history


    def _trim(self) -> None:
        """Keep the conversation history within the max_turns window."""
        if len(self._messages) > self.max_turns * 2:
            self._messages = self._messages[-self.max_turns * 2:]

    def clear(self) -> None:
        """Clear all stored messages."""
        self._messages.clear()
        self._auto_save()

    def count(self) -> int:
        """Return total number of messages in memory."""
        return len(self._messages)

    def _auto_save(self) -> None:
        if self.persist_path:
            self.save()

    def save(self) -> None:
        """Persist conversation to JSON file."""
        if not self.persist_path:
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.persist_path)), exist_ok=True)
            with open(self.persist_path, "w", encoding="utf-8") as f:
                json.dump([asdict(m) for m in self._messages], f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Memory] Failed to save memory: {e}")

    def load(self) -> None:
        """Load conversation from JSON file."""
        if not self.persist_path or not os.path.exists(self.persist_path):
            return
        try:
            with open(self.persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._messages = [
                    Message(
                        role=item["role"],
                        content=item["content"],
                        model=item.get("model"),
                        timestamp=item.get("timestamp", time.time())
                    )
                    for item in data
                ]
        except Exception as e:
            print(f"[Memory] Failed to load memory: {e}")
