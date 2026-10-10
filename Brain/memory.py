import os
import json
import time
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict, field
from config import debug_log, error_log


@dataclass
class Message:
    role: str  # "user", "assistant", or "system"
    content: str
    model: Optional[str] = None  # Stored provider label; current Nano answers use Ollama.
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

    # Feature: atomic conversation turn storage — Nano v0.4 — Purpose: persist one completed user/assistant exchange with a single memory write.
    def add_exchange(self, user_text: str, assistant_text: str, model: Optional[str] = None) -> None:
        self._messages.extend((
            Message(role="user", content=user_text),
            Message(role="assistant", content=assistant_text, model=model),
        ))
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
        # Feature: memory persistence diagnostics — Nano v0.4 — Purpose: log routine persistence only in debug mode while surfacing essential save failures.
        if not self.persist_path:
            return
        try:
            debug_log(f"Saving conversation memory; turns={len(self._messages)}; path={self.persist_path}.")
            os.makedirs(os.path.dirname(os.path.abspath(self.persist_path)), exist_ok=True)
            with open(self.persist_path, "w", encoding="utf-8") as f:
                json.dump([asdict(m) for m in self._messages], f, indent=2, ensure_ascii=False)
        except Exception as e:
            error_log(f"Failed to save conversation memory: {e}")

    def load(self) -> None:
        """Load conversation from JSON file."""
        # Feature: memory restoration diagnostics — Nano v0.4 — Purpose: trace preserved memory loading without noisy default terminal output.
        if not self.persist_path or not os.path.exists(self.persist_path):
            return
        try:
            debug_log(f"Loading conversation memory; path={self.persist_path}.")
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
            debug_log(f"Conversation memory loaded; turns={len(self._messages)}.")
        except Exception as e:
            error_log(f"Failed to load conversation memory: {e}")
