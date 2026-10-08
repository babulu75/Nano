from Brain.local import LocalBrain
from Brain.memory import ConversationMemory
from config import MEMORY_FILE, MAX_MEMORY_TURNS


class NanoBridge:

    def __init__(self):

        self.brain = LocalBrain()
        self.memory = ConversationMemory(
            max_turns=MAX_MEMORY_TURNS,
            persist_path=MEMORY_FILE,
        )

    def process(self, text: str) -> str:

        history = self.memory.get_ollama_history()

        response = self.brain.generate(text, history=history)

        # Keep transport/model errors out of the saved conversation context.
        if response.startswith("[Nano]"):
            return response

        self.memory.add_user_message(text)
        self.memory.add_assistant_message(response, model="ollama")

        return response

    def clear_memory(self):

        self.memory.clear()

    def shutdown(self):
        """No-op — kept for interface stability."""
        pass
