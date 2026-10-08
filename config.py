import os
from dotenv import load_dotenv

load_dotenv()

LOCAL_MODEL = os.getenv(
    "LOCAL_MODEL",
    "qwen3.5:2b-q4_K_M"
)

CLASSIFIER_MODEL = "qwen2.5:0.5b-instruct"

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434"
)

MEMORY_FILE = os.getenv("MEMORY_FILE", "conversation_memory.json")
MAX_MEMORY_TURNS = int(os.getenv("MAX_MEMORY_TURNS", "20"))
