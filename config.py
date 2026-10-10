import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Feature: centralized backend logging — Nano v0.4 — Purpose: one switch gates all diagnostic output.
DEBUG_LOGS = True


def debug_log(message: str) -> None:
    """Print backend diagnostics only when the single project switch is enabled."""
    if DEBUG_LOGS:
        print(f"[Nano debug] {message}")


def error_log(message: str) -> None:
    """Keep essential operational errors visible even when debug logs are disabled."""
    print(f"[Nano] {message}", file=sys.stderr)

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
