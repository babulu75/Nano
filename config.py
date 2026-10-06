import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

LOCAL_MODEL = os.getenv(
    "LOCAL_MODEL",
    "qwen3.5:2b-q4_K_M"
)

ONLINE_MODEL = os.getenv(
    "ONLINE_MODEL",
    "gemini-3.5-flash"
)

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434"
)