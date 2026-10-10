import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _default_chrome_path() -> str:
    candidates = [
        Path(os.environ.get("PROGRAMFILES", r"C:\Program Files"))
        / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"))
        / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData/Local")))
        / "Google/Chrome/Application/chrome.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return ""


CHROME_EXECUTABLE = os.getenv("NANO_CHROME_PATH", "").strip() or _default_chrome_path()
CHROME_USER_DATA_DIR = os.getenv("NANO_CHROME_PROFILE", "").strip() or str(
    Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData/Local"))) / "Nano/ChromeProfile"
)
BROWSER_TIMEOUT_MS = int(os.getenv("NANO_BROWSER_TIMEOUT_MS", "20000"))
MAX_SEARCH_RESULTS = int(os.getenv("NANO_MAX_SEARCH_RESULTS", "5"))
MAX_PAGES = int(os.getenv("NANO_MAX_PAGES", "3"))
MAX_EVIDENCE_CHARS = int(os.getenv("NANO_MAX_EVIDENCE_CHARS", "10000"))
MAX_TOTAL_EVIDENCE_CHARS = int(os.getenv("NANO_MAX_TOTAL_EVIDENCE_CHARS", "24000"))
BROWSER_RETRIES = int(os.getenv("NANO_BROWSER_RETRIES", "1"))
SEARCH_ENGINES = (
    ("Google", "https://www.google.com/search?q={query}"),
    ("Bing", "https://www.bing.com/search?q={query}"),
    ("DuckDuckGo", "https://html.duckduckgo.com/html/?q={query}"),
)
