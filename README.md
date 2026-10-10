# Nano v0.4

Nano is a local-first assistant. Ollama is its answering brain. Browser automation is a reusable tool that can gather evidence when a request needs current or external information.

## Request flow

```text
main.py (CLI)
  -> assistant.py (route and coordinate)
     -> Brain/requirement_classifier.py
        -> local request: Brain/local.py -> Ollama
        -> web request: tools/browser/research_pipeline.py
              -> shared tools/browser/chrome_manager.py
              -> Google AI Mode -> copy answer + collect citations
           -> Brain/local.py -> Ollama summarizes answer with citations
     -> Brain/memory.py (save completed conversation)
```

`NanoAssistant` owns one `ChromeManager` for the lifetime of the CLI session. A tool opens its own tab in that shared Chrome context and closes the tab when its task ends. The shared browser stays available for the next tool request and closes when Nano exits. Future browser tools, such as mail, should receive this same manager instead of starting another Chrome instance.

## Project layout

```text
Nano/
├── main.py                 # Terminal interface and long-lived async session
├── assistant.py            # Request routing, tool coordination, answer validation
├── config.py               # Local model, memory, and central DEBUG_LOGS switch
├── Brain/                  # Requirement classifier, Ollama answer model, conversation memory
├── tools/
│   └── browser/            # Shared Chrome lifecycle, Google AI Mode, evidence validation
├── models/                 # Data passed between tools and the assistant
├── scripts/                # Manual developer smoke tests
└── tests/                  # Automated unit and opt-in integration tests
```

Conversation memory, `.env`, and local Chrome profile data remain outside the source-package folders.

## Run

```bat
.venv\Scripts\python.exe main.py
```

Set `DEBUG_LOGS = True` or `False` in `config.py` to show or silence backend diagnostics. Install dependencies with:

```bat
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run automated tests with:

```bat
.venv\Scripts\python.exe -m pytest -q
```

The Chrome and Ollama integration tests are opt-in; see `NANO_V04_TEST_CASES.md`.
