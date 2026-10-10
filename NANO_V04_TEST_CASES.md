# Nano v0.4 — Google AI Mode Test Cases

## Purpose

Verify Nano's Google AI Mode → copied answer and citations → local Ollama pipeline.

**Environment:** Windows 11, project at `C:\Users\babul\Desktop\Nano`, existing `.venv`, Chrome, and local Ollama.
**Answer model:** `qwen3.5:2b-q4_K_M`
**Classifier:** `qwen2.5:0.5b-instruct` decides whether web information is needed; it does not answer the user or select a model.
**Browser:** Playwright controls Nano's persistent Chrome profile. Web requests open Google AI Mode, submit the query, click its Copy answer control, read the clipboard, and collect visible citation links. No screenshot OCR or paid search API is used.

## Test strategy

1. Unit tests mock Chrome and Ollama for offline behavior.
2. Live Google AI Mode and Ollama tests are opt-in; they open visible Chrome, access the internet, and use the configured local Chrome profile.
3. Never bypass CAPTCHA, sign-in, bot checks, or access-denied controls. Report a clear error if AI Mode is unavailable or its answer cannot be copied.

## Test cases

| ID | Type | Scenario | Expected result |
|---|---|---|---|
| TC-001 | Unit | Google AI Mode question composer is available | Fill it with the user's query and submit with Enter |
| TC-002 | Unit | AI Mode answer Copy control appears | Click it and read the copied answer through the browser clipboard |
| TC-003 | Unit | Google wraps a source URL in a redirect | Unwrap it and preserve the destination citation URL |
| TC-004 | Unit | AI Mode page exposes source cards | Collect unique visible external source links with the copied answer |
| TC-005 | Unit | Composer or Copy control is missing | Return a clear availability or timeout error; do not send stale clipboard contents to Ollama |
| TC-006 | Unit | Copy action returns empty text or the original query | Stop with a clear extraction error |
| TC-007 | Unit | Copied answer and citations fit within evidence limits | Pass both as untrusted browser evidence to Ollama |
| TC-008 | Unit | Ollama does not include a source URL in its answer | Keep its answer and append the collected Google AI Mode source links |
| TC-009 | Unit | Ollama request fails | Return its actionable error and do not save an error as conversation memory |
| TC-010 | Unit | Ordinary conversation does not need web access | Use local Ollama without opening Chrome |
| TC-011 | Unit | Chrome task succeeds or raises an error | Close the task tab; keep the shared browser available until application shutdown |
| TC-012 | Integration (opt-in) | Google AI Mode access and clipboard copy | Receive non-empty copied answer text and at least one citation link |
| TC-013 | Integration (opt-in) | Google AI Mode answer is passed to local Ollama | Return a non-empty answer without an operational error |

## Acceptance criteria

- All offline unit tests pass.
- The opt-in integration test confirms a live Google AI Mode answer and citations reach Ollama.
- Web answers use the copied AI Mode text and preserve its sources; no screenshot OCR step is used.
- If Google requests sign-in, blocks access, or changes the page so Nano cannot find the composer or Copy control, Nano reports that state without fabricating an answer.
- Browser and Ollama errors are reported clearly and do not crash the CLI.
- The browser profile remains separate from the user's everyday Chrome profile.

## Run

Run offline tests:

```bat
.venv\Scripts\python.exe -m pytest -q
```

Run the complete live Google AI Mode → Ollama check (opens visible Chrome and contacts the local Ollama service):

```bat
set NANO_RUN_GOOGLE_OLLAMA_INTEGRATION=1
.venv\Scripts\python.exe -m pytest -q tests\test_nano_v04.py::test_google_ai_mode_answer_flows_into_ollama
```

Nano uses its persistent Chrome profile under `%LOCALAPPDATA%\Nano\ChromeProfile`. Sign into Google in that profile if AI Mode requires it. Do not point Nano at your everyday Chrome profile.
