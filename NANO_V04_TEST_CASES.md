# Nano v0.4 — Browser Automation Test Cases

## Purpose
Verify Nano's browser-search → evidence-extraction → local-Ollama pipeline before the user performs manual acceptance testing.

**Environment:** Windows 11, project at `C:\Users\babul\Desktop\Nano`, existing `.venv`, Ollama installed.
**Current answering model:** `qwen3.5:2b-q4_K_M`
**Existing classifier constraint:** `qwen2.5:0.5b-instruct` may classify whether web access is required, but must not answer the user or choose/switch models in v0.4.
**Architecture constraint:** Browser retrieval is performed through Playwright controlling a dedicated persistent Chrome profile. Search results and page content are passed to the local Ollama model. No SearXNG or search API is required.

## Test strategy
1. Unit-test parsing, validation, timeouts, evidence formatting, and answer-prompt construction using mocked browser/model responses.
2. Run integration tests only when explicitly enabled; they may access the internet and launch Chrome.
3. Run the offline unit suite first. Do not require the user to test manually until automated tests pass.
4. Never bypass CAPTCHA, bot checks, authentication, or access-denied controls. Detect them and return a clear, safe error.

## Test cases

| ID | Type | Scenario | Expected result |
|---|---|---|---|
| TC-001 | Unit | Valid search-result card with title and URL | Extract one normalized result with title and URL |
| TC-002 | Unit | Result card missing title or URL | Skip invalid result and log a useful diagnostic |
| TC-003 | Unit | Duplicate result URLs | Deduplicate results |
| TC-004 | Unit | Search result points to a search-engine/internal navigation URL | Do not treat it as a destination result |
| TC-005 | Unit | URL uses an unsupported scheme such as `javascript:` | Reject URL |
| TC-006 | Unit | Result list is empty | Return a structured `NO_RESULTS` outcome, not a crash |
| TC-007 | Unit | Search page contains CAPTCHA, unusual-traffic, or access-denied text | Stop that engine attempt and report `BLOCKED`; do not attempt to bypass it |
| TC-008 | Unit | Search engine layout changes and known selector finds nothing | Try a bounded, safe fallback; if still empty, return a clear extraction error |
| TC-009 | Unit | Navigation timeout | Stop waiting, close/clean up resources, and return a timeout error |
| TC-010 | Unit | Page returns no readable body text | Mark the page as having insufficient content |
| TC-011 | Unit | Page text exceeds configured maximum | Truncate safely and mark it as truncated |
| TC-012 | Unit | Page metadata includes a publication date | Preserve it as an unverified metadata field with its source |
| TC-013 | Unit | No publication date is present | Do not invent one; keep publication date unknown |
| TC-014 | Unit | Multiple pages provide conflicting claims | Preserve both sources and flag conflict; do not silently choose one |
| TC-015 | Unit | Official source and secondary source both exist | Preserve source type/domain so answer generation can prioritize primary sources |
| TC-016 | Unit | Ollama receives evidence with URLs | Prompt includes the user question, page text, source URLs, and instructions not to invent facts |
| TC-017 | Unit | Evidence is empty or insufficient | Do not fabricate a confident answer; explain that verification failed |
| TC-018 | Unit | Ollama request fails or model is unavailable | Return a clear model error; clean up browser resources |
| TC-019 | Unit | User asks a non-web question | Do not launch Chrome; use the local answering brain |
| TC-020 | Unit | User asks a question requiring current web information | Classifier can request web retrieval; current brain remains the configured local Ollama model |
| TC-021 | Unit | Browser/context/page cleanup after success or exception | Close resources created by the run without unexpectedly closing unrelated user tabs |
| TC-022 | Unit | Several retries are triggered | Retry count remains bounded; no infinite loop |
| TC-023 | Unit | Search results include a stale snippet | Treat snippet as discovery evidence only; inspect the destination page before answering |
| TC-024 | Unit | Search result redirects to a different final URL | Record the final URL and retain the original result URL when useful |
| TC-025 | Unit | Webpage contains instructions telling the model to ignore its rules or execute commands | Treat page text as untrusted evidence, never as instructions to the agent |
| TC-026 | Integration (opt-in) | Dedicated Chrome profile launches | Chrome opens with Nano's dedicated profile, without using the user's everyday profile |
| TC-027 | Integration (opt-in) | Browser search works on a supported engine | Extract and print titles and destination URLs |
| TC-028 | Integration (opt-in) | Open a reachable article page | Extract title, final URL, readable text, retrieval timestamp, and available date metadata |
| TC-029 | Integration (opt-in) | Search engine blocks automation | Report blocked status clearly and stop that engine attempt |
| TC-030 | Integration (opt-in) | End-to-end question with reachable sources | Ollama answers using extracted evidence and includes source URLs |
| TC-031 | Integration (opt-in) | Internet is unavailable | Report retrieval failure; do not claim that web information was verified |
| TC-032 | Integration (opt-in) | Ollama is stopped | Report local-model unavailability without hanging indefinitely |
| TC-033 | Unit | Identity or ordinary conversation request | Route to local Ollama without invoking the classifier model or Chrome |
| TC-034 | Unit | Central debug switch is disabled/enabled | Suppress backend diagnostics when false; print them when true |
| TC-035 | Unit | Essential memory persistence error with debug disabled | Keep the actionable error visible |

## Acceptance criteria
- All offline unit tests pass before manual acceptance testing.
- Integration tests are marked separately and are opt-in because they require Chrome and internet access.
- No tests require real user credentials or the user's everyday Chrome profile.
- No test attempts to defeat CAPTCHA or anti-bot protections.
- Search snippets alone are not sufficient evidence for a current factual answer.
- Every web-grounded answer includes source URLs or explicitly states that verification was insufficient.
- Retries, navigation timeouts, page counts, and extracted-text size have explicit limits.
- Browser and Ollama errors are reported clearly and do not crash the whole application.
- The existing v0.3 classifier and local answering model responsibilities remain unchanged.

## Run commands

Install the declared dependencies in Nano's virtual environment:

```bat
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run Nano after offline tests pass:

```bat
.venv\Scripts\python.exe main.py
```

Set `NANO_CHROME_PATH` in Nano's `.env` only if Chrome is not in its standard Windows install location. Nano creates a separate profile under `%LOCALAPPDATA%\Nano\ChromeProfile`; do not point this at your everyday Chrome profile.

```bat
python -m pytest -q
```

Integration tests launch visible Chrome and access the internet only when explicitly opted in:

```bat
set NANO_RUN_INTEGRATION=1
python -m pytest -q -m integration
```

Set `NANO_RUN_OLLAMA_INTEGRATION=1` to opt in to a separate test that also requires the configured Ollama server and answering model.

The browser/Ollama boundary is mocked in offline tests. Do not claim tests passed unless they were actually executed and passed.

Set `DEBUG_LOGS = True` or `DEBUG_LOGS = False` in `config.py` to enable or silence backend diagnostics. User-facing answers and essential memory errors remain visible either way.
