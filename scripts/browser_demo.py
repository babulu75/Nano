"""Standalone smoke-test entry point for Nano's visible Chrome research pipeline."""

import asyncio

from assistant import NanoAssistant
from tools.browser.research_pipeline import ResearchError, format_evidence


def main() -> int:
    print("NANO v0.4 — visible Chrome search + local Ollama")
    query = input("Ask Nano something to search: ").strip()
    if not query:
        print("No question entered.")
        return 1

    async def run_query():
        nano = NanoAssistant()
        try:
            bundle = await nano.researcher.research(query)
            return nano.brain.generate(query, tool_context=format_evidence(bundle))
        finally:
            await nano.shutdown()

    try:
        answer = asyncio.run(run_query())
    except KeyboardInterrupt:
        print("\nStopped by user.")
        return 130
    except ResearchError as exc:
        print(f"[Nano] Web research could not verify this request: {exc}")
        return 1
    print("\nNano:")
    print(answer)
    return 0 if not answer.startswith("[Nano]") else 1


if __name__ == "__main__":
    raise SystemExit(main())
