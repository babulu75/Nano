import asyncio
import sys
from assistant import NanoAssistant
from config import debug_log

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


async def run_nano():

    # Feature: terminal lifecycle diagnostics — Nano v0.4 — Purpose: include startup and exit events in the central debug switch.
    debug_log("Nano CLI starting.")

    nano = NanoAssistant()

    print()
    print("NANO v0.4")
    print("Local Ollama brain + visible Chrome web research")
    print()

    try:

        while True:

            user_input = await asyncio.to_thread(input, "You: ")

            if not user_input.strip():
                continue

            if user_input.lower() in (
                "exit",
                "quit"
            ):
                debug_log("Nano CLI exit requested.")
                break

            if user_input.lower() in (
                "clear",
                "reset"
            ):
                nano.clear_memory()
                print("-> Conversation memory cleared.")
                continue

            response = await nano.process(user_input)

            print(
                f"Assistant: {response}"
            )

    finally:
        debug_log("Nano CLI shutting down.")
        await nano.shutdown()


def main():
    asyncio.run(run_nano())


if __name__ == "__main__":
    main()
