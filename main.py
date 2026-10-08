import sys
from bridge import NanoBridge

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main():

    nano = NanoBridge()

    print()
    print("NANO v0.3")
    print("Local Ollama brain")
    print()

    try:

        while True:

            user_input = input("You: ")

            if not user_input.strip():
                continue

            if user_input.lower() in (
                "exit",
                "quit"
            ):
                break

            if user_input.lower() in (
                "clear",
                "reset"
            ):
                nano.clear_memory()
                print("-> Conversation memory cleared.")
                continue

            response = nano.process(
                user_input
            )

            print(
                f"Assistant: {response}"
            )

    finally:

        nano.shutdown()


if __name__ == "__main__":
    main()
