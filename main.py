from bridge import AssistantBridge
from Input.text_output import get_input
from Output.text_output import show_output


assistant = AssistantBridge()


def main():

    print("================================")
    print("       ASSISTANT BRAIN")
    print("================================")
    print("Type 'offline' for local mode")
    print("Type 'online' for Gemini mode")
    print("Type 'exit' to stop")

    online = True

    while True:

        user_input = get_input()

        if user_input.lower() == "exit":
            print("Assistant stopped.")
            break

        if user_input.lower() == "offline":
            online = False
            print("→ Offline brain: Qwen")
            continue

        if user_input.lower() == "online":
            online = True
            print("→ Online brain: Gemini")
            continue

        try:

            response = assistant.process(
                user_input,
                online=online
            )

            show_output(response)

        except Exception as error:

            print(f"\nError: {error}")


if __name__ == "__main__":
    main()