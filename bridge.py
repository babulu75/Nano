from Brain.brain import think

class AssistantBridge:

    def process(
        self,
        user_input: str,
        online: bool = True
    ) -> str:

        response = think(
            user_input,
            online=online
        )

        return response