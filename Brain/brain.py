from .local import think_local
from .online import think_online
from .router import choose_brain


def think(prompt: str, online: bool = True) -> str:

    selected_brain = choose_brain(online)

    if selected_brain == "online":
        return think_online(prompt)

    return think_local(prompt)