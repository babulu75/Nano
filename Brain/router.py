def choose_brain(online: bool) -> str:

    if online:
        return "online"

    return "local"