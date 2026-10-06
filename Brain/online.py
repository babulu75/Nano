from google import genai
from config import GEMINI_API_KEY, ONLINE_MODEL


client = genai.Client(
    api_key=GEMINI_API_KEY
)


def think_online(prompt: str) -> str:

    response = client.models.generate_content(
        model=ONLINE_MODEL,
        contents=prompt
    )

    return response.text