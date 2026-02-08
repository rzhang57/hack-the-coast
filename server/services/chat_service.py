from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

load_dotenv()

MODEL = "gemini-2.5-flash"
SYSTEM_PROMPT = """You are an ADHD productivity assistant. You will receive screenshots of the user's recent screen activity.

Response format (strict):
1. One short sentence acknowledging the user's effort and reassuring them you're here to help.
2. One to two sentences suggesting a specific, concrete next step to get them back on track.

Never exceed 3 sentences total. Never use bullet points, headers, or lists. Write in a warm but direct tone."""


class ChatService:
    def __init__(self):
        self._client = genai.Client(api_key=os.getenv("CHAT_API_KEY"))
        self._history = []

    def init_chat(self, snapshots):
        self._history = []
        parts = [
            types.Part.from_bytes(data=snap["image_bytes"], mime_type=snap["mime_type"])
            for snap in snapshots
        ]
        prompt = "The user just said they're stuck. Analyze these screenshots to understand what they were doing, then respond following your format."
        self._history.append(types.Content(role="user", parts=[types.Part.from_text(text=prompt)] + parts))
        response = self._client.models.generate_content(
            model=MODEL,
            contents=self._history,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
        )
        self._history.append(types.Content(role="model", parts=[types.Part.from_text(text=response.text)]))
        return response.text

    def send_message(self, message):
        self._history.append(types.Content(role="user", parts=[types.Part.from_text(text=message)]))
        response = self._client.models.generate_content(
            model=MODEL,
            contents=self._history,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
        )
        self._history.append(types.Content(role="model", parts=[types.Part.from_text(text=response.text)]))
        return response.text
