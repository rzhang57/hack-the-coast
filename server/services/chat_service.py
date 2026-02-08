from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

load_dotenv()

# MODEL = "gemini-2.5-flash"
MODEL = "gemini-2.5-flash-lite"
SYSTEM_PROMPT = """You are an ADHD productivity assistant. You will receive screenshots of the user's recent screen activity.

Response format (strict):
1. One concise sentence acknowledging the user's effort and reassuring them you're here to help.
2. One concise sentence suggesting a very specific, concrete next step to get them back on track.
3. Ask a concise, supportive, thought provoking question.

Never exceed 3 sentences total. Never use bullet points, headers, or lists. Write in a warm but direct tone."""


class ChatService:
    def __init__(self):
        self._client = genai.Client(api_key=os.getenv("CHAT_API_KEY"))
        self._history = []

    def init_chat_stream(self, snapshots):
        self._history = []
        labeled_parts = []
        total = len(snapshots)
        half = total // 2
        for i, snap in enumerate(snapshots):
            if i < total - half:
                label = f"[Screenshot {i + 1}/{total} — earlier context]"
            elif i < total - 1:
                label = f"[Screenshot {i + 1}/{total} — recent]"
            else:
                label = f"[Screenshot {i + 1}/{total} — CURRENT SCREEN]"
            labeled_parts.append(types.Part.from_text(text=label))
            labeled_parts.append(types.Part.from_bytes(data=snap["image_bytes"], mime_type=snap["mime_type"]))
        prompt = "The user just said they're stuck. These screenshots are chronological. The last screenshot is their CURRENT screen — base your response primarily on it. Earlier screenshots show how they got there. Respond following your format."
        self._history.append(types.Content(role="user", parts=[types.Part.from_text(text=prompt)] + labeled_parts))
        full_text = yield from self._stream_response()
        return full_text

    def send_message_stream(self, message):
        self._history.append(types.Content(role="user", parts=[types.Part.from_text(text=message)]))
        full_text = yield from self._stream_response()
        return full_text

    def _stream_response(self):
        response_stream = self._client.models.generate_content_stream(
            model=MODEL,
            contents=self._history,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
        )
        full_text = ""
        for chunk in response_stream:
            if chunk.text:
                full_text += chunk.text
                yield chunk.text
        self._history.append(types.Content(role="model", parts=[types.Part.from_text(text=full_text)]))
        return full_text
