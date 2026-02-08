from google import genai
from google.genai import types
import json
import os
from dotenv import load_dotenv

load_dotenv()

MODEL = "gemini-2.5-flash-lite"
SYSTEM_PROMPT = """You are an ADHD productivity assistant. You will receive screenshots of the user's recent screen activity.

Response format (strict):
1. One concise sentence acknowledging the user's effort and reassuring them you're here to help.
2. One concise sentence suggesting a very specific, concrete next step to get them back on track.
3. Ask a concise, supportive, thought provoking question.

Never exceed 3 sentences total. Never use bullet points, headers, or lists. Write in a warm but direct tone."""

USERINFO_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'user_info.json')


class ChatService:
    def __init__(self):
        self._cached_key = None
        self._client = None
        self._history = []

    def _get_client(self):
        key = self._read_api_key()
        if key and key != self._cached_key:
            self._cached_key = key
            self._client = genai.Client(api_key=key)
        if not self._client:
            raise RuntimeError("No Gemini API key configured")
        return self._client

    def _read_api_key(self):
        if os.path.exists(USERINFO_PATH):
            with open(USERINFO_PATH, 'r') as f:
                data = json.load(f)
                key = data.get('gemini_api_key')
                if key:
                    return key
        return os.getenv("CHAT_API_KEY")

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
        response_stream = self._get_client().models.generate_content_stream(
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
