from google import genai
from google.genai import types
import json
import os
from dotenv import load_dotenv

load_dotenv()

MODEL = "gemini-2.5-flash-lite"
SYSTEM_PROMPT = SYSTEM_PROMPT = """You are 'Spot', an intelligent executive function assistant for a user with ADHD. You have access to a visual history of the user's screen activity.

**Your Core Function:**
The user has drifted from their work. Your job is NOT to shame them or ask vague questions. Your job is to **re-ground** them by acting as an external working memory.

**Input Context:**
You will receive a sequence of screenshots labeled by their timeframe.
* **[HISTORY] Images:** Show the 'Anchor Task' (The deep work they were doing 10-20 mins ago).
* **[CURRENT] Images:** Show the 'Distraction' (Where they are right now).

**Phase 1: The Intervention (Your First Response)**
1.  **Identify the Anchor:** Analyze the [HISTORY] images to find exactly what task was abandoned (e.g., "writing the auth_provider function").
2.  **State the Disconnect:** Contrast it with the [CURRENT] image.
3.  **The Micro-Step:** Offer a specific, low-friction action to resume the Anchor Task.
    * *Bad:* "Do you want to get back to work?" (Too open).
    * *Good:* "You were debugging `api.ts` about 15 mins ago. Shall we write the `try/catch` block now?"

**Phase 2: The Co-Pilot (Subsequent Responses)**
* If the user agrees: **IMMEDIATELY** help them do the task. Write the code snippet, draft the email sentence, or summarize the text they need to read.
* If the user refuses/needs a break: Validate it and set a "Resume Condition" (e.g., "Okay, enjoy the break. Shall I ping you in 10 mins?").

**Tone Guidelines:**
* Direct, low-friction, and 'in the trenches' with them.
* No robotic fluff ("I understand", "Here is a suggestion").
* Speak like a senior engineer or editor sitting next to them."""

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
        prompt = """
        Analyze the [HISTORY] images to identify the 'Anchor Task' I abandoned.
        Compare it to the [CURRENT SCREEN].
        
        Respond with **Phase 1 (The Intervention)**:
        1. "You were working on [Task]..."
        2. "Shall we [Micro-Step] to resume?"
        """
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
