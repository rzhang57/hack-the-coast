import threading
import time

from pynput import keyboard, mouse

DISENGAGEMENT_THRESHOLD = 5


class KeyboardService:
    def __init__(self):
        self._last_activity = time.time()
        self._lock = threading.Lock()
        self._keyboard_listener = None
        self._mouse_listener = None

    def _update_activity(self, *args, **kwargs):
        with self._lock:
            self._last_activity = time.time()

    @property
    def is_disengaged(self):
        with self._lock:
            return time.time() - self._last_activity > DISENGAGEMENT_THRESHOLD

    def start(self):
        if self._keyboard_listener is not None:
            return
        with self._lock:
            self._last_activity = time.time()
        self._keyboard_listener = keyboard.Listener(on_press=self._update_activity)
        self._keyboard_listener.daemon = True
        self._keyboard_listener.start()
        self._mouse_listener = mouse.Listener(
            on_click=self._update_activity,
            on_move=self._update_activity,
            on_scroll=self._update_activity,
        )
        self._mouse_listener.daemon = True
        self._mouse_listener.start()

    def stop(self):
        for listener in (self._keyboard_listener, self._mouse_listener):
            if listener is not None:
                listener.stop()
                listener.join(timeout=2)
        self._keyboard_listener = None
        self._mouse_listener = None
