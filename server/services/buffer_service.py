import io
import threading
import time
from collections import deque
from datetime import datetime, timezone

import mss
from PIL import Image

CAPTURE_INTERVAL = 3
MAX_BUFFER_SIZE = 20
RESIZE_WIDTH = 1280
JPEG_QUALITY = 60


class BufferService:
    def __init__(self):
        self._buffer = deque(maxlen=MAX_BUFFER_SIZE)
        self._lock = threading.Lock()
        self._running = False
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        self._buffer.clear()
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1)
            self._thread = None
        self._buffer.clear()

    def _capture_loop(self):
        while self._running:
            snapshot = self._take_screenshot()
            if snapshot:
                with self._lock:
                    self._buffer.append(snapshot)
            if self._stop_event.wait(timeout=CAPTURE_INTERVAL):
                break

    def _take_screenshot(self):
        try:
            with mss.mss() as sct:
                # multiple monitors consideration?
                monitor = sct.monitors[0]
                raw = sct.grab(monitor)
                img = Image.frombytes("RGB", raw.size, raw.rgb)

                scale = RESIZE_WIDTH / img.width
                new_size = (RESIZE_WIDTH, int(img.height * scale))
                img = img.resize(new_size, Image.LANCZOS)

                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=JPEG_QUALITY)
                jpeg_bytes = buf.getvalue()

                return {
                    "image_bytes": jpeg_bytes,
                    "mime_type": "image/jpeg",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
        except Exception as e:
            print(f"Screenshot capture failed: {e}")
            return None

    def flush_buffer(self, clear=True, max_send=8):
        with self._lock:
            all_snapshots = list(self._buffer)
            if clear:
                self._buffer.clear()
        sampled = self._sample(all_snapshots, max_send)
        print(f"[flush_buffer] captured={len(all_snapshots)} sending={len(sampled)}")
        return sampled

    @staticmethod
    def _sample(snapshots, max_send):
        n = len(snapshots)
        if n <= max_send:
            return snapshots
        recent = snapshots[-(max_send // 2):]
        earlier = snapshots[:n - len(recent)]
        step = max(1, len(earlier) // (max_send - len(recent)))
        sampled_early = earlier[::step][:max_send - len(recent)]
        return sampled_early + recent

    def get_status(self):
        with self._lock:
            count = len(self._buffer)
        return {
            "running": self._running,
            "buffer_count": count,
            "max_buffer_size": MAX_BUFFER_SIZE,
            "capture_interval_sec": CAPTURE_INTERVAL,
        }
