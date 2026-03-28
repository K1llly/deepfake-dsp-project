import threading
from datetime import datetime


class LogEntry:
    def __init__(self, message, level="info"):
        self.timestamp = datetime.now().strftime("%H:%M:%S")
        self.message = message
        self.level = level

    def __str__(self):
        tag = self.level.upper()
        return f"[{self.timestamp}] [{tag}] {self.message}"


class AppLogger:
    """Global logger that UI components can subscribe to."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._entries = []
                cls._instance._listeners = []
            return cls._instance

    def info(self, message):
        self._add(message, "info")

    def success(self, message):
        self._add(message, "success")

    def error(self, message):
        self._add(message, "error")

    def warning(self, message):
        self._add(message, "warning")

    def get_entries(self):
        return self._entries.copy()

    def subscribe(self, callback):
        self._listeners.append(callback)

    def clear(self):
        self._entries.clear()

    def _add(self, message, level):
        entry = LogEntry(message, level)
        self._entries.append(entry)
        for listener in self._listeners:
            try:
                listener(entry)
            except Exception:
                pass


def get_logger():
    return AppLogger()
