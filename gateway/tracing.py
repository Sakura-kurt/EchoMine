import json
import logging
import sys
import uuid
from datetime import datetime
from typing import Any


def generate_trace_id() -> str:
    """Generate a unique trace ID for request tracking."""
    return uuid.uuid4().hex[:16]


class TraceLogger:
    """Structured JSON logger for tracing events."""

    def __init__(self, name: str = "gateway"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # Avoid duplicate handlers
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(handler)

    def _log(self, level: str, event: str, trace_id: str = None, **kwargs: Any):
        """Internal method to format and emit log."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": level,
            "event": event,
        }

        if trace_id:
            log_entry["trace_id"] = trace_id

        log_entry.update(kwargs)

        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(json.dumps(log_entry))

    def info(self, event: str, trace_id: str = None, **kwargs: Any):
        self._log("INFO", event, trace_id, **kwargs)

    def error(self, event: str, trace_id: str = None, **kwargs: Any):
        self._log("ERROR", event, trace_id, **kwargs)

    def warning(self, event: str, trace_id: str = None, **kwargs: Any):
        self._log("WARNING", event, trace_id, **kwargs)

    # Convenience methods for common events
    def connection_start(self, trace_id: str, user_id: str, session_id: str, ip_address: str = None):
        self.info("connection_start", trace_id, user_id=user_id, session_id=session_id, ip_address=ip_address)

    def connection_end(self, trace_id: str, user_id: str, session_id: str, duration_ms: int, reason: str):
        self.info("connection_end", trace_id, user_id=user_id, session_id=session_id, duration_ms=duration_ms, reason=reason)

    def auth_success(self, trace_id: str, user_id: str):
        self.info("auth_success", trace_id, user_id=user_id)

    def auth_failure(self, trace_id: str, reason: str, ip_address: str = None):
        self.warning("auth_failure", trace_id, reason=reason, ip_address=ip_address)

    def session_created(self, trace_id: str, user_id: str, session_id: str):
        self.info("session_created", trace_id, user_id=user_id, session_id=session_id)

    def session_resumed(self, trace_id: str, user_id: str, session_id: str, history_length: int):
        self.info("session_resumed", trace_id, user_id=user_id, session_id=session_id, history_length=history_length)

    def stt_proxy_connected(self, trace_id: str, stt_url: str):
        self.info("stt_proxy_connected", trace_id, stt_url=stt_url)

    def speech_start(self, trace_id: str, session_id: str):
        self.info("speech_start", trace_id, session_id=session_id)

    def speech_end(self, trace_id: str, session_id: str):
        self.info("speech_end", trace_id, session_id=session_id)

    def transcription(self, trace_id: str, session_id: str, text_length: int, duration_ms: int = None):
        self.info("transcription", trace_id, session_id=session_id, text_length=text_length, duration_ms=duration_ms)

    def transcription_error(self, trace_id: str, session_id: str, error: str, stage: str = None):
        self.error("transcription_error", trace_id, session_id=session_id, error=error, stage=stage)


# Global logger instance
trace_logger = TraceLogger()
