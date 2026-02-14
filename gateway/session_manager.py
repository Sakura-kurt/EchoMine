import json
import uuid
from datetime import datetime
from typing import Optional, List

import redis.asyncio as redis

from .config import settings
from .tracing import trace_logger


class SessionManager:
    """Manages user sessions in Redis with TTL-based expiration."""

    def __init__(self):
        self.redis: Optional[redis.Redis] = None

    async def connect(self):
        """Initialize Redis connection."""
        self.redis = redis.from_url(settings.redis_url, decode_responses=True)

    async def close(self):
        """Close Redis connection."""
        if self.redis:
            await self.redis.close()

    async def create_session(self, user_id: str, trace_id: str = None) -> str:
        """
        Create a new session for a user. Returns session_id.
        """
        session_id = f"sess_{uuid.uuid4().hex[:16]}"
        session_data = {
            "session_id": session_id,
            "user_id": user_id,
            "created_at": datetime.utcnow().isoformat(),
            "last_active": datetime.utcnow().isoformat(),
            "conversation_history": [],
        }

        await self.redis.setex(
            f"session:{session_id}",
            settings.session_timeout_seconds,
            json.dumps(session_data)
        )

        # Map user to session for reconnection
        await self.redis.setex(
            f"user_session:{user_id}",
            settings.session_timeout_seconds,
            session_id
        )

        trace_logger.session_created(trace_id, user_id=user_id, session_id=session_id)
        return session_id

    async def get_session(self, session_id: str) -> Optional[dict]:
        """Get session data by session_id."""
        data = await self.redis.get(f"session:{session_id}")
        if not data:
            return None
        return json.loads(data)

    async def get_or_create_session(self, user_id: str, trace_id: str = None) -> tuple[str, bool]:
        """
        Get existing session for user or create new one.
        Returns (session_id, is_new_session).
        """
        # Check for existing session
        existing_session_id = await self.redis.get(f"user_session:{user_id}")

        if existing_session_id:
            session = await self.get_session(existing_session_id)
            if session:
                # Refresh TTL and update last_active
                await self._refresh_session(existing_session_id, session)
                history_length = len(session.get("conversation_history", []))
                trace_logger.session_resumed(trace_id, user_id=user_id, session_id=existing_session_id, history_length=history_length)
                return existing_session_id, False

        # Create new session
        session_id = await self.create_session(user_id, trace_id)
        return session_id, True

    async def _refresh_session(self, session_id: str, session_data: dict):
        """Refresh session TTL and update last_active."""
        session_data["last_active"] = datetime.utcnow().isoformat()

        await self.redis.setex(
            f"session:{session_id}",
            settings.session_timeout_seconds,
            json.dumps(session_data)
        )

        # Also refresh user_session mapping
        user_id = session_data.get("user_id")
        if user_id:
            await self.redis.setex(
                f"user_session:{user_id}",
                settings.session_timeout_seconds,
                session_id
            )

    async def update_session(self, session_id: str, **kwargs) -> bool:
        """Update session data and refresh TTL."""
        session = await self.get_session(session_id)
        if not session:
            return False

        session.update(kwargs)
        session["last_active"] = datetime.utcnow().isoformat()

        await self.redis.setex(
            f"session:{session_id}",
            settings.session_timeout_seconds,
            json.dumps(session)
        )
        return True

    async def add_to_history(self, session_id: str, role: str, text: str, trace_id: str = None) -> bool:
        """
        Add a message to conversation history.
        Role should be 'user' or 'assistant'.
        """
        session = await self.get_session(session_id)
        if not session:
            return False

        message = {
            "role": role,
            "text": text,
            "timestamp": datetime.utcnow().isoformat(),
        }

        history: List[dict] = session.get("conversation_history", [])
        history.append(message)

        # Keep last 100 messages to prevent unbounded growth
        if len(history) > 100:
            history = history[-100:]

        session["conversation_history"] = history
        session["last_active"] = datetime.utcnow().isoformat()

        await self.redis.setex(
            f"session:{session_id}",
            settings.session_timeout_seconds,
            json.dumps(session)
        )

        if role == "user":
            trace_logger.transcription(trace_id, session_id=session_id, text_length=len(text))

        return True

    async def get_history(self, session_id: str) -> List[dict]:
        """Get conversation history for a session."""
        session = await self.get_session(session_id)
        if not session:
            return []
        return session.get("conversation_history", [])

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        session = await self.get_session(session_id)
        if not session:
            return False

        user_id = session.get("user_id")

        await self.redis.delete(f"session:{session_id}")
        if user_id:
            await self.redis.delete(f"user_session:{user_id}")

        return True


# Global session manager instance
session_manager = SessionManager()
