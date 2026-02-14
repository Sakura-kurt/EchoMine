import json
import uuid
from datetime import datetime, timedelta
from typing import Optional

import bcrypt
import jwt
import redis.asyncio as redis

from .config import settings
from .tracing import trace_logger


class AuthManager:
    """Handles JWT authentication and user management with Redis."""

    def __init__(self):
        self.redis: Optional[redis.Redis] = None

    async def connect(self):
        """Initialize Redis connection."""
        self.redis = redis.from_url(settings.redis_url, decode_responses=True)

    async def close(self):
        """Close Redis connection."""
        if self.redis:
            await self.redis.close()

    # Password hashing
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash."""
        return bcrypt.checkpw(password.encode(), hashed.encode())

    # JWT operations
    def create_access_token(self, user_id: str) -> str:
        """Create a JWT access token."""
        payload = {
            "sub": user_id,
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(hours=settings.jwt_expiration_hours),
            "jti": uuid.uuid4().hex,  # Unique token ID
        }
        return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)

    def decode_token(self, token: str) -> Optional[dict]:
        """Decode and validate JWT token. Returns payload or None if invalid."""
        try:
            payload = jwt.decode(
                token,
                settings.jwt_secret,
                algorithms=[settings.jwt_algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    # User management
    async def register_user(self, username: str, password: str) -> Optional[str]:
        """
        Register a new user. Returns user_id if successful, None if username exists.
        """
        # Check if username already exists
        existing = await self.redis.get(f"username:{username}")
        if existing:
            return None

        user_id = f"user_{uuid.uuid4().hex[:12]}"
        user_data = {
            "user_id": user_id,
            "username": username,
            "password_hash": self.hash_password(password),
            "created_at": datetime.utcnow().isoformat(),
            "is_active": True,
        }

        # Store user by user_id
        await self.redis.set(f"user:{user_id}", json.dumps(user_data))
        # Map username to user_id for login lookup
        await self.redis.set(f"username:{username}", user_id)

        return user_id

    async def authenticate_user(self, username: str, password: str, trace_id: str = None, ip_address: str = None) -> Optional[str]:
        """
        Authenticate user credentials. Returns JWT token if valid, None otherwise.
        """
        # Get user_id from username
        user_id = await self.redis.get(f"username:{username}")
        if not user_id:
            trace_logger.auth_failure(trace_id, reason="user_not_found", ip_address=ip_address)
            return None

        # Get user data
        user_data = await self.redis.get(f"user:{user_id}")
        if not user_data:
            trace_logger.auth_failure(trace_id, reason="user_data_missing", ip_address=ip_address)
            return None

        user = json.loads(user_data)

        # Check if active
        if not user.get("is_active", False):
            trace_logger.auth_failure(trace_id, reason="user_inactive", ip_address=ip_address)
            return None

        # Verify password
        if not self.verify_password(password, user["password_hash"]):
            trace_logger.auth_failure(trace_id, reason="invalid_password", ip_address=ip_address)
            return None

        # Create token
        token = self.create_access_token(user_id)
        trace_logger.auth_success(trace_id, user_id=user_id)

        return token

    async def get_user(self, user_id: str) -> Optional[dict]:
        """Get user data by user_id."""
        user_data = await self.redis.get(f"user:{user_id}")
        if not user_data:
            return None
        return json.loads(user_data)

    async def get_current_user(self, token: str) -> Optional[dict]:
        """Extract and return user from JWT token."""
        payload = self.decode_token(token)
        if not payload:
            return None

        user_id = payload.get("sub")
        if not user_id:
            return None

        return await self.get_user(user_id)


# Global auth manager instance
auth_manager = AuthManager()