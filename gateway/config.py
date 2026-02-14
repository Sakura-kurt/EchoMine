import os
from dataclasses import dataclass


@dataclass
class Settings:
    # Redis
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")

    # JWT
    jwt_secret: str = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24

    # Session
    session_timeout_seconds: int = 30 * 60  # 30 minutes
    max_session_age_seconds: int = 24 * 60 * 60  # 24 hours

    # STT Server
    stt_server_url: str = os.getenv("STT_SERVER_URL", "ws://localhost:8001/ws/stt")

    # Gateway
    gateway_host: str = os.getenv("GATEWAY_HOST", "0.0.0.0")
    gateway_port: int = int(os.getenv("GATEWAY_PORT", "8000"))


settings = Settings()