from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from pydantic import BaseModel

from .auth import auth_manager
from .config import settings
from .proxy import STTProxy
from .session_manager import session_manager
from .tracing import generate_trace_id, trace_logger


# Request/Response models
class RegisterRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    user_id: str
    username: str


class SessionResponse(BaseModel):
    session_id: str
    user_id: str
    created_at: str
    last_active: str
    history_length: int


# Lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await auth_manager.connect()
    await session_manager.connect()
    trace_logger.info("gateway_started", host=settings.gateway_host, port=settings.gateway_port)
    yield
    # Shutdown
    await auth_manager.close()
    await session_manager.close()
    trace_logger.info("gateway_stopped")


app = FastAPI(
    title="STT Gateway",
    description="Authentication and session management gateway for STT service",
    version="1.0.0",
    lifespan=lifespan,
)


# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# Auth endpoints
@app.post("/auth/register", response_model=UserResponse)
async def register(request: RegisterRequest):
    trace_id = generate_trace_id()
    trace_logger.info("register_request", trace_id, username=request.username)

    user_id = await auth_manager.register_user(request.username, request.password)

    if not user_id:
        trace_logger.warning("register_failed", trace_id, username=request.username, reason="username_exists")
        raise HTTPException(status_code=400, detail="Username already exists")

    trace_logger.info("register_success", trace_id, user_id=user_id)
    return UserResponse(user_id=user_id, username=request.username)


@app.post("/auth/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    trace_id = generate_trace_id()
    trace_logger.info("login_request", trace_id, username=request.username)

    token = await auth_manager.authenticate_user(
        request.username,
        request.password,
        trace_id=trace_id
    )

    if not token:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return TokenResponse(access_token=token)


@app.post("/auth/refresh", response_model=TokenResponse)
async def refresh_token(token: str = Query(..., description="Current JWT token")):
    trace_id = generate_trace_id()

    # Decode current token
    user = await auth_manager.get_current_user(token)
    if not user:
        trace_logger.auth_failure(trace_id, reason="invalid_token_refresh")
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    # Issue new token
    new_token = auth_manager.create_access_token(user["user_id"])
    trace_logger.info("token_refreshed", trace_id, user_id=user["user_id"])

    return TokenResponse(access_token=new_token)


# Session info endpoint (optional, for debugging)
@app.get("/session/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str, token: str = Query(..., description="JWT token")):
    trace_id = generate_trace_id()

    # Validate token
    user = await auth_manager.get_current_user(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    # Get session
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Check ownership
    if session.get("user_id") != user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")

    return SessionResponse(
        session_id=session["session_id"],
        user_id=session["user_id"],
        created_at=session["created_at"],
        last_active=session["last_active"],
        history_length=len(session.get("conversation_history", []))
    )


# WebSocket STT endpoint
@app.websocket("/ws/stt")
async def websocket_stt(
    websocket: WebSocket,
    token: Optional[str] = Query(None, description="JWT token"),
):
    trace_id = generate_trace_id()
    client_ip = websocket.client.host if websocket.client else "unknown"

    # Accept connection first (required for WebSocket)
    await websocket.accept()

    # Validate token
    if not token:
        trace_logger.auth_failure(trace_id, reason="missing_token", ip_address=client_ip)
        await websocket.close(code=4001, reason="Missing token")
        return

    user = await auth_manager.get_current_user(token)
    if not user:
        trace_logger.auth_failure(trace_id, reason="invalid_token", ip_address=client_ip)
        await websocket.close(code=4001, reason="Invalid or expired token")
        return

    user_id = user["user_id"]
    trace_logger.auth_success(trace_id, user_id=user_id)

    # Get or create session
    session_id, is_new = await session_manager.get_or_create_session(user_id, trace_id)

    # Log connection start
    trace_logger.connection_start(trace_id, user_id=user_id, session_id=session_id, ip_address=client_ip)

    # Create and run proxy
    proxy = STTProxy(
        client_ws=websocket,
        user_id=user_id,
        session_id=session_id,
        trace_id=trace_id,
    )

    try:
        await proxy.run()
    except WebSocketDisconnect:
        await proxy.close("client_disconnect")
    except Exception as e:
        trace_logger.error("websocket_error", trace_id, error=str(e), session_id=session_id)
        await proxy.close("error")


# Run with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "gateway.main:app",
        host=settings.gateway_host,
        port=settings.gateway_port,
        reload=True,
    )
