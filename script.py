import time
import uuid
from typing import Optional, Dict

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# =============================
# App
# =============================
app = FastAPI(title="LLM Gateway (API Layer)", version="0.1")

# =============================
# (1) Config / Secrets
# =============================
# 实习生版本：直接写死
# 企业里：环境变量 / Secret Manager
VALID_API_KEYS = {"intern-dev-key"}

# =============================
# (2) In-memory Rate Limit
# =============================
_last_seen: Dict[str, float] = {}

def enforce_rate_limit(key: str, min_interval_sec: float = 0.3) -> None:
    now = time.time()
    last = _last_seen.get(key, 0.0)
    if now - last < min_interval_sec:
        raise HTTPException(status_code=429, detail="RATE_LIMITED")
    _last_seen[key] = now

# =============================
# (3) Schemas
# =============================
class ChatRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=64)
    session_id: str = Field(min_length=1, max_length=64)
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)

class ChatResponse(BaseModel):
    request_id: str
    answer: str
    latency_ms: int

# =============================
# (4) Error Response Helper
# =============================
def error_response(request_id: str, code: str, status_code: int, message: str = ""):
    return JSONResponse(
        status_code=status_code,
        content={
            "request_id": request_id,
            "error": {
                "code": code,
                "message": message or code,
            },
        },
    )

# =============================
# (5) Middleware
# request_id + latency + basic log
# =============================
@app.middleware("http")
async def api_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    start = time.time()

    try:
        response = await call_next(request)
        latency_ms = int((time.time() - start) * 1000)

        response.headers["x-request-id"] = request_id
        response.headers["x-latency-ms"] = str(latency_ms)

        # 实习生级别日志（够用）
        print({
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "latency_ms": latency_ms,
        })
        return response

    except HTTPException as e:
        latency_ms = int((time.time() - start) * 1000)
        print({
            "request_id": request_id,
            "status": e.status_code,
            "error": str(e.detail),
            "latency_ms": latency_ms,
        })
        return error_response(request_id, "HTTP_ERROR", e.status_code, str(e.detail))

    except Exception:
        latency_ms = int((time.time() - start) * 1000)
        print({
            "request_id": request_id,
            "status": 500,
            "error": "INTERNAL_ERROR",
            "latency_ms": latency_ms,
        })
        return error_response(request_id, "INTERNAL_ERROR", 500)

# =============================
# (6) Auth Helper
# =============================
def require_api_key(x_api_key: Optional[str]) -> str:
    if not x_api_key or x_api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="UNAUTHORIZED")
    return x_api_key

# =============================
# (7) API Endpoint
# =============================
@app.post("/v1/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    x_api_key: Optional[str] = Header(default=None),
):
    # API 层只做：校验 + 保护
    key = require_api_key(x_api_key)
    enforce_rate_limit(key)

    # 真实企业中，这里会调用：
    # result = orchestrator.handle(req)
    # API 层不实现业务逻辑
    answer = "ACK: request accepted (API layer only)."

    return ChatResponse(
        request_id=str(uuid.uuid4()),
        answer=answer,
        latency_ms=0,
    )