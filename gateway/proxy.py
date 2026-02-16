import asyncio
import json
import time
from typing import Optional

import websockets
from fastapi import WebSocket

from .config import settings
from .session_manager import session_manager
from .tracing import trace_logger


class STTProxy:
    """Bidirectional WebSocket proxy between client and STT server."""

    def __init__(
        self,
        client_ws: WebSocket,
        user_id: str,
        session_id: str,
        trace_id: str,
    ):
        self.client_ws = client_ws
        self.user_id = user_id
        self.session_id = session_id
        self.trace_id = trace_id
        self.stt_ws: Optional[websockets.WebSocketClientProtocol] = None
        self.start_time = time.time()
        self._closed = False

    async def connect_to_stt(self) -> bool:
        """Establish connection to STT server."""
        try:
            self.stt_ws = await websockets.connect(settings.stt_server_url)
            trace_logger.stt_proxy_connected(self.trace_id, stt_url=settings.stt_server_url)
            return True
        except Exception as e:
            trace_logger.error(
                "stt_connection_failed",
                self.trace_id,
                error=str(e),
                session_id=self.session_id
            )
            return False

    async def forward_client_to_stt(self):
        """Forward messages from client to STT server."""
        try:
            while not self._closed:
                # Receive from client (can be text or binary)
                message = await self.client_ws.receive()

                if message["type"] == "websocket.disconnect":
                    break

                if "bytes" in message and message["bytes"]:
                    # Binary audio data
                    await self.stt_ws.send(message["bytes"])
                elif "text" in message and message["text"]:
                    # Text message (if any)
                    await self.stt_ws.send(message["text"])

        except Exception as e:
            if not self._closed:
                trace_logger.error(
                    "client_forward_error",
                    self.trace_id,
                    error=str(e),
                    session_id=self.session_id
                )

    async def forward_stt_to_client(self):
        """Forward messages from STT server to client, intercepting for session history."""
        try:
            async for message in self.stt_ws:
                if self._closed:
                    break

                if isinstance(message, bytes):
                    # Binary data (unlikely from STT, but handle it)
                    await self.client_ws.send_bytes(message)
                else:
                    # Text/JSON message
                    await self.client_ws.send_text(message)

                    # Intercept and process STT events
                    await self._process_stt_message(message)

        except websockets.ConnectionClosed:
            pass
        except Exception as e:
            if not self._closed:
                trace_logger.error(
                    "stt_forward_error",
                    self.trace_id,
                    error=str(e),
                    session_id=self.session_id
                )

    async def _process_stt_message(self, message: str):
        """Process STT server messages for logging and session history."""
        try:
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "speech_start":
                trace_logger.speech_start(self.trace_id, session_id=self.session_id)

            elif msg_type == "speech_end":
                trace_logger.speech_end(self.trace_id, session_id=self.session_id)

            elif msg_type == "final":
                # Transcription result - save to session history
                text = data.get("text", "")
                if text.strip():
                    await session_manager.add_to_history(
                        self.session_id,
                        role="user",
                        text=text,
                        trace_id=self.trace_id
                    )

            elif msg_type == "answer":
                # RAG answer relayed through STT - log to session history
                response = data.get("response", "")
                if response.strip():
                    await session_manager.add_to_history(
                        self.session_id,
                        role="assistant",
                        text=response,
                        trace_id=self.trace_id
                    )

            elif msg_type == "error":
                trace_logger.transcription_error(
                    self.trace_id,
                    session_id=self.session_id,
                    error=data.get("message", "unknown"),
                    stage=data.get("stage")
                )

        except json.JSONDecodeError:
            pass  # Non-JSON message, ignore

    async def run(self):
        """Run the bidirectional proxy."""
        if not await self.connect_to_stt():
            await self.client_ws.close(code=1011, reason="Failed to connect to STT server")
            return

        try:
            # Run both directions concurrently
            await asyncio.gather(
                self.forward_client_to_stt(),
                self.forward_stt_to_client(),
            )
        finally:
            await self.close("proxy_complete")

    async def close(self, reason: str = "unknown"):
        """Clean up connections."""
        if self._closed:
            return

        self._closed = True
        duration_ms = int((time.time() - self.start_time) * 1000)

        # Close STT connection
        if self.stt_ws:
            try:
                await self.stt_ws.close()
            except Exception:
                pass

        trace_logger.connection_end(
            self.trace_id,
            user_id=self.user_id,
            session_id=self.session_id,
            duration_ms=duration_ms,
            reason=reason
        )
