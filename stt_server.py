import asyncio
import json
import time
import uuid
from typing import Dict

import aio_pika
import numpy as np
import webrtcvad
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from faster_whisper import WhisperModel

from rabbitmq_config import (
    get_connection, setup_exchanges_and_queues,
    RAG_EXCHANGE,
)

app = FastAPI(title="Streaming STT (VAD + FasterWhisper)")

# ===== Audio settings (MUST match client) =====
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2  # int16 => 2 bytes
FRAME_MS = 20
FRAME_BYTES = int(SAMPLE_RATE * FRAME_MS / 1000.0 * SAMPLE_WIDTH)  # 640

# ===== VAD settings =====
vad = webrtcvad.Vad(2)
SILENCE_CUTOFF_MS = 700
MIN_UTTERANCE_MS = 250

# ===== Whisper model (tiny first for stability/speed) =====
model = WhisperModel("tiny", device="cpu", compute_type="int8")

# ===== RabbitMQ state =====
rmq_connection: aio_pika.abc.AbstractRobustConnection | None = None
rmq_channel: aio_pika.abc.AbstractChannel | None = None
rag_exchange: aio_pika.abc.AbstractExchange | None = None


def transcribe_blocking(pcm: bytes) -> str:
    """
    pcm: raw int16 mono 16kHz.
    Use numpy waveform to avoid PyAV file-like issues.
    """
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    segments, _info = model.transcribe(audio, language="en", vad_filter=False)
    return "".join(seg.text for seg in segments).strip()


@app.on_event("startup")
async def startup():
    global rmq_connection, rmq_channel, rag_exchange
    print("[stt] Connecting to RabbitMQ...")
    rmq_connection = await get_connection()
    rmq_channel = await rmq_connection.channel()
    rag_ex, _ = await setup_exchanges_and_queues(rmq_channel)
    rag_exchange = rag_ex
    print("[stt] RabbitMQ ready.")


@app.on_event("shutdown")
async def shutdown():
    if rmq_connection and not rmq_connection.is_closed:
        await rmq_connection.close()
        print("[stt] RabbitMQ connection closed.")


@app.websocket("/ws/stt")
async def ws_stt(ws: WebSocket, forward_transcription: bool = Query(True)):
    await ws.accept()

    connection_id = uuid.uuid4().hex[:16]

    await ws.send_json({
        "type": "ready",
        "sample_rate": SAMPLE_RATE,
        "frame_ms": FRAME_MS,
        "frame_bytes": FRAME_BYTES,
        "silence_cutoff_ms": SILENCE_CUTOFF_MS,
        "connection_id": connection_id,
    })

    # Per-connection reply queue
    reply_channel = await rmq_connection.channel()
    reply_queue = await reply_channel.declare_queue(
        f"rag.replies.{connection_id}",
        exclusive=True,
        auto_delete=True,
    )

    speech_buf = bytearray()
    in_speech = False
    last_voice_ts = time.time()
    seq = 0

    # recv stats
    recv_count = 0
    last_log = time.time()

    # Background task: consume replies and send to client
    reply_task = None
    ws_closed = False

    async def consume_replies():
        try:
            async with reply_queue.iterator() as queue_iter:
                async for message in queue_iter:
                    async with message.process():
                        if ws_closed:
                            break
                        try:
                            data = json.loads(message.body.decode())
                            await ws.send_json({
                                "type": "answer",
                                "seq": data.get("seq"),
                                "query": data.get("query", ""),
                                "response": data.get("response", ""),
                            })
                        except Exception as e:
                            print(f"[stt] reply send error: {e}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            if not ws_closed:
                print(f"[stt] reply consumer error: {e}")

    reply_task = asyncio.create_task(consume_replies())

    try:
        while True:
            frame = await ws.receive_bytes()

            # stats: frames/sec
            recv_count += 1
            now = time.time()
            if now - last_log >= 1.0:
                print(f"[server] recv {recv_count} frames/sec, last={len(frame)} bytes")
                recv_count = 0
                last_log = now

            # normalize frame length
            if len(frame) < FRAME_BYTES:
                frame = frame + b"\x00" * (FRAME_BYTES - len(frame))
            elif len(frame) > FRAME_BYTES:
                frame = frame[:FRAME_BYTES]

            is_voice = vad.is_speech(frame, SAMPLE_RATE)

            if is_voice:
                last_voice_ts = now
                if not in_speech:
                    in_speech = True
                    speech_buf.clear()
                    await ws.send_json({"type": "speech_start"})
                speech_buf.extend(frame)
            else:
                if in_speech:
                    silence_ms = (now - last_voice_ts) * 1000.0
                    if silence_ms >= SILENCE_CUTOFF_MS:
                        in_speech = False
                        await ws.send_json({"type": "speech_end"})

                        utter_ms = len(speech_buf) / (SAMPLE_RATE * SAMPLE_WIDTH) * 1000.0
                        if utter_ms < MIN_UTTERANCE_MS:
                            speech_buf.clear()
                            await ws.send_json({"type": "final", "text": "", "reason": "too_short"})
                            continue

                        pcm = bytes(speech_buf)
                        speech_buf.clear()

                        # transcription with protection:
                        # 1) run in executor
                        # 2) add timeout so it won't hang forever
                        loop = asyncio.get_running_loop()
                        try:
                            text = await asyncio.wait_for(
                                loop.run_in_executor(None, transcribe_blocking, pcm),
                                timeout=30.0
                            )

                            seq += 1

                            if forward_transcription:
                                await ws.send_json({"type": "final", "text": text, "seq": seq})

                            # Publish to RabbitMQ for RAG processing
                            if text.strip() and rag_exchange:
                                await rag_exchange.publish(
                                    aio_pika.Message(
                                        body=json.dumps({
                                            "text": text,
                                            "connection_id": connection_id,
                                            "seq": seq,
                                        }).encode(),
                                        delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                                    ),
                                    routing_key="queries",
                                )

                        except asyncio.TimeoutError:
                            await ws.send_json({"type": "error", "stage": "transcribe", "message": "timeout"})
                        except Exception as e:
                            # IMPORTANT: don't crash the websocket handler
                            await ws.send_json({"type": "error", "stage": "transcribe", "message": repr(e)})

    except WebSocketDisconnect:
        print("[server] client disconnected")
    except Exception as e:
        # last-resort catch: still try to tell client
        try:
            await ws.send_json({"type": "error", "stage": "server", "message": repr(e)})
        except Exception:
            pass
        print("[server] fatal:", repr(e))
    finally:
        ws_closed = True
        if reply_task:
            reply_task.cancel()
            try:
                await reply_task
            except asyncio.CancelledError:
                pass
        try:
            await reply_channel.close()
        except Exception:
            pass
