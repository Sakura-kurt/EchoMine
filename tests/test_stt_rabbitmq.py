"""
Test the full STT → RabbitMQ → RAG → reply flow.

Connects directly to the STT server via WebSocket, sends synthetic speech
audio (a spoken-like tone burst to trigger VAD), and verifies the message
types received including the RAG answer.

Prerequisites:
  - RabbitMQ running (localhost:5672)
  - Ollama serving nemotron-3-nano + nomic-embed-text
  - STT server running: uvicorn stt_server:app --port 8001
  - RAG worker running: python rag_worker.py

Usage:
  python tests/test_stt_rabbitmq.py
  python tests/test_stt_rabbitmq.py --stt-url ws://localhost:8001/ws/stt
"""

import argparse
import asyncio
import json
import struct
import sys

import numpy as np
import websockets

SAMPLE_RATE = 16000
FRAME_MS = 20
SAMPLES_PER_FRAME = int(SAMPLE_RATE * FRAME_MS / 1000)  # 320
FRAME_BYTES = SAMPLES_PER_FRAME * 2  # int16 = 2 bytes


def generate_speech_frames(duration_s: float = 1.5) -> list[bytes]:
    """Generate frames of a 400Hz tone that triggers VAD as speech."""
    total_samples = int(SAMPLE_RATE * duration_s)
    t = np.arange(total_samples) / SAMPLE_RATE
    # Mix of frequencies to sound more speech-like to VAD
    signal = (
        0.4 * np.sin(2 * np.pi * 300 * t) +
        0.3 * np.sin(2 * np.pi * 600 * t) +
        0.2 * np.sin(2 * np.pi * 1200 * t) +
        0.1 * np.random.randn(total_samples)  # noise
    )
    pcm16 = (np.clip(signal, -1.0, 1.0) * 16000).astype(np.int16)

    frames = []
    for i in range(0, len(pcm16), SAMPLES_PER_FRAME):
        chunk = pcm16[i:i + SAMPLES_PER_FRAME]
        if len(chunk) < SAMPLES_PER_FRAME:
            chunk = np.pad(chunk, (0, SAMPLES_PER_FRAME - len(chunk)))
        frames.append(chunk.tobytes())
    return frames


def generate_silence_frames(duration_s: float = 1.0) -> list[bytes]:
    """Generate silent frames to trigger speech_end."""
    total_samples = int(SAMPLE_RATE * duration_s)
    pcm16 = np.zeros(total_samples, dtype=np.int16)
    frames = []
    for i in range(0, len(pcm16), SAMPLES_PER_FRAME):
        chunk = pcm16[i:i + SAMPLES_PER_FRAME]
        if len(chunk) < SAMPLES_PER_FRAME:
            chunk = np.pad(chunk, (0, SAMPLES_PER_FRAME - len(chunk)))
        frames.append(chunk.tobytes())
    return frames


async def test_stt_flow(stt_url: str, timeout: float = 120.0):
    print(f"[test] Connecting to STT server: {stt_url}")
    print(f"[test] timeout: {timeout}s")
    print()

    seen_types = set()
    results = {}

    async with websockets.connect(stt_url, max_size=None) as ws:
        # 1. Wait for ready
        ready_msg = json.loads(await ws.recv())
        assert ready_msg["type"] == "ready", f"Expected 'ready', got {ready_msg}"
        connection_id = ready_msg.get("connection_id", "")
        print(f"[test] Ready! connection_id={connection_id}")
        seen_types.add("ready")

        # 2. Send speech frames
        speech_frames = generate_speech_frames(duration_s=1.5)
        print(f"[test] Sending {len(speech_frames)} speech frames (~1.5s)...")
        for frame in speech_frames:
            await ws.send(frame)
            await asyncio.sleep(FRAME_MS / 1000.0)

        # 3. Send silence to trigger speech_end
        silence_frames = generate_silence_frames(duration_s=1.5)
        print(f"[test] Sending {len(silence_frames)} silence frames (~1.5s)...")
        for frame in silence_frames:
            await ws.send(frame)
            await asyncio.sleep(FRAME_MS / 1000.0)

        # 4. Collect messages (wait for answer)
        print(f"[test] Waiting for messages (up to {timeout}s)...")
        try:
            deadline = asyncio.get_event_loop().time() + timeout
            while asyncio.get_event_loop().time() < deadline:
                remaining = deadline - asyncio.get_event_loop().time()
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=remaining))
                msg_type = msg.get("type", "")
                seen_types.add(msg_type)

                if msg_type == "speech_start":
                    print("[test] -> speech_start")
                elif msg_type == "speech_end":
                    print("[test] -> speech_end")
                elif msg_type == "final":
                    text = msg.get("text", "")
                    seq = msg.get("seq", "")
                    print(f"[test] -> final (seq={seq}): {text}")
                    results["final"] = msg
                elif msg_type == "answer":
                    query = msg.get("query", "")
                    response = msg.get("response", "")
                    seq = msg.get("seq", "")
                    print(f"[test] -> answer (seq={seq}):")
                    print(f"         Q: {query}")
                    print(f"         A: {response[:120]}...")
                    results["answer"] = msg
                    break  # got what we need
                elif msg_type == "error":
                    print(f"[test] -> error: {msg}")
                else:
                    print(f"[test] -> {msg_type}: {msg}")

        except asyncio.TimeoutError:
            pass

    # Report
    print()
    print("=" * 50)
    print(f"[test] Message types seen: {sorted(seen_types)}")

    passed = True
    for expected in ["ready", "speech_start", "speech_end", "final"]:
        if expected in seen_types:
            print(f"  [OK] {expected}")
        else:
            print(f"  [MISSING] {expected}")
            passed = False

    if "answer" in seen_types:
        print(f"  [OK] answer (RAG reply via RabbitMQ)")
    else:
        print(f"  [MISSING] answer — RAG worker may not be running")
        passed = False

    print()
    print(f"[test] {'PASSED' if passed else 'FAILED'}")


def main():
    parser = argparse.ArgumentParser(description="Test STT → RabbitMQ → RAG flow")
    parser.add_argument("--stt-url", default="ws://localhost:8001/ws/stt",
                        help="STT server WebSocket URL")
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="Seconds to wait for answer (default: 120)")
    args = parser.parse_args()

    asyncio.run(test_stt_flow(args.stt_url, args.timeout))


if __name__ == "__main__":
    main()
