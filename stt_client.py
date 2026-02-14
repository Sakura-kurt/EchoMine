import argparse
import asyncio
import json
import sys

import httpx
import numpy as np
import sounddevice as sd
import websockets

SAMPLE_RATE = 16000
FRAME_MS = 20
SAMPLES_PER_FRAME = int(SAMPLE_RATE * FRAME_MS / 1000)


def float_to_pcm16(x: np.ndarray) -> bytes:
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767).astype(np.int16).tobytes()


class STTClient:
    def __init__(self, server: str, username: str, password: str):
        self.server = server.rstrip("/")
        self.username = username
        self.password = password
        self.token: str | None = None

    async def register(self) -> bool:
        """Register a new user. Returns True if successful."""
        url = f"{self.server}/auth/register"
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, json={
                    "username": self.username,
                    "password": self.password,
                })
                if response.status_code == 200:
                    data = response.json()
                    print(f"[register] Success! user_id: {data['user_id']}")
                    return True
                elif response.status_code == 400:
                    print(f"[register] Username already exists")
                    return False
                else:
                    print(f"[register] Failed: {response.text}")
                    return False
            except Exception as e:
                print(f"[register] Error: {e}")
                return False

    async def login(self) -> bool:
        """Login and get JWT token. Returns True if successful."""
        url = f"{self.server}/auth/login"
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, json={
                    "username": self.username,
                    "password": self.password,
                })
                if response.status_code == 200:
                    data = response.json()
                    self.token = data["access_token"]
                    print(f"[login] Success! Token received.")
                    return True
                else:
                    print(f"[login] Failed: {response.text}")
                    return False
            except Exception as e:
                print(f"[login] Error: {e}")
                return False

    async def stream_audio(self):
        """Connect to WebSocket and stream audio."""
        if not self.token:
            print("[error] No token. Please login first.")
            return

        # Convert http:// to ws://
        ws_server = self.server.replace("http://", "ws://").replace("https://", "wss://")
        uri = f"{ws_server}/ws/stt?token={self.token}"

        print(f"[ws] Connecting to {ws_server}/ws/stt ...")

        try:
            async with websockets.connect(uri, max_size=None) as ws:
                print("[ws] Connected!")

                # Wait for ready message
                ready_msg = await ws.recv()
                print(f"[ws] Server ready: {ready_msg}")

                q: asyncio.Queue[bytes] = asyncio.Queue()
                loop = asyncio.get_running_loop()
                printed = False

                def callback(indata, frames, time_info, status):
                    nonlocal printed
                    pcm = float_to_pcm16(indata[:, 0])
                    if not printed:
                        print(f"[audio] callback frames: {frames}, pcm_bytes: {len(pcm)}")
                        printed = True
                    loop.call_soon_threadsafe(q.put_nowait, pcm)

                async def sender():
                    try:
                        count = 0
                        t0 = asyncio.get_event_loop().time()
                        while True:
                            frame = await q.get()
                            await ws.send(frame)
                            count += 1
                            now = asyncio.get_event_loop().time()
                            if now - t0 >= 1.0:
                                print(f"[client] sent {count} frames/sec")
                                count = 0
                                t0 = now
                    except asyncio.CancelledError:
                        pass

                async def receiver():
                    try:
                        while True:
                            msg = await ws.recv()
                            data = json.loads(msg)
                            msg_type = data.get("type", "")

                            if msg_type == "speech_start":
                                print("\n[speech] Started speaking...")
                            elif msg_type == "speech_end":
                                print("[speech] Stopped speaking, processing...")
                            elif msg_type == "final":
                                text = data.get("text", "")
                                print(f"\n>>> TRANSCRIPTION: {text}\n")
                            elif msg_type == "error":
                                print(f"[error] {data.get('stage')}: {data.get('message')}")
                            else:
                                print(f"[ws] {msg}")
                    except asyncio.CancelledError:
                        pass

                with sd.InputStream(
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype="float32",
                    blocksize=SAMPLES_PER_FRAME,
                    callback=callback
                ):
                    print("\n[ready] Speak now! Press Ctrl+C to stop.\n")
                    await asyncio.gather(sender(), receiver())

        except websockets.ConnectionClosed as e:
            print(f"[ws] Connection closed: {e}")
        except Exception as e:
            print(f"[ws] Error: {e}")


async def main():
    parser = argparse.ArgumentParser(description="STT Client with Gateway Authentication")
    parser.add_argument("--server", default="http://127.0.0.1:8000", help="Gateway server URL")
    parser.add_argument("--username", "-u", required=True, help="Username")
    parser.add_argument("--password", "-p", required=True, help="Password")
    parser.add_argument("--register", "-r", action="store_true", help="Register new user first")

    args = parser.parse_args()

    client = STTClient(args.server, args.username, args.password)

    # Register if requested
    if args.register:
        success = await client.register()
        if not success:
            print("[info] Continuing to login anyway...")

    # Login
    if not await client.login():
        print("[error] Login failed. Exiting.")
        sys.exit(1)

    # Stream audio
    await client.stream_audio()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nClient stopped.")
