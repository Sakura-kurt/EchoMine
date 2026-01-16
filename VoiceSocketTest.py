
import asyncio
import websockets

async def main():
    uri = "ws://127.0.0.1:8000/ws/stt"
    async with websockets.connect(uri, max_size=None) as ws:
        msg = await ws.recv()
        print("server says:", msg)

asyncio.run(main())