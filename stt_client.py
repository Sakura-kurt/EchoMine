import asyncio
import sounddevice as sd
import numpy as np
import websockets

SAMPLE_RATE = 16000
FRAME_MS = 20
SAMPLES_PER_FRAME = int(SAMPLE_RATE * FRAME_MS / 1000)

def float_to_pcm16(x: np.ndarray) -> bytes:
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767).astype(np.int16).tobytes()

async def main():
    uri = "ws://127.0.0.1:8000/ws/stt"
    async with websockets.connect(uri, max_size=None) as ws:
        print("connected")
        print("server:", await ws.recv())  # ready

        q: asyncio.Queue[bytes] = asyncio.Queue()
        loop = asyncio.get_running_loop()  # ✅ 获取主线程 event loop

        printed = False

        def callback(indata, frames, time_info, status):
            nonlocal printed
            # indata: float32 [-1, 1], shape (frames, 1)
            pcm = float_to_pcm16(indata[:, 0])

            # 只打印一次，用于确认 20ms/帧 => 320 samples => 640 bytes
            if not printed:
                print("audio callback frames:", frames, "pcm_bytes:", len(pcm))
                printed = True

            # ✅ 用主线程 loop，把数据线程安全地塞进 asyncio 队列
            loop.call_soon_threadsafe(q.put_nowait, pcm)

        async def sender():
            count = 0
            t0 = asyncio.get_event_loop().time()
            while True:
                frame = await q.get()
                await ws.send(frame)
                count += 1
                now = asyncio.get_event_loop().time()
                if now - t0 >= 1.0:
                    print(f"[client] sent {count} frames/sec, last={len(frame)} bytes")
                    count = 0
                    t0 = now

        async def receiver():
            while True:
                msg = await ws.recv()
                print("recv:", msg)

        # blocksize 必须匹配 20ms 帧，否则 frame_bytes 不对
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=SAMPLES_PER_FRAME,
            callback=callback
        ):
            await asyncio.gather(sender(), receiver())

if __name__ == "__main__":
    asyncio.run(main())