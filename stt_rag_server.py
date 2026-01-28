import asyncio
import time
from typing import List, Dict

import numpy as np
import webrtcvad
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
from langchain_ollama import ChatOllama, OllamaEmbeddings

from rag_pipeline import (
    load_vectorstore, create_vectorstore, load_documents,
    create_qa_chain, memory_gate, add_memory,
    KNOWLEDGE_DIR, CHROMA_DIR,
)
import os

app = FastAPI(title="STT + RAG Server (Sakura)")

# ===== Audio settings (MUST match client) =====
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
FRAME_MS = 20
FRAME_BYTES = int(SAMPLE_RATE * FRAME_MS / 1000.0 * SAMPLE_WIDTH)  # 640

# ===== VAD settings =====
vad = webrtcvad.Vad(2)
SILENCE_CUTOFF_MS = 700
MIN_UTTERANCE_MS = 250

# ===== Models (loaded at startup) =====
whisper_model = None
llm = None
vectorstore = None
qa_chain = None


@app.on_event("startup")
async def startup():
    global whisper_model, llm, vectorstore, qa_chain

    print("[startup] Loading Whisper model...")
    whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")

    print("[startup] Loading LLM and embeddings...")
    llm = ChatOllama(model="nemotron-3-nano")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    print("[startup] Loading vector store...")
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        vectorstore = load_vectorstore(embeddings, CHROMA_DIR)
    else:
        documents = load_documents(KNOWLEDGE_DIR)
        if documents:
            vectorstore = create_vectorstore(documents, embeddings, CHROMA_DIR)
        else:
            print("[startup] WARNING: No knowledge documents found!")
            return

    qa_chain = create_qa_chain(llm, vectorstore)
    print("[startup] Ready.")


def transcribe_blocking(pcm: bytes) -> str:
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    segments, _info = whisper_model.transcribe(audio, language="en", vad_filter=False)
    return "".join(seg.text for seg in segments).strip()


def rag_query_blocking(query: str) -> str:
    result = qa_chain.invoke({"query": query})
    return result["result"]


def memory_gate_blocking(user_msg: str, assistant_msg: str):
    should_save, summary = memory_gate(llm, user_msg, assistant_msg)
    if should_save:
        add_memory(vectorstore, summary)
        print(f"[memory] SAVED: {summary[:60]}")
    else:
        print(f"[memory] SKIPPED: {summary[:60]}")


@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket):
    await ws.accept()
    await ws.send_json({
        "type": "ready",
        "sample_rate": SAMPLE_RATE,
        "frame_ms": FRAME_MS,
        "frame_bytes": FRAME_BYTES,
        "silence_cutoff_ms": SILENCE_CUTOFF_MS,
    })

    speech_buf = bytearray()
    in_speech = False
    last_voice_ts = time.time()
    recv_count = 0
    last_log = time.time()
    loop = asyncio.get_running_loop()

    try:
        while True:
            frame = await ws.receive_bytes()

            recv_count += 1
            now = time.time()
            if now - last_log >= 1.0:
                print(f"[server] recv {recv_count} frames/sec")
                recv_count = 0
                last_log = now

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

                        # Step 1: Transcribe
                        try:
                            text = await asyncio.wait_for(
                                loop.run_in_executor(None, transcribe_blocking, pcm),
                                timeout=30.0,
                            )
                        except asyncio.TimeoutError:
                            await ws.send_json({"type": "error", "stage": "transcribe", "message": "timeout"})
                            continue
                        except Exception as e:
                            await ws.send_json({"type": "error", "stage": "transcribe", "message": repr(e)})
                            continue

                        if not text:
                            await ws.send_json({"type": "final", "text": "", "reason": "empty"})
                            continue

                        await ws.send_json({"type": "transcription", "text": text})
                        print(f"[stt] \"{text}\"")

                        # Step 2: RAG query
                        try:
                            response = await asyncio.wait_for(
                                loop.run_in_executor(None, rag_query_blocking, text),
                                timeout=60.0,
                            )
                            await ws.send_json({"type": "answer", "query": text, "response": response})
                            print(f"[rag] \"{response[:80]}\"")
                        except asyncio.TimeoutError:
                            await ws.send_json({"type": "error", "stage": "rag", "message": "timeout"})
                            continue
                        except Exception as e:
                            await ws.send_json({"type": "error", "stage": "rag", "message": repr(e)})
                            continue

                        # Step 3: Memory gate (fire and forget)
                        loop.run_in_executor(None, memory_gate_blocking, text, response)

    except WebSocketDisconnect:
        print("[server] client disconnected")
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "stage": "server", "message": repr(e)})
        except Exception:
            pass
        print("[server] fatal:", repr(e))
