import asyncio
import json
import os

import aio_pika
from langchain_ollama import ChatOllama, OllamaEmbeddings

from rag_pipeline import (
    load_vectorstore, create_vectorstore, load_documents,
    memory_gate, add_memory,
    KNOWLEDGE_DIR, CHROMA_DIR,
)
from rabbitmq_config import (
    get_connection, setup_exchanges_and_queues,
    MEMORY_GATE_QUEUE,
    MAX_RETRIES, get_retry_count,
)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


async def main():
    # Init LLM + vectorstore
    print("[memory-worker] Loading LLM and embeddings...")
    llm = ChatOllama(model="nemotron-3-nano", base_url=OLLAMA_HOST)
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_HOST)

    print("[memory-worker] Loading vector store...")
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        vectorstore = load_vectorstore(embeddings, CHROMA_DIR)
    else:
        documents = load_documents(KNOWLEDGE_DIR)
        if documents:
            vectorstore = create_vectorstore(documents, embeddings, CHROMA_DIR)
        else:
            print("[memory-worker] WARNING: No knowledge documents found!")
            return

    print("[memory-worker] Ready.")

    # Connect to RabbitMQ
    print("[memory-worker] Connecting to RabbitMQ...")
    connection = await get_connection()
    channel = await connection.channel()
    await channel.set_qos(prefetch_count=1)

    await setup_exchanges_and_queues(channel)

    queue = await channel.get_queue(MEMORY_GATE_QUEUE)

    loop = asyncio.get_running_loop()

    async def on_message(message: aio_pika.abc.AbstractIncomingMessage):
        retries = get_retry_count(message)
        if retries >= MAX_RETRIES:
            print(f"[memory-worker] Message exceeded {MAX_RETRIES} retries, discarding.")
            await message.ack()
            return

        try:
            body = json.loads(message.body.decode())
            user_msg = body["user_message"]
            assistant_msg = body["assistant_response"]

            should_save, summary = await loop.run_in_executor(
                None, memory_gate, llm, user_msg, assistant_msg,
            )

            if should_save:
                await loop.run_in_executor(
                    None, add_memory, vectorstore, summary,
                )
                print(f"[memory-worker] SAVED: {summary[:60]}")
            else:
                print(f"[memory-worker] SKIPPED: {summary[:60]}")

            await message.ack()

        except Exception as e:
            print(f"[memory-worker] Error (retry {retries + 1}/{MAX_RETRIES}): {e}")
            await message.nack(requeue=False)

    await queue.consume(on_message)

    print("[memory-worker] Consuming from memory.gate. Press Ctrl+C to stop.")
    try:
        await asyncio.Future()  # run forever
    except asyncio.CancelledError:
        pass
    finally:
        await connection.close()
        print("[memory-worker] Shutdown.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[memory-worker] Stopped.")
