import asyncio
import json
import os

import aio_pika
from langchain_ollama import ChatOllama, OllamaEmbeddings

from rag_pipeline import (
    load_vectorstore, create_vectorstore, load_documents,
    create_qa_chain,
    KNOWLEDGE_DIR, CHROMA_DIR,
)
from rabbitmq_config import (
    get_connection, setup_exchanges_and_queues,
    RAG_QUERIES_QUEUE, MEMORY_EXCHANGE,
    MAX_RETRIES, get_retry_count,
)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


async def main():
    # Init LLM + vectorstore
    print("[rag-worker] Loading LLM and embeddings...")
    llm = ChatOllama(model="nemotron-3-nano", base_url=OLLAMA_HOST)
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_HOST)

    print("[rag-worker] Loading vector store...")
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        vectorstore = load_vectorstore(embeddings, CHROMA_DIR)
    else:
        documents = load_documents(KNOWLEDGE_DIR)
        if documents:
            vectorstore = create_vectorstore(documents, embeddings, CHROMA_DIR)
        else:
            print("[rag-worker] WARNING: No knowledge documents found!")
            return

    qa_chain = create_qa_chain(llm, vectorstore)
    print("[rag-worker] QA chain ready.")

    # Connect to RabbitMQ
    print("[rag-worker] Connecting to RabbitMQ...")
    connection = await get_connection()
    channel = await connection.channel()
    await channel.set_qos(prefetch_count=1)

    rag_exchange, memory_exchange = await setup_exchanges_and_queues(channel)

    queue = await channel.get_queue(RAG_QUERIES_QUEUE)

    loop = asyncio.get_running_loop()

    async def on_message(message: aio_pika.abc.AbstractIncomingMessage):
        retries = get_retry_count(message)
        if retries >= MAX_RETRIES:
            print(f"[rag-worker] Message exceeded {MAX_RETRIES} retries, discarding.")
            await message.ack()
            return

        try:
            body = json.loads(message.body.decode())
            text = body["text"]
            connection_id = body["connection_id"]
            seq = body["seq"]

            print(f"[rag-worker] Processing seq={seq}: {text[:60]}...")

            result = await loop.run_in_executor(
                None, lambda: qa_chain.invoke({"query": text})
            )
            response_text = result["result"]

            # Publish reply to the connection-specific reply queue
            await channel.default_exchange.publish(
                aio_pika.Message(
                    body=json.dumps({
                        "query": text,
                        "response": response_text,
                        "seq": seq,
                    }).encode(),
                ),
                routing_key=f"rag.replies.{connection_id}",
            )

            # Publish memory gate job
            await memory_exchange.publish(
                aio_pika.Message(
                    body=json.dumps({
                        "user_message": text,
                        "assistant_response": response_text,
                    }).encode(),
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                ),
                routing_key="gate",
            )

            await message.ack()
            print(f"[rag-worker] seq={seq} done, reply sent.")

        except Exception as e:
            print(f"[rag-worker] Error (retry {retries + 1}/{MAX_RETRIES}): {e}")
            await message.nack(requeue=False)

    await queue.consume(on_message)

    print("[rag-worker] Consuming from rag.queries. Press Ctrl+C to stop.")
    try:
        await asyncio.Future()  # run forever
    except asyncio.CancelledError:
        pass
    finally:
        await connection.close()
        print("[rag-worker] Shutdown.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[rag-worker] Stopped.")
