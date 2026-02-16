"""
Test script for the RAG worker via RabbitMQ.

Prerequisites:
  - RabbitMQ running (localhost:5672)
  - Ollama serving nemotron-3-nano + nomic-embed-text
  - RAG worker running: python rag_worker.py

Usage:
  python tests/test_rag_worker.py
  python tests/test_rag_worker.py "Who is Sakura?"
  python tests/test_rag_worker.py --timeout 30 "Tell me about the island"
"""

import argparse
import asyncio
import json
import sys
import uuid

import aio_pika

sys.path.insert(0, ".")
from rabbitmq_config import (
    get_connection, setup_exchanges_and_queues,
    RAG_EXCHANGE,
)


async def test_query(query: str, timeout: float = 60.0):
    connection_id = uuid.uuid4().hex[:16]
    print(f"[test] connection_id: {connection_id}")
    print(f"[test] query: {query}")
    print(f"[test] timeout: {timeout}s")
    print()

    connection = await get_connection()
    channel = await connection.channel()
    rag_exchange, _ = await setup_exchanges_and_queues(channel)

    # Create exclusive reply queue
    reply_queue = await channel.declare_queue(
        f"rag.replies.{connection_id}",
        exclusive=True,
        auto_delete=True,
    )

    result_future: asyncio.Future = asyncio.get_running_loop().create_future()

    async def on_reply(message: aio_pika.abc.AbstractIncomingMessage):
        async with message.process():
            data = json.loads(message.body.decode())
            if not result_future.done():
                result_future.set_result(data)

    await reply_queue.consume(on_reply)

    # Publish query
    await rag_exchange.publish(
        aio_pika.Message(
            body=json.dumps({
                "text": query,
                "connection_id": connection_id,
                "seq": 1,
            }).encode(),
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        ),
        routing_key="queries",
    )
    print("[test] Published to rag.queries, waiting for reply...")

    try:
        data = await asyncio.wait_for(result_future, timeout=timeout)
        print(f"[test] Got reply (seq={data.get('seq')}):")
        print(f"  Q: {data.get('query', '')}")
        print(f"  A: {data.get('response', '')}")
        print()
        print("[test] PASSED")
    except asyncio.TimeoutError:
        print(f"[test] FAILED — no reply within {timeout}s")
        print("  Check that rag_worker.py is running and Ollama is serving.")
    finally:
        await connection.close()


def main():
    parser = argparse.ArgumentParser(description="Test RAG worker via RabbitMQ")
    parser.add_argument("query", nargs="?", default="Who is Sakura?",
                        help="Query to send (default: 'Who is Sakura?')")
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="Seconds to wait for reply (default: 60)")
    args = parser.parse_args()

    asyncio.run(test_query(args.query, args.timeout))


if __name__ == "__main__":
    main()
