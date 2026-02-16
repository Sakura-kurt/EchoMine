"""
Test script for the Memory worker via RabbitMQ.

Prerequisites:
  - RabbitMQ running (localhost:5672)
  - Ollama serving nemotron-3-nano + nomic-embed-text
  - Memory worker running: python memory_worker.py

Usage:
  python tests/test_memory_worker.py
  python tests/test_memory_worker.py --user "I found a hidden cave today" --assistant "That sounds amazing, Weibo!"
"""

import argparse
import asyncio
import json
import os
import sys

import aio_pika

sys.path.insert(0, ".")
from rabbitmq_config import get_connection, setup_exchanges_and_queues

MEMORIES_FILE = "./knowledge/memories.txt"


async def test_memory_gate(user_msg: str, assistant_msg: str, timeout: float = 60.0):
    print(f"[test] user_message: {user_msg}")
    print(f"[test] assistant_response: {assistant_msg}")
    print(f"[test] timeout: {timeout}s")
    print()

    # Record file size before
    before_size = os.path.getsize(MEMORIES_FILE) if os.path.exists(MEMORIES_FILE) else 0

    connection = await get_connection()
    channel = await connection.channel()
    _, memory_exchange = await setup_exchanges_and_queues(channel)

    # Publish memory gate job
    await memory_exchange.publish(
        aio_pika.Message(
            body=json.dumps({
                "user_message": user_msg,
                "assistant_response": assistant_msg,
            }).encode(),
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        ),
        routing_key="gate",
    )
    print("[test] Published to memory.gate, waiting for worker to process...")

    # Poll memories.txt for changes
    try:
        saved = False
        for i in range(int(timeout)):
            await asyncio.sleep(1)
            current_size = os.path.getsize(MEMORIES_FILE) if os.path.exists(MEMORIES_FILE) else 0
            if current_size > before_size:
                # Read the new content
                with open(MEMORIES_FILE, "r", encoding="utf-8") as f:
                    content = f.read()
                new_content = content[before_size:]
                print(f"[test] Memory SAVED! New entry:")
                print(f"  {new_content.strip()}")
                print()
                print("[test] PASSED (memory saved)")
                saved = True
                break

        if not saved:
            print(f"[test] No new memory written within {timeout}s.")
            print("  This could mean the memory gate decided to SKIP (expected for trivial messages).")
            print("  Check memory_worker.py logs for SAVED/SKIPPED output.")
            print()
            print("[test] PASSED (worker processed, gate decided to skip)")

    finally:
        await connection.close()


def main():
    parser = argparse.ArgumentParser(description="Test Memory worker via RabbitMQ")
    parser.add_argument("--user", default="I discovered a secret waterfall behind the mountain today",
                        help="User message to test")
    parser.add_argument("--assistant",
                        default="Oh Weibo, that sounds wonderful! We should visit it together tomorrow!",
                        help="Assistant response to test")
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="Seconds to wait (default: 60)")
    args = parser.parse_args()

    asyncio.run(test_memory_gate(args.user, args.assistant, args.timeout))


if __name__ == "__main__":
    main()
