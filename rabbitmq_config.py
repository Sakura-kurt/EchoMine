import os

import aio_pika

RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")

# Exchange names
RAG_EXCHANGE = "rag"
MEMORY_EXCHANGE = "memory"
DLX_EXCHANGE = "dlx"

# Queue names
RAG_QUERIES_QUEUE = "rag.queries"
MEMORY_GATE_QUEUE = "memory.gate"

# DLX queue names (messages wait here before retry)
RAG_QUERIES_DLX_QUEUE = "rag.queries.dlx"
MEMORY_GATE_DLX_QUEUE = "memory.gate.dlx"

# DLX settings
DLX_MESSAGE_TTL = 5000  # 5 seconds in milliseconds
MAX_RETRIES = 3


async def get_connection() -> aio_pika.abc.AbstractRobustConnection:
    return await aio_pika.connect_robust(RABBITMQ_URL)


async def setup_exchanges_and_queues(channel: aio_pika.abc.AbstractChannel):
    """Declare all exchanges and queues (idempotent)."""

    # Main exchanges
    rag_exchange = await channel.declare_exchange(
        RAG_EXCHANGE, aio_pika.ExchangeType.TOPIC, durable=True,
    )
    memory_exchange = await channel.declare_exchange(
        MEMORY_EXCHANGE, aio_pika.ExchangeType.TOPIC, durable=True,
    )
    dlx_exchange = await channel.declare_exchange(
        DLX_EXCHANGE, aio_pika.ExchangeType.TOPIC, durable=True,
    )

    # --- RAG queues ---

    # Main rag.queries queue (with DLX arguments)
    rag_queries = await channel.declare_queue(
        RAG_QUERIES_QUEUE,
        durable=True,
        arguments={
            "x-dead-letter-exchange": DLX_EXCHANGE,
            "x-dead-letter-routing-key": "rag.queries.dlx",
        },
    )
    await rag_queries.bind(rag_exchange, routing_key="queries")

    # DLX queue for rag.queries (TTL → re-route back to main queue)
    rag_queries_dlx = await channel.declare_queue(
        RAG_QUERIES_DLX_QUEUE,
        durable=True,
        arguments={
            "x-message-ttl": DLX_MESSAGE_TTL,
            "x-dead-letter-exchange": RAG_EXCHANGE,
            "x-dead-letter-routing-key": "queries",
        },
    )
    await rag_queries_dlx.bind(dlx_exchange, routing_key="rag.queries.dlx")

    # --- Memory queues ---

    memory_gate = await channel.declare_queue(
        MEMORY_GATE_QUEUE,
        durable=True,
        arguments={
            "x-dead-letter-exchange": DLX_EXCHANGE,
            "x-dead-letter-routing-key": "memory.gate.dlx",
        },
    )
    await memory_gate.bind(memory_exchange, routing_key="gate")

    memory_gate_dlx = await channel.declare_queue(
        MEMORY_GATE_DLX_QUEUE,
        durable=True,
        arguments={
            "x-message-ttl": DLX_MESSAGE_TTL,
            "x-dead-letter-exchange": MEMORY_EXCHANGE,
            "x-dead-letter-routing-key": "gate",
        },
    )
    await memory_gate_dlx.bind(dlx_exchange, routing_key="memory.gate.dlx")

    return rag_exchange, memory_exchange


def get_retry_count(message: aio_pika.abc.AbstractIncomingMessage) -> int:
    """Extract retry count from x-death header."""
    if not message.headers or "x-death" not in message.headers:
        return 0
    x_death = message.headers["x-death"]
    total = 0
    for entry in x_death:
        total += entry.get("count", 0)
    return total
