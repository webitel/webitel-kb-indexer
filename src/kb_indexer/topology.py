"""Broker objects of the re-indexing contract.

Both sides declare every object with exactly these parameters: a redeclare with
different properties fails the channel, not just the call.
"""

from __future__ import annotations

from typing import Any

# The indexing exchange and the queue bound to it deliberately share a name.
# Publish to the exchange; the default exchange with this routing key bypasses
# it and reaches nothing.
REINDEX_EXCHANGE = "kb.reindex"
REINDEX_EXCHANGE_TYPE = "topic"
REINDEX_QUEUE = "kb.reindex"
REINDEX_BINDING = "#"

# Where a delivery this worker gave up on ends.
REINDEX_DLX = "kb.reindex.dlx"
REINDEX_DLX_TYPE = "fanout"
REINDEX_DLQ = "kb.reindex.dlq"
REINDEX_DLQ_BINDING = ""

CONTENT_TYPE = "application/json"

# The outbox id of the message, for tracing. Never a deduplication key.
MESSAGE_ID_HEADER = "x-message-id"

# One delivery at a time: this is what keeps per-article order in v1.
PREFETCH = 1


def queue_arguments() -> dict[str, str]:
    """Declare arguments of the indexing queue."""
    return {"x-dead-letter-exchange": REINDEX_DLX}


def declare(channel: Any) -> None:
    """Create every object of the contract. Idempotent, and run on every connect."""
    channel.exchange_declare(
        REINDEX_EXCHANGE,
        exchange_type=REINDEX_EXCHANGE_TYPE,
        durable=True,
        auto_delete=False,
        internal=False,
    )
    channel.exchange_declare(
        REINDEX_DLX,
        exchange_type=REINDEX_DLX_TYPE,
        durable=True,
        auto_delete=False,
        internal=False,
    )
    channel.queue_declare(
        REINDEX_QUEUE,
        durable=True,
        exclusive=False,
        auto_delete=False,
        arguments=queue_arguments(),
    )
    channel.queue_bind(REINDEX_QUEUE, REINDEX_EXCHANGE, routing_key=REINDEX_BINDING)
    channel.queue_declare(REINDEX_DLQ, durable=True, exclusive=False, auto_delete=False)
    channel.queue_bind(REINDEX_DLQ, REINDEX_DLX, routing_key=REINDEX_DLQ_BINDING)
