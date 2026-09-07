"""Composition root: what the process is made of and how it stops."""

from __future__ import annotations

import logging
import signal
import threading
from contextlib import ExitStack

from kb_indexer import SERVICE_NAME, SERVICE_VERSION
from kb_indexer.config import Settings
from kb_indexer.consumer import Consumer, Policy
from kb_indexer.indexing import IndexingHandler
from kb_indexer.store import connect
from kb_indexer.telemetry import telemetry

log = logging.getLogger(__name__)


def run(settings: Settings) -> int:
    """Run until a termination signal.

    The stack is the wiring: what is entered first may be used by everything
    entered after it, and is left last. A failure while entering unwinds what
    is already running.
    """
    stopping = threading.Event()
    _install_signals(stopping)

    with ExitStack() as stack:
        stack.enter_context(telemetry(SERVICE_NAME, SERVICE_VERSION, export_logs=settings.log_otel))
        store = stack.enter_context(connect(settings.postgres_dsn))

        consumer = Consumer(
            url=settings.pubsub_url,
            handler=IndexingHandler(store),
            stopping=stopping,
            policy=Policy(
                retries=settings.consumer_retries,
                retry_backoff=settings.consumer_retry_backoff,
                shutdown_timeout=settings.consumer_shutdown_timeout,
            ),
        )

        log.info("indexer started", extra={"version": SERVICE_VERSION})
        consumer.run()
        log.info("indexer stopping")

    log.info("indexer stopped")

    return 0


def _install_signals(stopping: threading.Event) -> None:
    """Turn SIGINT and SIGTERM into an ordered shutdown."""

    def handle(signum: int, _frame: object) -> None:
        log.info("signal received", extra={"signal": signal.Signals(signum).name})
        stopping.set()

    for received in (signal.SIGINT, signal.SIGTERM):
        signal.signal(received, handle)
