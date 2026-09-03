"""The delivery loop: one message at a time, settled only after processing."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import pika
from pika.exceptions import AMQPError

from kb_indexer import topology
from kb_indexer.events import ArticleReindex, EnvelopeError, parse
from kb_indexer.handler import Handler, TransientError

log = logging.getLogger(__name__)

# How long the loop waits for broker traffic before it looks at the stop flag.
# It also bounds how long a heartbeat can go unanswered.
TICK = 1.0

# Delays between two connection attempts, and the age at which a session counts
# as healthy enough to start those delays over.
RECONNECT_DELAY = 1.0
RECONNECT_DELAY_MAX = 30.0
HEALTHY_SESSION = 30.0

# Ceiling of the delay between two attempts at the same message.
RETRY_DELAY_MAX = 30.0

# How long one turn of the shutdown waits for the delivery in flight.
DRAIN_TICK = 0.2


class Settlement(Enum):
    """What the loop does with a delivery once processing ends."""

    ACK = auto()
    DEAD_LETTER = auto()
    REDELIVER = auto()


@dataclass(frozen=True, slots=True)
class Policy:
    """How stubborn the worker is with one message.

    Without defaults on purpose: the configuration is the only place these
    numbers are chosen, so a default here could not drift away from it.
    """

    retries: int
    retry_backoff: float
    shutdown_timeout: float


def delays(base: float, ceiling: float) -> Iterator[float]:
    """Doubling delays, never past the ceiling."""
    delay = min(base, ceiling)

    while True:
        yield delay
        delay = min(delay * 2, ceiling)


@dataclass
class _Session:
    """One broker connection and the single delivery it is working on."""

    connection: Any
    channel: Any
    working: threading.Thread | None = None


class Consumer:
    """Consumes the re-indexing queue until asked to stop.

    Processing runs off the connection thread. One article can take longer than
    a heartbeat interval, and a connection thread blocked that long loses the
    connection and the work with it.
    """

    def __init__(
        self,
        url: str,
        handler: Handler,
        stopping: threading.Event,
        policy: Policy,
    ) -> None:
        self._parameters = pika.URLParameters(url)
        self._handler = handler
        self._stopping = stopping
        self._policy = policy
        self._session: _Session | None = None

    def run(self) -> None:
        """Consume, reconnecting for as long as the stop flag is down."""
        pauses = delays(RECONNECT_DELAY, RECONNECT_DELAY_MAX)

        while not self._stopping.is_set():
            lived = 0.0
            try:
                lived = self._serve()
            except AMQPError:
                log.exception("broker session ended")

            if self._stopping.is_set():
                break

            if lived >= HEALTHY_SESSION:
                pauses = delays(RECONNECT_DELAY, RECONNECT_DELAY_MAX)

            pause = next(pauses)
            log.warning("reconnecting to the broker", extra={"delay": pause})
            self._stopping.wait(pause)

    def _serve(self) -> float:
        """Hold one connection for as long as it and the process live."""
        started = time.monotonic()
        connection = pika.BlockingConnection(self._parameters)

        try:
            channel = connection.channel()
            topology.declare(channel)
            channel.basic_qos(prefetch_count=topology.PREFETCH)

            session = _Session(connection, channel)
            self._session = session

            channel.basic_consume(topology.REINDEX_QUEUE, on_message_callback=self._deliver)
            log.info("consuming", extra={"queue": topology.REINDEX_QUEUE, "prefetch": topology.PREFETCH})

            while not self._stopping.is_set() and connection.is_open:
                connection.process_data_events(time_limit=TICK)

            self._drain(session)
        finally:
            self._session = None
            _close(connection)

        return time.monotonic() - started

    def _deliver(self, _channel: Any, method: Any, properties: Any, body: bytes) -> None:
        """Hand the delivery to a worker thread. Runs on the connection thread."""
        session = self._session
        if session is None:  # the session ended between delivery and dispatch
            return

        session.working = threading.Thread(
            target=self._work,
            args=(session, method.delivery_tag, body, properties),
            name="indexing",
            daemon=True,
        )
        session.working.start()

    def _work(self, session: _Session, tag: int, body: bytes, properties: Any) -> None:
        """Process one delivery and schedule its settlement. Runs off the connection thread."""
        settlement = self._outcome(body, properties)

        try:
            session.connection.add_callback_threadsafe(lambda: self._settle(session, tag, settlement))
        except Exception:
            log.warning(
                "the delivery could not be settled and will be redelivered",
                extra={"delivery_tag": tag},
                exc_info=True,
            )

    def _outcome(self, body: bytes, properties: Any) -> Settlement:
        """Decide the fate of one delivery."""
        try:
            event = parse(body)
        except EnvelopeError as broken:
            # Nothing identifies the article, so nothing can be marked failed.
            log.error("envelope rejected", extra={"reason": str(broken), "message_id": message_id(properties)})

            return Settlement.DEAD_LETTER

        failure = self._index(event)
        if failure is None:
            return Settlement.ACK

        if self._stopping.is_set() and isinstance(failure, TransientError):
            log.warning("indexing abandoned at shutdown, the delivery returns to the queue", extra=event.as_fields())

            return Settlement.REDELIVER

        log.error("article will not be indexed from this event", extra=event.as_fields(), exc_info=failure)

        try:
            self._handler.give_up(event)
        except Exception:
            # Dead lettering now would leave the article indexing forever, with
            # nothing left to deliver it. It goes back to the queue instead.
            log.exception("the failure could not be recorded", extra=event.as_fields())

            return Settlement.REDELIVER

        return Settlement.DEAD_LETTER

    def _index(self, event: ArticleReindex) -> Exception | None:
        """Run the pipeline, retrying transient failures. Returns what defeated it."""
        pauses = delays(self._policy.retry_backoff, RETRY_DELAY_MAX)
        left = self._policy.retries

        while True:
            try:
                self._handler.handle(event)
            except TransientError as transient:
                if left <= 0 or self._stopping.is_set():
                    return transient

                left -= 1
                pause = next(pauses)
                log.warning(
                    "indexing failed, trying again",
                    extra={**event.as_fields(), "attempts_left": left + 1, "delay": pause, "reason": str(transient)},
                )
                # A stop must not start another attempt: indexing one article
                # can run for minutes, and that work would be thrown away.
                if self._stopping.wait(pause):
                    return transient
            except Exception as permanent:
                return permanent
            else:
                return None

    def _settle(self, session: _Session, tag: int, settlement: Settlement) -> None:
        """Tell the broker what happened. Runs on the connection thread."""
        if self._session is not session or not session.channel.is_open:
            log.warning("the delivery outlived its connection and will be redelivered", extra={"delivery_tag": tag})

            return

        if settlement is Settlement.ACK:
            session.channel.basic_ack(tag)

            return

        if settlement is Settlement.DEAD_LETTER:
            session.channel.basic_nack(tag, requeue=False)

            return

        # Returning a delivery means giving the connection up: a requeue on the
        # spot would be handed straight back to this consumer.
        session.connection.close()

    def _drain(self, session: _Session) -> None:
        """Give the delivery in flight a bounded chance to finish and settle."""
        worker = session.working
        if worker is None or not worker.is_alive():
            return

        log.info("waiting for the delivery in flight", extra={"timeout": self._policy.shutdown_timeout})

        deadline = time.monotonic() + self._policy.shutdown_timeout
        while worker.is_alive() and time.monotonic() < deadline and session.connection.is_open:
            session.connection.process_data_events(time_limit=DRAIN_TICK)

        if worker.is_alive():
            log.warning("the delivery in flight did not finish, it will be redelivered")

            return

        if session.connection.is_open:
            # The settlement is queued on the connection thread; let it out.
            session.connection.process_data_events(time_limit=DRAIN_TICK)


def message_id(properties: Any) -> str:
    """The outbox id the relay put on the message. Tracing only."""
    headers = getattr(properties, "headers", None) or {}

    return str(headers.get(topology.MESSAGE_ID_HEADER, ""))


def _close(connection: Any) -> None:
    """Close a connection without letting the failure hide the real one."""
    try:
        if connection.is_open:
            connection.close()
    except Exception:
        log.debug("closing the broker connection failed", exc_info=True)
