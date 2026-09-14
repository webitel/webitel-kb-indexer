import json
import threading
import time
from collections import deque
from collections.abc import Callable
from types import SimpleNamespace

import pika
import pytest
from pika.exceptions import AMQPConnectionError

from kb_indexer import consumer, topology
from kb_indexer.handler import PermanentError, TransientError
from tests.conftest import Recorded

URL = "amqp://webitel:secret@rabbit:5672/"

ENVELOPE = json.dumps(
    {
        "type": "article.reindex",
        "schema": 1,
        "occurred_at": "2026-07-27T10:30:00Z",
        "article_id": 7,
        "version_id": 19,
        "space_id": 3,
        "domain_id": 1,
    },
).encode()


class Method:
    def __init__(self, delivery_tag):
        self.delivery_tag = delivery_tag


class Properties:
    def __init__(self, headers=None):
        self.headers = headers


class Declared:
    """The answer of the broker to a queue declaration."""

    def __init__(self, message_count):
        self.method = SimpleNamespace(message_count=message_count)


class FakeChannel:
    def __init__(self):
        self.is_open = True
        self.prefetch = None
        self.consuming = None
        self.acked = []
        self.nacked = []
        self.declared = []
        self.depths = {topology.REINDEX_QUEUE: 0, topology.REINDEX_DLQ: 0}
        self.probed = 0

    def exchange_declare(self, exchange, **arguments):
        self.declared.append(exchange)

    def queue_declare(self, queue, passive=False, **arguments):
        if passive:
            self.probed += 1

            return Declared(self.depths[queue])

        self.declared.append(queue)

        return Declared(0)

    def queue_bind(self, queue, exchange, **arguments):
        self.declared.append((queue, exchange))

    def basic_qos(self, prefetch_count):
        self.prefetch = prefetch_count

    def basic_consume(self, queue, on_message_callback):
        self.consuming = (queue, on_message_callback)

    def basic_ack(self, delivery_tag):
        self.acked.append(delivery_tag)

    def basic_nack(self, delivery_tag, requeue=True):
        self.nacked.append((delivery_tag, requeue))


class FakeConnection:
    """A broker connection whose io loop only runs inside process_data_events."""

    def __init__(self):
        self.is_open = True
        self.ticks = 0
        self.channel_ = None
        self._guard = threading.Lock()
        self._pending: deque[Callable[[], None]] = deque()
        self._deliveries: deque[tuple[int, bytes, dict[str, str] | None]] = deque()

    def channel(self):
        self.channel_ = FakeChannel()

        return self.channel_

    def add_callback_threadsafe(self, callback):
        if not self.is_open:
            msg = "connection is closed"
            raise RuntimeError(msg)

        with self._guard:
            self._pending.append(callback)

    def process_data_events(self, time_limit=0):
        self.ticks += 1
        with self._guard:
            deliveries, self._deliveries = list(self._deliveries), deque()
            pending, self._pending = list(self._pending), deque()

        for tag, body, headers in deliveries:
            assert self.channel_ is not None
            _, callback = self.channel_.consuming
            callback(self.channel_, Method(tag), Properties(headers), body)

        for callback in pending:
            callback()

        time.sleep(0.005)

    def close(self):
        self.is_open = False
        if self.channel_ is not None:
            self.channel_.is_open = False

    # Test side.

    def deliver(self, body=ENVELOPE, tag=1, headers=None):
        with self._guard:
            self._deliveries.append((tag, body, headers))


class Pipeline:
    """A handler whose every outcome the test dictates."""

    def __init__(self, outcomes=(), block=None):
        self.outcomes = deque(outcomes)
        self.block = block
        self.on_handle = None
        self.handled = []
        self.given_up = []
        self.give_up_fails = False

    def handle(self, event):
        self.handled.append(event)
        if self.on_handle is not None:
            self.on_handle()
        if self.block is not None:
            self.block.wait(5)
        if self.outcomes:
            failure = self.outcomes.popleft()
            if failure is not None:
                raise failure

    def give_up(self, event):
        self.given_up.append(event)
        if self.give_up_fails:
            msg = "the database is unreachable"
            raise RuntimeError(msg)


class Harness:
    def __init__(self, monkeypatch, handler, policy=None, connections=1, probe=60.0):
        monkeypatch.setattr(consumer, "RECONNECT_DELAY", 0.01)
        monkeypatch.setattr(consumer, "RECONNECT_DELAY_MAX", 0.01)
        monkeypatch.setattr(consumer, "DEPTH_PROBE", probe)

        self.connections = [FakeConnection() for _ in range(connections)]
        self.attempts = 0

        available = list(self.connections)

        def dial(_parameters):
            self.attempts += 1
            if not available:
                msg = "the broker is gone"
                raise AMQPConnectionError(msg)

            return available.pop(0)

        monkeypatch.setattr(pika, "BlockingConnection", dial)

        self.stopping = threading.Event()
        self.recorded = Recorded()
        self.consumer = consumer.Consumer(URL, handler, self.stopping, policy, self.recorded.metrics)
        self.thread = threading.Thread(target=self.consumer.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        wait_for(lambda: self.connections[0].channel_ is not None and self.connections[0].channel_.consuming)

        return self

    def __exit__(self, *failure):
        self.stopping.set()
        self.thread.join(5)
        self.recorded.close()
        assert not self.thread.is_alive(), "the consumer did not stop"

        return False

    def failed(self):
        return {attributes["reason"]: count for attributes, count in self.recorded.points("kb_reindex_failed_total")}

    def depth(self):
        queue = self.recorded.points("kb_reindex_queue_depth")
        dlq = self.recorded.points("kb_reindex_dlq_depth")

        return (queue[0][1], dlq[0][1]) if queue and dlq else None

    @property
    def channel(self):
        return self.connections[0].channel_


def wait_for(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.01)

    msg = "the expected state was never reached"
    raise AssertionError(msg)


@pytest.fixture
def brisk():
    """A policy that does not make the tests wait out real backoff."""
    return consumer.Policy(retries=2, retry_backoff=0.001, shutdown_timeout=2.0)


def test_the_topology_is_declared_before_the_queue_is_consumed(monkeypatch, brisk):
    with Harness(monkeypatch, Pipeline(), brisk) as harness:
        channel = harness.channel

        assert topology.REINDEX_DLQ in channel.declared
        assert channel.prefetch == topology.PREFETCH
        assert channel.consuming[0] == topology.REINDEX_QUEUE


def test_a_handled_delivery_is_acknowledged(monkeypatch, brisk):
    pipeline = Pipeline()

    with Harness(monkeypatch, pipeline, brisk) as harness:
        harness.connections[0].deliver(tag=17)
        wait_for(lambda: harness.channel.acked)

        assert harness.channel.acked == [17]
        assert harness.channel.nacked == []
        assert [event.article_id for event in pipeline.handled] == [7]


@pytest.mark.parametrize(
    "body",
    [
        b"{ not json",
        json.dumps({"type": "article.published", "schema": 1}).encode(),
        ENVELOPE.replace(b'"schema": 1', b'"schema": 2'),
        ENVELOPE.replace(b'"article_id": 7', b'"article_id": 0'),
    ],
)
def test_an_envelope_the_worker_cannot_act_on_goes_to_the_dead_letter_queue(monkeypatch, brisk, body):
    pipeline = Pipeline()

    with Harness(monkeypatch, pipeline, brisk) as harness:
        harness.connections[0].deliver(body=body, tag=3)
        wait_for(lambda: harness.channel.nacked)

        # requeue would hand it straight back and never reach the dead letters.
        assert harness.channel.nacked == [(3, False)]
        assert pipeline.handled == []
        assert pipeline.given_up == []
        assert harness.failed() == {"envelope": 1}


def test_a_transient_failure_is_retried_and_then_given_up_on(monkeypatch, brisk):
    pipeline = Pipeline(outcomes=[TransientError("timeout")] * 10)

    with Harness(monkeypatch, pipeline, brisk) as harness:
        harness.connections[0].deliver(tag=5)
        wait_for(lambda: harness.channel.nacked)

        assert len(pipeline.handled) == brisk.retries + 1
        assert len(pipeline.given_up) == 1
        assert harness.channel.nacked == [(5, False)]
        assert harness.failed() == {"exhausted": 1}


def test_a_transient_failure_that_clears_is_acknowledged(monkeypatch, brisk):
    pipeline = Pipeline(outcomes=[TransientError("timeout"), None])

    with Harness(monkeypatch, pipeline, brisk) as harness:
        harness.connections[0].deliver(tag=5)
        wait_for(lambda: harness.channel.acked)

        assert len(pipeline.handled) == 2
        assert pipeline.given_up == []
        assert harness.channel.nacked == []


@pytest.mark.parametrize(
    ("failure", "reason"),
    [(PermanentError("malformed body"), "permanent"), (ValueError("a bug"), "unexpected")],
)
def test_a_failure_that_is_not_transient_is_not_retried(monkeypatch, brisk, failure, reason):
    pipeline = Pipeline(outcomes=[failure])

    with Harness(monkeypatch, pipeline, brisk) as harness:
        harness.connections[0].deliver(tag=9)
        wait_for(lambda: harness.channel.nacked)

        assert len(pipeline.handled) == 1
        assert len(pipeline.given_up) == 1
        assert harness.channel.nacked == [(9, False)]
        assert harness.failed() == {reason: 1}


def test_a_failure_that_cannot_be_recorded_returns_the_delivery(monkeypatch, brisk):
    """Dead lettering it would leave the article indexing with nothing left to deliver it."""
    pipeline = Pipeline(outcomes=[PermanentError("no chunker")])
    pipeline.give_up_fails = True

    with Harness(monkeypatch, pipeline, brisk) as harness:
        harness.connections[0].deliver(tag=11)
        wait_for(lambda: not harness.connections[0].is_open)

        assert harness.channel.nacked == []
        assert harness.channel.acked == []
        assert harness.failed() == {}


def test_a_shutdown_during_a_retry_returns_the_delivery(monkeypatch):
    policy = consumer.Policy(retries=5, retry_backoff=0.05, shutdown_timeout=2.0)
    pipeline = Pipeline(outcomes=[TransientError("timeout")] * 10)

    with Harness(monkeypatch, pipeline, policy) as harness:
        harness.connections[0].deliver(tag=13)
        wait_for(lambda: pipeline.handled)
        harness.stopping.set()
        wait_for(lambda: not harness.connections[0].is_open)

        assert harness.channel.nacked == []
        assert pipeline.given_up == []


def test_a_stop_during_the_backoff_does_not_start_another_attempt(monkeypatch):
    """Indexing one article can run for minutes; a stop must not begin that work."""
    policy = consumer.Policy(retries=5, retry_backoff=0.5, shutdown_timeout=2.0)
    pipeline = Pipeline(outcomes=[TransientError("timeout")] * 10)

    with Harness(monkeypatch, pipeline, policy) as harness:
        pipeline.on_handle = lambda: threading.Timer(0.05, harness.stopping.set).start()
        harness.connections[0].deliver(tag=41)
        harness.thread.join(5)

        assert len(pipeline.handled) == 1
        assert pipeline.given_up == []
        assert harness.channel.nacked == []


def test_processing_does_not_hold_the_connection_thread(monkeypatch, brisk):
    """A long article must not cost the connection its heartbeats."""
    working = threading.Event()
    pipeline = Pipeline(block=working)

    with Harness(monkeypatch, pipeline, brisk) as harness:
        harness.connections[0].deliver()
        wait_for(lambda: pipeline.handled)

        served = harness.connections[0].ticks
        wait_for(lambda: harness.connections[0].ticks > served + 5)

        working.set()
        wait_for(lambda: harness.channel.acked)


def test_a_connection_lost_while_working_leaves_the_delivery_to_be_redelivered(monkeypatch, brisk):
    working = threading.Event()
    pipeline = Pipeline(block=working)

    with Harness(monkeypatch, pipeline, brisk, connections=2) as harness:
        harness.connections[0].deliver(tag=27)
        wait_for(lambda: pipeline.handled)

        harness.connections[0].close()
        working.set()
        wait_for(lambda: harness.connections[1].channel_ is not None)

        assert harness.channel.acked == []
        assert harness.channel.nacked == []


def test_the_shutdown_waits_for_the_delivery_in_flight(monkeypatch, brisk):
    working = threading.Event()
    pipeline = Pipeline(block=working)

    with Harness(monkeypatch, pipeline, brisk) as harness:
        harness.connections[0].deliver(tag=21)
        wait_for(lambda: pipeline.handled)

        harness.stopping.set()
        time.sleep(0.1)
        working.set()

        wait_for(lambda: harness.channel.acked == [21])


def test_a_delivery_that_outlives_the_shutdown_is_left_for_redelivery(monkeypatch):
    policy = consumer.Policy(retries=0, retry_backoff=0.001, shutdown_timeout=0.2)
    working = threading.Event()
    pipeline = Pipeline(block=working)

    with Harness(monkeypatch, pipeline, policy) as harness:
        harness.connections[0].deliver(tag=23)
        wait_for(lambda: pipeline.handled)

        harness.stopping.set()
        harness.thread.join(5)

        assert harness.channel.acked == []
        assert harness.channel.nacked == []
        working.set()


def test_the_loop_reconnects_after_a_session_ends(monkeypatch, brisk):
    with Harness(monkeypatch, Pipeline(), brisk, connections=2) as harness:
        harness.connections[0].close()
        wait_for(lambda: harness.connections[1].channel_ is not None and harness.connections[1].channel_.consuming)

        harness.connections[1].deliver(tag=31)
        wait_for(lambda: harness.connections[1].channel_.acked == [31])


def test_a_broker_that_cannot_be_reached_is_retried(monkeypatch, brisk):
    harness = Harness(monkeypatch, Pipeline(), brisk, connections=0)
    harness.thread.start()
    try:
        wait_for(lambda: harness.attempts > 2)
    finally:
        harness.stopping.set()
        harness.thread.join(5)

    assert not harness.thread.is_alive()


def test_the_depth_of_the_queues_is_read_over_the_consuming_connection(monkeypatch, brisk):
    pipeline = Pipeline()

    with Harness(monkeypatch, pipeline, brisk, probe=0.05) as harness:
        declared = list(harness.channel.declared)
        harness.channel.depths = {topology.REINDEX_QUEUE: 5, topology.REINDEX_DLQ: 2}
        wait_for(lambda: harness.depth() == (5, 2))

        assert harness.channel.probed >= 2
        assert harness.channel.declared == declared


def test_a_shutdown_during_a_retry_returns_the_delivery_and_counts_no_failure(monkeypatch):
    policy = consumer.Policy(retries=5, retry_backoff=0.05, shutdown_timeout=2.0)
    pipeline = Pipeline(outcomes=[TransientError("timeout")] * 10)

    with Harness(monkeypatch, pipeline, policy) as harness:
        harness.connections[0].deliver(tag=13)
        wait_for(lambda: len(pipeline.handled) == 1)
        harness.stopping.set()
        wait_for(lambda: not harness.connections[0].is_open)

        assert harness.failed() == {}


def test_the_depth_does_not_outlive_the_session(monkeypatch, brisk):
    pipeline = Pipeline()

    with Harness(monkeypatch, pipeline, brisk) as harness:
        wait_for(lambda: harness.depth() == (0, 0))

    # A worker that lost its broker stops reporting rather than freezing at
    # its last reading.
    assert harness.depth() is None


@pytest.mark.parametrize(
    ("base", "ceiling", "expected"),
    [
        (1.0, 30.0, [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0]),
        (0.5, 1.0, [0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        (60.0, 30.0, [30.0] * 7),
    ],
)
def test_delays_double_up_to_the_ceiling(base, ceiling, expected):
    delays = consumer.delays(base, ceiling)

    assert [next(delays) for _ in expected] == expected


@pytest.mark.parametrize(
    ("properties", "expected"),
    [
        (Properties({"x-message-id": "0f0b"}), "0f0b"),
        (Properties({}), ""),
        (Properties(None), ""),
        (None, ""),
    ],
)
def test_the_tracing_header_is_read_when_it_is_there(properties, expected):
    assert consumer.message_id(properties) == expected
