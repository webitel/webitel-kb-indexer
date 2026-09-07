import os
import signal
import threading
from contextlib import contextmanager
from typing import ClassVar

import pytest

from kb_indexer import app, config
from kb_indexer.indexing import IndexingHandler


@pytest.fixture
def settings(complete_env):
    return config.load(env_file=None)


class FakeStore:
    def __init__(self, dsn):
        self.dsn = dsn


@pytest.fixture(autouse=True)
def database(monkeypatch):
    """A unit test must not dial a database."""
    opened = []

    @contextmanager
    def fake_connect(dsn):
        store = FakeStore(dsn)
        opened.append(store)
        yield store

    monkeypatch.setattr(app, "connect", fake_connect)

    return opened


class FakeConsumer:
    """A consumer that only runs until the process is asked to stop."""

    made: ClassVar[list["FakeConsumer"]] = []

    def __init__(self, url, handler, stopping, policy):
        self.url = url
        self.handler = handler
        self.stopping = stopping
        self.policy = policy
        FakeConsumer.made.append(self)

    def run(self):
        self.stopping.wait()


@pytest.fixture
def consumer(monkeypatch):
    FakeConsumer.made = []
    monkeypatch.setattr(app, "Consumer", FakeConsumer)

    return FakeConsumer


def test_the_components_are_left_before_the_process_returns(settings, consumer, monkeypatch):
    journal = []

    @contextmanager
    def fake_telemetry(name, version, *, export_logs):
        journal.append("enter")
        try:
            yield
        finally:
            journal.append("leave")

    monkeypatch.setattr(app, "telemetry", fake_telemetry)
    threading.Timer(0.2, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()

    assert app.run(settings) == 0
    assert journal == ["enter", "leave"]


def test_the_consumer_is_given_the_configured_broker_and_policy(complete_env, consumer, monkeypatch):
    complete_env.setenv("CONSUMER_RETRIES", "2")
    complete_env.setenv("CONSUMER_RETRY_BACKOFF", "0.5")
    complete_env.setenv("CONSUMER_SHUTDOWN_TIMEOUT", "7")
    monkeypatch.setattr(app, "telemetry", lambda *_args, **_kwargs: _nothing())
    threading.Timer(0.2, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()

    settings = config.load(env_file=None)

    assert app.run(settings) == 0

    built = consumer.made[-1]
    assert built.url == settings.pubsub_url
    assert (built.policy.retries, built.policy.retry_backoff, built.policy.shutdown_timeout) == (2, 0.5, 7.0)


def test_the_pipeline_is_given_the_configured_database(settings, consumer, database, monkeypatch):
    monkeypatch.setattr(app, "telemetry", lambda *_args, **_kwargs: _nothing())
    threading.Timer(0.2, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()

    assert app.run(settings) == 0

    built = consumer.made[-1]
    assert isinstance(built.handler, IndexingHandler)
    assert database[-1].dsn == settings.postgres_dsn


@contextmanager
def _nothing():
    yield


class Refuses:
    """A component that fails while it is being entered."""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        msg = "no collector"
        raise RuntimeError(msg)

    def __exit__(self, *exception):
        return False


def test_a_component_that_cannot_start_stops_the_process(settings, consumer, monkeypatch):
    monkeypatch.setattr(app, "telemetry", Refuses)
    # A safety net: a lifecycle that swallowed the failure would wait for a
    # signal forever, and a hanging test says far less than a failing one.
    threading.Timer(1, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()

    with pytest.raises(RuntimeError, match="no collector"):
        app.run(settings)
