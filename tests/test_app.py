import os
import signal
import threading
from contextlib import contextmanager

import pytest

from kb_indexer import app, config


@pytest.fixture
def settings(complete_env):
    return config.load(env_file=None)


def test_the_components_are_left_before_the_process_returns(settings, monkeypatch):
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


class Refuses:
    """A component that fails while it is being entered."""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        msg = "no collector"
        raise RuntimeError(msg)

    def __exit__(self, *exception):
        return False


def test_a_component_that_cannot_start_stops_the_process(settings, monkeypatch):
    monkeypatch.setattr(app, "telemetry", Refuses)
    # A safety net: a lifecycle that swallowed the failure would wait for a
    # signal forever, and a hanging test says far less than a failing one.
    threading.Timer(1, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()

    with pytest.raises(RuntimeError, match="no collector"):
        app.run(settings)
