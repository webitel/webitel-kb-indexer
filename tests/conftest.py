import pytest

from kb_indexer.config import Settings
from kb_indexer.telemetry import ENDPOINT_VARS

# Every variable the process reads, derived from the settings themselves so a
# new field cannot quietly escape the isolation below.
_VARIABLES = tuple(name.upper() for name in Settings.model_fields) + ENDPOINT_VARS

DSN = "postgres://kb:secret@db:5432/webitel"
AMQP = "amqp://webitel:secret@rabbit:5672/"


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """The environment of the developer must not decide the outcome of a test."""
    for name in _VARIABLES:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def complete_env(monkeypatch):
    """The minimum the process needs to start."""
    monkeypatch.setenv("POSTGRES_DSN", DSN)
    monkeypatch.setenv("PUBSUB_URL", AMQP)

    return monkeypatch
