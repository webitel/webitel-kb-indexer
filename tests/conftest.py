import pytest

from kb_indexer.config import Settings
from kb_indexer.telemetry import ENDPOINT_VARS


def pytest_addoption(parser):
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="rewrite the recorded chunking output instead of comparing against it",
    )


@pytest.fixture
def update_golden(request):
    """Whether a golden file is rewritten rather than asserted."""
    return request.config.getoption("--update-golden")


# Every variable the process reads, derived from the settings themselves so a
# new field cannot quietly escape the isolation below.
_VARIABLES = tuple(name.upper() for name in Settings.model_fields) + ENDPOINT_VARS

DSN = "postgres://kb:secret@db:5432/webitel"
AMQP = "amqp://webitel:secret@rabbit:5672/"
KB_API = "kb-api:8080"
TOKEN = "0123456789abcdef0123456789abcdef"


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
    monkeypatch.setenv("KB_API_ADDR", KB_API)
    monkeypatch.setenv("KB_API_SERVICE_TOKEN", TOKEN)

    return monkeypatch
