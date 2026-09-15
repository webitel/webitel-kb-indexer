import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from kb_indexer.config import Settings
from kb_indexer.metrics import Metrics


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
_VARIABLES = (
    *(name.upper() for name in Settings.model_fields),
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
    "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT",
)

DSN = "postgres://kb:secret@db:5432/webitel"
AMQP = "amqp://webitel:secret@rabbit:5672/"
CONSUL = "consul:8500"
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
    monkeypatch.setenv("CONSUL_ADDR", CONSUL)
    monkeypatch.setenv("KB_API_SERVICE_TOKEN", TOKEN)

    return monkeypatch


class Recorded:
    """The instruments over a reader the test can look into."""

    def __init__(self):
        self.reader = InMemoryMetricReader()
        self.provider = MeterProvider(metric_readers=[self.reader])
        self.metrics = Metrics(self.provider.get_meter("test"))

    def points(self, name):
        """The data points of one instrument, as (attributes, value or count)."""
        data = self.reader.get_metrics_data()
        for resource in data.resource_metrics if data is not None else ():
            for scope in resource.scope_metrics:
                for metric in scope.metrics:
                    if metric.name == name:
                        return [(dict(point.attributes or {}), _value(point)) for point in metric.data.data_points]

        return []

    def close(self):
        self.provider.shutdown()


def _value(point):
    return point.count if hasattr(point, "bucket_counts") else point.value


@pytest.fixture
def recorded():
    """Metrics whose readings the test can assert on."""
    made = Recorded()
    yield made
    made.close()
