import logging

import pytest
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter as GrpcLogExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter as GrpcMetricExporter
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter as HttpLogExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter as HttpMetricExporter

from kb_indexer.telemetry import log_exporter, metric_exporter, shutdown_quietly, telemetry


@pytest.mark.parametrize(
    ("name", "metrics", "logs"),
    [
        ("otlpgrpc", GrpcMetricExporter, GrpcLogExporter),
        ("otlphttp", HttpMetricExporter, HttpLogExporter),
        ("none", type(None), type(None)),
        ("", type(None), type(None)),
    ],
)
def test_the_exporter_follows_the_configured_name(name, metrics, logs):
    assert type(metric_exporter(name)) is metrics
    assert type(log_exporter(name)) is logs


def test_without_an_exporter_the_block_still_runs(caplog):
    entered = False

    with caplog.at_level(logging.INFO), telemetry("svc", "1.0"):
        entered = True

    assert entered
    assert "telemetry disabled" in caplog.text


def test_logs_asked_for_without_an_exporter_are_reported(caplog):
    with caplog.at_level(logging.INFO), telemetry("svc", "1.0", export_logs=True):
        pass

    assert "no log exporter is configured" in caplog.text
    assert "telemetry disabled" in caplog.text


def test_a_failing_provider_does_not_abort_the_shutdown(caplog):
    def refuse():
        msg = "collector is gone"
        raise RuntimeError(msg)

    with caplog.at_level(logging.ERROR):
        shutdown_quietly("metrics", refuse)

    assert "collector is gone" in caplog.text
