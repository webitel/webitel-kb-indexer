"""OpenTelemetry providers, over the exporters the configuration names."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from opentelemetry import metrics
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter as GrpcLogExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter as GrpcMetricExporter
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter as HttpLogExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter as HttpMetricExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor, LogRecordExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import MetricExporter, PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource

from kb_indexer.config import OTLP_GRPC, OTLP_HTTP

log = logging.getLogger(__name__)

_SHUTDOWN_TIMEOUT_MS = 5_000


def shutdown_quietly(name: str, shutdown: Callable[[], object]) -> None:
    """Shut a provider down without letting it abort the rest of the shutdown."""
    try:
        shutdown()
    except Exception:
        log.exception("telemetry failed to shut down", extra={"provider": name})


def metric_exporter(name: str) -> MetricExporter | None:
    """The metric exporter of a configured name, or nothing when export is off."""
    if name == OTLP_GRPC:
        return GrpcMetricExporter()

    if name == OTLP_HTTP:
        return HttpMetricExporter()

    return None


def log_exporter(name: str) -> LogRecordExporter | None:
    """The log exporter of a configured name, or nothing when export is off."""
    if name == OTLP_GRPC:
        return GrpcLogExporter()

    if name == OTLP_HTTP:
        return HttpLogExporter()

    return None


@contextmanager
def telemetry(
    service_name: str,
    service_version: str,
    *,
    metrics_exporter: str = "",
    logs_exporter: str = "",
    export_logs: bool = False,
) -> Iterator[None]:
    """Metrics and, when asked, a log sink, for as long as the block runs."""
    metric_sink = metric_exporter(metrics_exporter)
    log_sink = log_exporter(logs_exporter) if export_logs else None

    if export_logs and log_sink is None:
        log.warning("log export requested but no log exporter is configured")

    if metric_sink is None and log_sink is None:
        log.info("telemetry disabled, no exporter configured")
        yield

        return

    resource = Resource.create({"service.name": service_name, "service.version": service_version})

    meters = None
    if metric_sink is not None:
        meters = MeterProvider(resource=resource, metric_readers=[PeriodicExportingMetricReader(metric_sink)])
        metrics.set_meter_provider(meters)

    logs = None
    handler = None
    if log_sink is not None:
        logs = LoggerProvider(resource=resource)
        logs.add_log_record_processor(BatchLogRecordProcessor(log_sink))
        set_logger_provider(logs)
        handler = LoggingHandler(logger_provider=logs)
        logging.getLogger().addHandler(handler)

    log.info(
        "telemetry enabled",
        extra={"metrics": metrics_exporter if meters else "", "logs": logs_exporter if logs else ""},
    )

    try:
        yield
    finally:
        # Shutdown flushes what is buffered. A broken collector must neither
        # hold the process nor abort the rest of the shutdown.
        if handler is not None:
            logging.getLogger().removeHandler(handler)

        if logs is not None:
            shutdown_quietly("logs", logs.shutdown)

        if meters is not None:
            shutdown_quietly("metrics", lambda: meters.shutdown(timeout_millis=_SHUTDOWN_TIMEOUT_MS))
