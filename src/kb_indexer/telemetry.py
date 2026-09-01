"""OpenTelemetry providers, configured by the standard OTEL_* variables."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from opentelemetry import metrics
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource

log = logging.getLogger(__name__)

ENDPOINT_VARS = (
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
    "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT",
)

_SHUTDOWN_TIMEOUT_MS = 5_000


def shutdown_quietly(name: str, shutdown: Callable[[], object]) -> None:
    """Shut a provider down without letting it abort the rest of the shutdown."""
    try:
        shutdown()
    except Exception:
        log.exception("telemetry failed to shut down", extra={"provider": name})


@contextmanager
def telemetry(service_name: str, service_version: str, *, export_logs: bool = False) -> Iterator[None]:
    """Metrics and, when asked, a log sink, for as long as the block runs."""
    if not any(os.environ.get(name) for name in ENDPOINT_VARS):
        if export_logs:
            log.warning("log export requested but no otlp endpoint is configured")

        log.info("telemetry disabled, no otlp endpoint configured")
        yield

        return

    resource = Resource.create({"service.name": service_name, "service.version": service_version})

    meters = MeterProvider(
        resource=resource,
        metric_readers=[PeriodicExportingMetricReader(OTLPMetricExporter())],
    )
    metrics.set_meter_provider(meters)

    logs = None
    handler = None
    if export_logs:
        logs = LoggerProvider(resource=resource)
        logs.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter()))
        set_logger_provider(logs)
        handler = LoggingHandler(logger_provider=logs)
        logging.getLogger().addHandler(handler)

    log.info("telemetry enabled", extra={"logs": export_logs})

    try:
        yield
    finally:
        # Shutdown flushes what is buffered. A broken collector must neither
        # hold the process nor abort the rest of the shutdown.
        if handler is not None:
            logging.getLogger().removeHandler(handler)

        if logs is not None:
            shutdown_quietly("logs", logs.shutdown)

        shutdown_quietly("metrics", lambda: meters.shutdown(timeout_millis=_SHUTDOWN_TIMEOUT_MS))
