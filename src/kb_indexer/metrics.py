"""The metrics of the worker."""

from __future__ import annotations

import threading
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime

from opentelemetry.metrics import CallbackOptions, Meter, Observation

# Why a delivery ended in the dead letter queue. A closed set: the label must
# not grow with the text of the failures.
REASON_ENVELOPE = "envelope"
REASON_PERMANENT = "permanent"
REASON_EXHAUSTED = "exhausted"
REASON_UNEXPECTED = "unexpected"

OUTCOME_OK = "ok"
OUTCOME_ERROR = "error"

LAG_BUCKETS: Sequence[float] = (0.5, 1, 2, 5, 10, 15, 20, 30, 45, 60, 120, 300, 600, 1800)

# One provider call, up to the timeout it is given.
EMBEDDING_BUCKETS: Sequence[float] = (0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 30)


class Metrics:
    """The instruments, over one meter.

    The depth of the queues is observed, not set: with nothing observed the
    gauge is absent from the export, so a worker that lost its broker stops
    reporting rather than freezing at its last reading.
    """

    def __init__(self, meter: Meter) -> None:
        self._lag = meter.create_histogram(
            "kb_reindex_lag_seconds",
            unit="s",
            description="Time from the edit of an article to the version being searchable",
            explicit_bucket_boundaries_advisory=LAG_BUCKETS,
        )
        self._embedding = meter.create_histogram(
            "kb_embedding_duration_seconds",
            unit="s",
            description="Duration of one embedding call to a provider",
            explicit_bucket_boundaries_advisory=EMBEDDING_BUCKETS,
        )
        self._failed = meter.create_counter(
            "kb_reindex_failed_total",
            unit="{job}",
            description="Jobs that ended in the dead letter queue",
        )
        meter.create_observable_gauge(
            "kb_reindex_queue_depth",
            callbacks=[self._observe_queue],
            unit="{message}",
            description="Jobs waiting in the indexing queue",
        )
        meter.create_observable_gauge(
            "kb_reindex_dlq_depth",
            callbacks=[self._observe_dlq],
            unit="{message}",
            description="Jobs in the dead letter queue, waiting for attention",
        )

        self._guard = threading.Lock()
        self._depth: tuple[int, int] | None = None

    def indexed(self, occurred_at: datetime, *, embedded: bool, now: datetime | None = None) -> float:
        """Record how long the edit took to become searchable, and return it.

        The edit time comes from another host, so a clock that runs behind
        this one is read as no lag rather than a negative one.
        """
        moment = datetime.now(UTC) if now is None else now
        lag = max((moment - occurred_at).total_seconds(), 0.0)
        self._lag.record(lag, {"embedded": embedded})

        return lag

    def embedding(self, provider: str, model: str, seconds: float, *, ok: bool) -> None:
        """Record one call to a provider, whether or not it answered."""
        outcome = OUTCOME_OK if ok else OUTCOME_ERROR
        self._embedding.record(seconds, {"provider": provider, "model": model, "outcome": outcome})

    def failed(self, reason: str) -> None:
        """Count a job that ended in the dead letter queue."""
        self._failed.add(1, {"reason": reason})

    def depth(self, queue: int, dlq: int) -> None:
        """Remember the latest reading of the queues."""
        with self._guard:
            self._depth = (queue, dlq)

    def forget_depth(self) -> None:
        """Drop the reading: the connection it was taken over is gone."""
        with self._guard:
            self._depth = None

    def _observe_queue(self, _options: CallbackOptions) -> Iterable[Observation]:
        return self._observed(0)

    def _observe_dlq(self, _options: CallbackOptions) -> Iterable[Observation]:
        return self._observed(1)

    def _observed(self, position: int) -> list[Observation]:
        with self._guard:
            depth = self._depth

        if depth is None:
            return []

        return [Observation(depth[position])]
