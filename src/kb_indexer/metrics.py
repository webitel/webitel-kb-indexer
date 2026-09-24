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

# Names follow the Webitel semantic conventions of webitel-go-kit: dotted, under
# the webitel namespace, the unit kept out of the name. What OpenTelemetry
# already defines is taken from it instead.
INDEX_DURATION = "webitel.kb.article.index.duration"
INDEX_JOB_COUNT = "webitel.kb.article.index.job.count"
INDEX_JOB_FAILED = "webitel.kb.article.index.job.failed"
ATTR_EMBEDDED = "webitel.kb.article.index.embedded"
ATTR_ERROR_TYPE = "error.type"

# The index state kb-api reports articles under: a job waiting in the indexing
# queue is `pending`, one in its dead letter queue is `failed`.
ATTR_INDEX_STATE = "webitel.kb.article.index.state"
STATE_PENDING = "pending"
STATE_FAILED = "failed"

# One provider call, under the GenAI semantic conventions of OpenTelemetry.
GEN_AI_OPERATION_DURATION = "gen_ai.client.operation.duration"
ATTR_GEN_AI_OPERATION = "gen_ai.operation.name"
ATTR_GEN_AI_PROVIDER = "gen_ai.provider.name"
ATTR_GEN_AI_MODEL = "gen_ai.request.model"
GEN_AI_EMBEDDINGS = "embeddings"

# Providers the GenAI conventions have a name for; the rest are self-hosted and
# are reported under the name kb-api registers them with.
GEN_AI_PROVIDERS = {"gemini": "gcp.gemini"}

LAG_BUCKETS: Sequence[float] = (0.5, 1, 2, 5, 10, 15, 20, 30, 45, 60, 120, 300, 600, 1800)

# The boundaries the GenAI conventions ask for: from 10 ms, doubling up to 81.92 s.
EMBEDDING_BUCKETS: Sequence[float] = tuple(0.01 * 2**power for power in range(14))


class Metrics:
    """The instruments, over one meter.

    The depth of the queues is observed, not set: with nothing observed the
    gauge is absent from the export, so a worker that lost its broker stops
    reporting rather than freezing at its last reading. It is a gauge, not an
    up-down counter: every worker reads the same broker queues, so summing
    the readings of several workers would count one queue several times.
    """

    def __init__(self, meter: Meter) -> None:
        self._lag = meter.create_histogram(
            INDEX_DURATION,
            unit="s",
            description="Time from the edit of an article to the version being searchable",
            explicit_bucket_boundaries_advisory=LAG_BUCKETS,
        )
        self._embedding = meter.create_histogram(
            GEN_AI_OPERATION_DURATION,
            unit="s",
            description="GenAI operation duration.",
            explicit_bucket_boundaries_advisory=EMBEDDING_BUCKETS,
        )
        self._failed = meter.create_counter(
            INDEX_JOB_FAILED,
            unit="{job}",
            description="Jobs that ended in the dead letter queue",
        )
        meter.create_observable_gauge(
            INDEX_JOB_COUNT,
            callbacks=[self._observe_depth],
            unit="{job}",
            description="Jobs waiting in the indexing queue and in its dead letter queue",
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
        self._lag.record(lag, {ATTR_EMBEDDED: embedded})

        return lag

    def embedding(self, provider: str, model: str, seconds: float, *, error_type: str | None = None) -> None:
        """Record one call to a provider, whether or not it answered."""
        attributes = {
            ATTR_GEN_AI_OPERATION: GEN_AI_EMBEDDINGS,
            ATTR_GEN_AI_PROVIDER: GEN_AI_PROVIDERS.get(provider, provider),
            ATTR_GEN_AI_MODEL: model,
        }
        if error_type is not None:
            attributes[ATTR_ERROR_TYPE] = error_type

        self._embedding.record(seconds, attributes)

    def failed(self, reason: str) -> None:
        """Count a job that ended in the dead letter queue."""
        self._failed.add(1, {ATTR_ERROR_TYPE: reason})

    def depth(self, queue: int, dlq: int) -> None:
        """Remember the latest reading of the queues."""
        with self._guard:
            self._depth = (queue, dlq)

    def forget_depth(self) -> None:
        """Drop the reading: the connection it was taken over is gone."""
        with self._guard:
            self._depth = None

    def _observe_depth(self, _options: CallbackOptions) -> Iterable[Observation]:
        with self._guard:
            depth = self._depth

        if depth is None:
            return []

        queue, dlq = depth

        return [
            Observation(queue, {ATTR_INDEX_STATE: STATE_PENDING}),
            Observation(dlq, {ATTR_INDEX_STATE: STATE_FAILED}),
        ]
