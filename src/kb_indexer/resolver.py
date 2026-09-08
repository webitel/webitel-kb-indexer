"""The kb-api internal API: how a space is embedded, and with what credential."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Protocol, cast

import grpc

from kb_indexer.config import Settings
from kb_indexer.handler import PermanentError, TransientError
from kb_indexer.kbapi import indexing_pb2, indexing_pb2_grpc

log = logging.getLogger(__name__)

# What kb-api authorizes a service call by.
SERVICE_TOKEN_HEADER = "x-webitel-service-token"  # noqa: S105

# Failures worth another attempt: the service is down, slow, overloaded, or
# broke on its own side. Everything else describes this worker or the
# registration, and no attempt of ours can change it.
TRANSIENT_CODES = frozenset(
    {
        grpc.StatusCode.UNAVAILABLE,
        grpc.StatusCode.DEADLINE_EXCEEDED,
        grpc.StatusCode.RESOURCE_EXHAUSTED,
        grpc.StatusCode.CANCELLED,
        grpc.StatusCode.UNKNOWN,
        grpc.StatusCode.INTERNAL,
    },
)


@dataclass(frozen=True, slots=True)
class SpaceEmbedding:
    """The embedding model of a space, as kb-api resolved it."""

    model_id: int
    provider: str
    model_ref: str
    dimensions: int
    endpoint: str
    validated: bool
    # The provider credential. Out of the repr on purpose: this object travels
    # through logs, and the protobuf message it came from prints the key.
    api_key: str = field(repr=False)


class Resolver:
    """Asks kb-api how a space is embedded."""

    def __init__(self, channel: grpc.Channel, token: str, timeout: float) -> None:
        self._stub = indexing_pb2_grpc.IndexingStub(channel)
        self._metadata = ((SERVICE_TOKEN_HEADER, token),)
        self._timeout = timeout

    def resolve(self, space_id: int) -> SpaceEmbedding | None:
        """The model of the space, or nothing when the space is not embedded."""
        request = indexing_pb2.ResolveSpaceEmbeddingRequest(space_id=space_id)

        try:
            answer = self._stub.ResolveSpaceEmbedding(request, metadata=self._metadata, timeout=self._timeout)
        except grpc.RpcError as refused:
            raise _classified(cast("grpc.Call", refused)) from refused

        if not answer.vector_search_enabled:
            log.debug("the space is not embedded", extra={"space_id": space_id})

            return None

        log.debug(
            "space embedding resolved",
            extra={
                "space_id": space_id,
                "model_id": answer.model_id,
                "provider": answer.provider,
                "validated": answer.validated,
            },
        )

        return SpaceEmbedding(
            model_id=answer.model_id,
            provider=answer.provider,
            model_ref=answer.model_ref,
            dimensions=answer.dimensions,
            endpoint=answer.endpoint,
            validated=answer.validated,
            api_key=answer.api_key,
        )


class Answering(Protocol):
    """What the cache wraps: something that answers about a space."""

    def resolve(self, space_id: int) -> SpaceEmbedding | None:
        """The model of the space, or nothing when the space is not embedded."""
        ...


class Cached:
    """A resolver that reuses an answer for a while.

    The entries are guarded: a broker reconnect leaves the previous delivery
    running while its redelivery starts, so two threads can ask at once.
    """

    def __init__(
        self,
        resolver: Answering,
        ttl: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._resolver = resolver
        self._ttl = ttl
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: dict[int, tuple[float, SpaceEmbedding | None]] = {}
        self._forgotten: dict[int, int] = {}

    def resolve(self, space_id: int) -> SpaceEmbedding | None:
        """The remembered model of the space, or a fresh answer."""
        asked = self._clock()

        with self._lock:
            entry = self._entries.get(space_id)
            forgotten = self._forgotten.get(space_id, 0)

        if entry is not None and entry[0] > asked:
            return entry[1]

        found = self._resolver.resolve(space_id)

        with self._lock:
            if self._forgotten.get(space_id, 0) == forgotten:
                self._entries[space_id] = (asked + self._ttl, found)

        return found

    def forget(self, space_id: int) -> None:
        """Drop the entry, so a rotated credential is picked up at once."""
        with self._lock:
            self._entries.pop(space_id, None)
            self._forgotten[space_id] = self._forgotten.get(space_id, 0) + 1


@contextmanager
def connect(settings: Settings) -> Iterator[Resolver]:
    """Hold the channel to kb-api for as long as the process runs.

    The channel dials lazily, so a kb-api that is down makes the deliveries
    fail and be retried rather than stopping the worker from starting.
    """
    channel = (
        grpc.secure_channel(settings.kb_api_addr, grpc.ssl_channel_credentials())
        if settings.kb_api_tls
        else grpc.insecure_channel(settings.kb_api_addr)
    )
    log.info("kb-api channel opened", extra={"addr": settings.kb_api_addr, "tls": settings.kb_api_tls})

    try:
        yield Resolver(channel, settings.kb_api_service_token, settings.kb_api_timeout)
    finally:
        channel.close()


def _classified(error: grpc.Call) -> Exception:
    """Tell a kb-api worth waiting for apart from an answer that will not change."""
    code = error.code()
    msg = f"kb-api refused to resolve the space: {code.name.lower()}: {error.details()}"

    if code in TRANSIENT_CODES:
        return TransientError(msg)

    return PermanentError(msg)
