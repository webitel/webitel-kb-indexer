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

from kb_indexer import discovery
from kb_indexer.config import Settings
from kb_indexer.handler import PermanentError, TransientError
from kb_indexer.kbapi import indexing_pb2, indexing_pb2_grpc

log = logging.getLogger(__name__)

# What kb-api authorizes a service call by.
SERVICE_TOKEN_HEADER = "x-webitel-service-token"  # noqa: S105

# Every healthy instance gets calls. With one address it changes nothing.
CHANNEL_OPTIONS = (("grpc.service_config", '{"loadBalancingConfig": [{"round_robin": {}}]}'),)

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


class Reconnecting:
    """A resolver that opens the channel when it is needed and drops it when a
    call fails."""

    def __init__(self, target: Callable[[], str], tls: bool, token: str, timeout: float) -> None:
        self._target = target
        self._tls = tls
        self._token = token
        self._timeout = timeout
        self._lock = threading.Lock()
        self._channel: grpc.Channel | None = None
        self._resolver: Resolver | None = None

    def resolve(self, space_id: int) -> SpaceEmbedding | None:
        """The model of the space, over the channel of the moment."""
        resolver, channel = self._open()

        try:
            return resolver.resolve(space_id)
        except TransientError:
            self._drop(channel)

            raise
        except ValueError:
            # A call on a channel that was closed under it is a ValueError
            # rather than a refusal. It is ours only when the channel was
            # given up meanwhile: another delivery failed, or we are stopping.
            if not self._given_up(channel):
                raise

            msg = "the channel to kb-api was closed while the call was in flight"
            raise TransientError(msg) from None

    def close(self) -> None:
        """Close the channel, if one was ever opened."""
        self._drop(self._channel)

    def _open(self) -> tuple[Resolver, grpc.Channel]:
        """The channel and its resolver, opening them on the first call."""
        with self._lock:
            if self._resolver is None or self._channel is None:
                target = self._target()
                self._channel = (
                    grpc.secure_channel(target, grpc.ssl_channel_credentials(), options=CHANNEL_OPTIONS)
                    if self._tls
                    else grpc.insecure_channel(target, options=CHANNEL_OPTIONS)
                )
                self._resolver = Resolver(self._channel, self._token, self._timeout)
                log.info("kb-api channel opened", extra={"target": target, "tls": self._tls})

            return self._resolver, self._channel

    def _given_up(self, channel: grpc.Channel) -> bool:
        """Whether the channel of the call is no longer the one in use."""
        with self._lock:
            return self._channel is not channel

    def _drop(self, channel: grpc.Channel | None) -> None:
        """Give up the channel, unless another thread already replaced it."""
        if channel is None:
            return

        with self._lock:
            if self._channel is not channel:
                return

            self._channel, self._resolver = None, None

        channel.close()
        log.info("kb-api channel closed")


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
def connect(settings: Settings) -> Iterator[Reconnecting]:
    """Hold the connection to kb-api for as long as the process runs."""
    consul_addr, service = settings.consul_addr, settings.kb_api_service
    resolver = Reconnecting(
        lambda: discovery.target(discovery.lookup(consul_addr, service)),
        settings.kb_api_tls,
        settings.kb_api_service_token,
        settings.kb_api_timeout,
    )

    try:
        yield resolver
    finally:
        resolver.close()


def _classified(error: grpc.Call) -> Exception:
    """Tell a kb-api worth waiting for apart from an answer that will not change."""
    code = error.code()
    msg = f"kb-api refused to resolve the space: {code.name.lower()}: {error.details()}"

    if code in TRANSIENT_CODES:
        return TransientError(msg)

    return PermanentError(msg)
