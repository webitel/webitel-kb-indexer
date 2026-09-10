"""Embedding providers: the calls kb-api validates a registration with."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Protocol

import httpx

from kb_indexer.handler import PermanentError, TransientError
from kb_indexer.resolver import SpaceEmbedding

log = logging.getLogger(__name__)

PROVIDER_GEMINI = "gemini"

# Providers served by one local service over the canonical contract.
ENDPOINT_PROVIDERS = frozenset({"bge-m3", "e5", "byom"})

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com"

# Gemini unit-normalizes only its native output; a shorter vector is ours to scale.
GEMINI_NATIVE_DIMENSIONS = 3072

# How many texts one call carries. The self-hosted batch is the smaller one: it
# is usually a single box with one accelerator.
GEMINI_BATCH = 100
ENDPOINT_BATCH = 32

# Documents are stored and searched; queries are embedded at retrieval time,
# by kb-api, and never here.
GEMINI_TASK = "RETRIEVAL_DOCUMENT"
ENDPOINT_TASK = "document"

# How much of a refusal is kept for the log.
MAX_ERROR_BODY = 2048

TOO_MANY_REQUESTS = 429
SERVER_ERROR = 500
UNAUTHORIZED = frozenset({401, 403})


class CredentialRefusedError(TransientError):
    """The provider rejected the credential.

    Transient because the stored key may have been rotated between the answer
    of kb-api and this call: the attempt that follows asks for it again.
    """


class Embedder(Protocol):
    """One provider, as the pipeline uses it."""

    batch: int

    def embed(self, model: SpaceEmbedding, texts: list[str]) -> list[list[float]]:
        """One vector per text, in the order the texts were given."""
        ...


class Gemini:
    """Google AI embeddings."""

    batch = GEMINI_BATCH

    def __init__(self, client: httpx.Client, base_url: str = GEMINI_BASE_URL) -> None:
        self._client = client
        self._base_url = base_url.rstrip("/")

    def embed(self, model: SpaceEmbedding, texts: list[str]) -> list[list[float]]:
        """One vector per text, in the order the texts were given."""
        path = _model_path(model.model_ref)
        common: dict[str, Any] = {"model": path, "taskType": GEMINI_TASK}
        if model.dimensions > 0:
            common["outputDimensionality"] = model.dimensions

        body = {"requests": [{**common, "content": {"parts": [{"text": text}]}} for text in texts]}
        url = f"{self._base_url}/v1beta/{path}:batchEmbedContents"
        payload = _posted(self._client, url, body, {"x-goog-api-key": model.api_key})

        vectors = [_numbers(item.get("values")) for item in _listed(payload, "embeddings")]
        if 0 < model.dimensions < GEMINI_NATIVE_DIMENSIONS:
            vectors = [_normalized(vector) for vector in vectors]

        return _checked(vectors, len(texts), model.dimensions)


class Endpoint:
    """A self-hosted service serving the canonical `/embed` contract."""

    batch = ENDPOINT_BATCH

    def __init__(self, client: httpx.Client) -> None:
        self._client = client

    def embed(self, model: SpaceEmbedding, texts: list[str]) -> list[list[float]]:
        """One vector per text, in the order the texts were given."""
        if not model.endpoint:
            msg = f"model {model.model_ref!r} is self-hosted but carries no endpoint url"
            raise PermanentError(msg)

        body = {
            "model": model.model_ref,
            "texts": texts,
            "dimensions": model.dimensions,
            "task": ENDPOINT_TASK,
        }
        url = f"{model.endpoint.rstrip('/')}/embed"
        payload = _posted(self._client, url, body, {})

        vectors = [_numbers(item) for item in _listed(payload, "embeddings")]

        return _checked(vectors, len(texts), model.dimensions)


class Providers:
    """The providers this worker can call, over one client."""

    def __init__(self, client: httpx.Client) -> None:
        self._client = client

    def for_provider(self, provider: str) -> Embedder:
        """The client of a provider, or a refusal no attempt can fix."""
        if provider == PROVIDER_GEMINI:
            return Gemini(self._client)

        if provider in ENDPOINT_PROVIDERS:
            return Endpoint(self._client)

        msg = f"unsupported embedding provider {provider!r}"
        raise PermanentError(msg)


@contextmanager
def providers(timeout: float) -> Iterator[Providers]:
    """Hold one http client for as long as the process runs."""
    with httpx.Client(timeout=timeout) as client:
        log.info("embedding clients ready", extra={"timeout": timeout})
        yield Providers(client)


def _posted(
    client: httpx.Client,
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
) -> dict[str, Any]:
    """One call to a provider, with the failure told apart from the bug."""
    try:
        response = client.post(url, json=body, headers=headers)
    except httpx.TimeoutException as slow:
        msg = f"the provider did not answer in time: {slow}"
        raise TransientError(msg) from slow
    except httpx.UnsupportedProtocol as unusable:
        msg = f"the endpoint of the registration cannot be called: {unusable}"
        raise PermanentError(msg) from unusable
    except httpx.HTTPError as unreachable:
        msg = f"the provider could not be reached: {unreachable}"
        raise TransientError(msg) from unreachable

    _accepted(response)

    try:
        payload = response.json()
    except ValueError as broken:
        msg = f"the provider answered with something other than json: {broken}"
        raise PermanentError(msg) from broken

    if not isinstance(payload, dict):
        msg = f"the provider answered with a {type(payload).__name__}, expected an object"
        raise PermanentError(msg)

    return payload


def _accepted(response: httpx.Response) -> None:
    """Refuse a status, saying whether another attempt is worth making."""
    if response.is_success:
        return

    detail = response.text[:MAX_ERROR_BODY]
    msg = f"the provider returned status {response.status_code}: {detail}"

    if response.status_code in UNAUTHORIZED:
        raise CredentialRefusedError(msg)

    if response.status_code == TOO_MANY_REQUESTS or response.status_code >= SERVER_ERROR:
        raise TransientError(msg)

    raise PermanentError(msg)


def _listed(payload: dict[str, Any], name: str) -> list[Any]:
    """The array of the answer, or a refusal naming what was expected."""
    found = payload.get(name)
    if not isinstance(found, list):
        msg = f"the provider answered without a {name} array"
        raise PermanentError(msg)

    return found


def _numbers(raw: Any) -> list[float]:
    """One vector of the answer, as numbers."""
    if not isinstance(raw, list):
        msg = f"the provider returned a {type(raw).__name__} where a vector was expected"
        raise PermanentError(msg)

    try:
        return [float(value) for value in raw]
    except (TypeError, ValueError) as unusable:
        msg = f"the provider returned a vector that is not numeric: {unusable}"
        raise PermanentError(msg) from unusable


def _normalized(vector: list[float]) -> list[float]:
    """Scale to unit length; a zero vector is left as it is."""
    length = math.sqrt(math.fsum(value * value for value in vector))
    if length == 0:
        return vector

    return [value / length for value in vector]


def _checked(vectors: list[list[float]], count: int, dimensions: int) -> list[list[float]]:
    """Refuse an answer the column would not take, before it reaches the database."""
    if len(vectors) != count:
        msg = f"the provider returned {len(vectors)} vectors for {count} texts"
        raise PermanentError(msg)

    for vector in vectors:
        if len(vector) != dimensions:
            msg = f"the provider returned a vector of {len(vector)} values, expected {dimensions}"
            raise PermanentError(msg)

        if not all(math.isfinite(value) for value in vector):
            msg = "the provider returned a vector holding a value that is not a number"
            raise PermanentError(msg)

    return vectors


def _model_path(model_ref: str) -> str:
    """The `models/` prefix the api expects."""
    if model_ref.startswith("models/"):
        return model_ref

    return f"models/{model_ref}"
