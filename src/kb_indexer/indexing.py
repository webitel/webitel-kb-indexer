"""The pipeline: what one re-indexing job does with one article version."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from typing import Protocol

from kb_indexer import chunking
from kb_indexer.embedding import Embedder
from kb_indexer.events import ArticleReindex
from kb_indexer.handler import PermanentError, TransientError
from kb_indexer.metrics import Metrics
from kb_indexer.resolver import SpaceEmbedding
from kb_indexer.store import Job

log = logging.getLogger(__name__)


class Storage(Protocol):
    """The database, as the pipeline uses it."""

    def job(self, version_id: int) -> Job | None:
        """Read the version to index."""
        ...

    def superseded(self, version_id: int, model_id: int) -> bool:
        """Whether a newer version or another model of the space took over."""
        ...

    def mark_indexing(self, article_id: int, version_id: int, model_id: int) -> None:
        """Record that the article is being indexed, unless a newer job speaks for it."""
        ...

    def mark_failed(self, article_id: int, version_id: int, model_id: int | None) -> None:
        """Record that the article was not indexed, unless a newer job speaks for it."""
        ...

    def write_chunks(self, version_id: int, contents: list[str]) -> list[int]:
        """Replace the chunks of a version."""
        ...

    def embedded(self, model_id: int, chunk_ids: list[int]) -> set[int]:
        """Which of the chunks already carry a vector of the model."""
        ...

    def write_embeddings(
        self,
        job: Job,
        model_id: int,
        dimensions: int,
        vectors: list[tuple[int, list[float]]],
    ) -> None:
        """Store the vectors of the chunks of a job under the model, in the column of its size."""
        ...

    def publish(self, article_id: int, version_id: int, version_number: int, model_id: int) -> bool:
        """Point the article at the version and drop what is no longer published."""
        ...

    def prune(self, article_id: int) -> None:
        """Drop the chunks older than the published version and the vectors of other models."""
        ...


class Resolving(Protocol):
    """kb-api, as the pipeline uses it."""

    def resolve(self, space_id: int) -> SpaceEmbedding | None:
        """The model of the space, or nothing when the space is not embedded."""
        ...

    def forget(self, space_id: int) -> None:
        """Ask for the model of the space again next time."""
        ...


class Providing(Protocol):
    """The embedding providers, as the pipeline uses them."""

    def for_provider(self, provider: str) -> Embedder:
        """The client of a provider."""
        ...


class _SupersededError(Exception):
    """A newer version or another model of the space took over, and its own job does the rest."""


class IndexingHandler:
    """Turns an article version into chunks, embeds them and publishes it."""

    def __init__(
        self,
        store: Storage,
        resolver: Resolving,
        providers: Providing,
        metrics: Metrics,
        policy: chunking.Policy = chunking.DEFAULT,
    ) -> None:
        self._store = store
        self._resolver = resolver
        self._providers = providers
        self._metrics = metrics
        self._policy = policy

    def handle(self, event: ArticleReindex) -> None:
        """Re-index one article version."""
        job = self._store.job(event.version_id)
        if job is None or job.article_deleted:
            log.info("nothing to index", extra={**event.as_fields(), "reason": _absent(job)})

            return

        self._accept(event, job)

        try:
            self._index(event, job)
        except _SupersededError:
            self._leave(event, job)
        except (PermanentError, TransientError):
            # A failure of a version nobody needs anymore is not a failure.
            if not self._store.superseded(job.version_id, job.model_id):
                raise

            self._leave(event, job)

    def give_up(self, event: ArticleReindex) -> None:
        """Record that the article will not be indexed from this event."""
        self._store.mark_failed(event.article_id, event.version_id, None)

    def _index(self, event: ArticleReindex, job: Job) -> None:
        """Chunk, embed and publish one version."""
        self._ensure_newest(job)
        self._store.mark_indexing(job.article_id, job.version_id, job.model_id)

        model = self._model(job)
        embedder = None if model is None else self._providers.for_provider(model.provider)

        chunks = chunking.split(job.body_markdown, title=job.subject, policy=self._policy)
        written = self._store.write_chunks(job.version_id, chunks)
        embedded = 0 if model is None or embedder is None else self._embed(job, model, embedder, written, chunks)
        published = self._store.publish(job.article_id, job.version_id, job.version_number, job.model_id)

        lag = self._metrics.indexed(event.occurred_at, embedded=model is not None) if published else None

        log.info(
            "article indexed",
            extra={
                **event.as_fields(),
                "chunks": len(chunks),
                "embedded": embedded,
                "model_id": 0 if model is None else model.model_id,
                "published": published,
                "lag": lag,
            },
        )

    def _leave(self, event: ArticleReindex, job: Job) -> None:
        """End a job a newer one took over, keeping only what is published."""
        self._store.prune(job.article_id)
        log.info("superseded by a newer version or model", extra=event.as_fields())

    def _ensure_newest(self, job: Job) -> None:
        """Stop the work once a newer version or another model of the space exists."""
        if self._store.superseded(job.version_id, job.model_id):
            raise _SupersededError

    def _model(self, job: Job) -> SpaceEmbedding | None:
        """The model of the space, the one the job read from the database."""
        model = self._resolver.resolve(job.space_id)
        if _model_id(model) == job.model_id:
            return model

        # The cache may still hold the model the space had before.
        self._resolver.forget(job.space_id)
        model = self._resolver.resolve(job.space_id)
        if _model_id(model) == job.model_id:
            return model

        self._ensure_newest(job)

        msg = f"kb-api resolves model {_model_id(model)} for space {job.space_id}, the database holds {job.model_id}"
        raise TransientError(msg)

    def _embed(
        self,
        job: Job,
        model: SpaceEmbedding,
        embedder: Embedder,
        chunk_ids: list[int],
        contents: list[str],
    ) -> int:
        """Vectorize the chunks that carry no vector of this model yet."""
        if not chunk_ids:
            return 0

        stored = self._store.embedded(model.model_id, chunk_ids)
        pending = [pair for pair in zip(chunk_ids, contents, strict=True) if pair[0] not in stored]
        if not pending:
            return 0

        written = 0
        for batch in _batched(pending, embedder.batch):
            # A long article can take minutes; a newer version must not wait for it.
            self._ensure_newest(job)
            try:
                vectors = self._call(model, embedder, [content for _chunk_id, content in batch])
            except TransientError:
                # The key or the address of the model may have changed under
                # us: the retry asks kb-api again rather than wait the cache out.
                self._resolver.forget(job.space_id)

                raise

            # After every batch, so a failure halfway keeps what it paid for.
            self._store.write_embeddings(
                job,
                model.model_id,
                model.dimensions,
                [(chunk_id, vector) for (chunk_id, _content), vector in zip(batch, vectors, strict=True)],
            )
            written += len(batch)

        return written

    def _call(self, model: SpaceEmbedding, embedder: Embedder, texts: list[str]) -> list[list[float]]:
        """One timed call to the provider; a refusal costs time too."""
        started = time.monotonic()
        error_type = None
        try:
            return embedder.embed(model, texts)
        except BaseException as failure:
            error_type = type(failure).__name__
            raise
        finally:
            self._metrics.embedding(model.provider, model.model_ref, time.monotonic() - started, error_type=error_type)

    def _accept(self, event: ArticleReindex, job: Job) -> None:
        """Refuse a job no attempt of this worker can complete."""
        if job.article_id != event.article_id:
            msg = f"version {event.version_id} belongs to article {job.article_id}, not {event.article_id}"
            raise PermanentError(msg)

        if job.chunking_strategy != chunking.STRATEGY:
            msg = f"unsupported chunking strategy {job.chunking_strategy!r}"
            raise PermanentError(msg)


def _batched(pending: list[tuple[int, str]], size: int) -> Iterator[list[tuple[int, str]]]:
    """Cut the work into the calls one provider takes."""
    for start in range(0, len(pending), size):
        yield pending[start : start + size]


def _model_id(model: SpaceEmbedding | None) -> int:
    return 0 if model is None else model.model_id


def _absent(job: Job | None) -> str:
    return "the version is gone" if job is None else "the article is deleted"
