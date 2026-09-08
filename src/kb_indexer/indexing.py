"""The pipeline: what one re-indexing job does with one article version."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Protocol

from kb_indexer import chunking
from kb_indexer.embedding import CredentialRefusedError, Embedder
from kb_indexer.events import ArticleReindex
from kb_indexer.handler import PermanentError
from kb_indexer.resolver import SpaceEmbedding
from kb_indexer.store import Job

log = logging.getLogger(__name__)


class Storage(Protocol):
    """The database, as the pipeline uses it."""

    def job(self, version_id: int) -> Job | None:
        """Read the version to index."""
        ...

    def mark_indexing(self, article_id: int) -> None:
        """Record that the article is being indexed."""
        ...

    def mark_failed(self, article_id: int) -> None:
        """Record that the article was not indexed."""
        ...

    def write_chunks(self, version_id: int, contents: list[str]) -> list[int]:
        """Replace the chunks of a version."""
        ...

    def embedded(self, model_id: int, chunk_ids: list[int]) -> set[int]:
        """Which of the chunks already carry a vector of the model."""
        ...

    def write_embeddings(self, model_id: int, vectors: list[tuple[int, list[float]]]) -> None:
        """Store the vectors of chunks under the model."""
        ...

    def publish(self, article_id: int, version_id: int, version_number: int) -> bool:
        """Point the article at the version and drop what is no longer published."""
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


class IndexingHandler:
    """Turns an article version into chunks, embeds them and publishes it."""

    def __init__(
        self,
        store: Storage,
        resolver: Resolving,
        providers: Providing,
        policy: chunking.Policy = chunking.DEFAULT,
    ) -> None:
        self._store = store
        self._resolver = resolver
        self._providers = providers
        self._policy = policy

    def handle(self, event: ArticleReindex) -> None:
        """Re-index one article version."""
        job = self._store.job(event.version_id)
        if job is None or job.article_deleted:
            log.info("nothing to index", extra={**event.as_fields(), "reason": _absent(job)})

            return

        self._accept(event, job)
        self._store.mark_indexing(job.article_id)

        model = self._resolver.resolve(job.space_id)
        embedder = None if model is None else self._providers.for_provider(model.provider)

        chunks = chunking.split(job.body_markdown, title=job.subject, policy=self._policy)
        written = self._store.write_chunks(job.version_id, chunks)
        embedded = 0 if model is None or embedder is None else self._embed(job, model, embedder, written, chunks)
        published = self._store.publish(job.article_id, job.version_id, job.version_number)

        log.info(
            "article indexed",
            extra={
                **event.as_fields(),
                "chunks": len(chunks),
                "embedded": embedded,
                "model_id": 0 if model is None else model.model_id,
                "published": published,
            },
        )

    def give_up(self, event: ArticleReindex) -> None:
        """Record that the article will not be indexed from this event."""
        self._store.mark_failed(event.article_id)

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
            try:
                vectors = embedder.embed(model, [content for _chunk_id, content in batch])
            except CredentialRefusedError:
                # The stored credential changed under us: ask kb-api again
                # rather than wait the cache out.
                self._resolver.forget(job.space_id)

                raise

            # After every batch, so a failure halfway keeps what it paid for.
            self._store.write_embeddings(
                model.model_id,
                [(chunk_id, vector) for (chunk_id, _content), vector in zip(batch, vectors, strict=True)],
            )
            written += len(batch)

        return written

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


def _absent(job: Job | None) -> str:
    return "the version is gone" if job is None else "the article is deleted"
