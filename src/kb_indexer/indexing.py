"""The pipeline: what one re-indexing job does with one article version."""

from __future__ import annotations

import logging
from typing import Protocol

from kb_indexer import chunking
from kb_indexer.events import ArticleReindex
from kb_indexer.handler import PermanentError
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

    def publish(self, article_id: int, version_id: int, version_number: int) -> bool:
        """Point the article at the version and drop what is no longer published."""
        ...


class IndexingHandler:
    """Turns an article version into chunks and publishes it."""

    def __init__(self, store: Storage, policy: chunking.Policy = chunking.DEFAULT) -> None:
        self._store = store
        self._policy = policy

    def handle(self, event: ArticleReindex) -> None:
        """Re-index one article version."""
        job = self._store.job(event.version_id)
        if job is None or job.article_deleted:
            log.info("nothing to index", extra={**event.as_fields(), "reason": _absent(job)})

            return

        self._accept(event, job)
        self._store.mark_indexing(job.article_id)

        chunks = chunking.split(job.body_markdown, title=job.subject, policy=self._policy)
        self._store.write_chunks(job.version_id, chunks)
        published = self._store.publish(job.article_id, job.version_id, job.version_number)

        log.info("article indexed", extra={**event.as_fields(), "chunks": len(chunks), "published": published})

    def give_up(self, event: ArticleReindex) -> None:
        """Record that the article will not be indexed from this event."""
        self._store.mark_failed(event.article_id)

    def _accept(self, event: ArticleReindex, job: Job) -> None:
        """Refuse a job no attempt of this worker can complete."""
        if job.article_id != event.article_id:
            msg = f"version {event.version_id} belongs to article {job.article_id}, not {event.article_id}"
            raise PermanentError(msg)

        if job.chunking_strategy != chunking.STRATEGY:
            msg = f"unsupported chunking strategy {job.chunking_strategy!r}"
            raise PermanentError(msg)


def _absent(job: Job | None) -> str:
    return "the version is gone" if job is None else "the article is deleted"
