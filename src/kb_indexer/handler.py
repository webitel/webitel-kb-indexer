"""What the transport hands a delivery to, and how a failure is classified."""

from __future__ import annotations

from typing import Protocol

from kb_indexer.events import ArticleReindex


class TransientError(Exception):
    """A failure worth another attempt: a timeout, a rate limit, a dependency that is down."""


class PermanentError(Exception):
    """A failure no attempt can fix."""


class Handler(Protocol):
    """The indexing pipeline, as the transport sees it.

    Anything raised other than `TransientError` ends the delivery at once:
    classifying a failure is the pipeline's job, and a bug must not be retried
    five times before it is reported.
    """

    def handle(self, event: ArticleReindex) -> None:
        """Re-index one article version."""
        ...

    def give_up(self, event: ArticleReindex) -> None:
        """Record that the article will not be indexed from this event."""
        ...
