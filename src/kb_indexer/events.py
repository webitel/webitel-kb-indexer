"""The `article.reindex` envelope, as it arrives from the queue."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, TypeGuard

TYPE_ARTICLE_REINDEX = "article.reindex"

# The envelope schema this worker understands. A higher one carries meaning we
# would have to guess at, so it is refused rather than acted on.
SUPPORTED_SCHEMA = 1


class EnvelopeError(Exception):
    """The body is not an envelope this worker can act on."""


@dataclass(frozen=True, slots=True)
class ArticleReindex:
    """One article version to re-index.

    Identifiers only: the content is read from the database by `version_id`.
    """

    occurred_at: datetime
    article_id: int
    version_id: int
    space_id: int
    domain_id: int

    def as_fields(self) -> dict[str, int]:
        """The identifiers, for structured logs and metric labels."""
        return {
            "article_id": self.article_id,
            "version_id": self.version_id,
            "space_id": self.space_id,
            "domain_id": self.domain_id,
        }


def parse(body: bytes) -> ArticleReindex:
    """Read a wire envelope, refusing anything this worker must not act on.

    Unknown fields are ignored: adding one is an additive change that keeps the
    schema number.
    """
    try:
        fields = json.loads(body)
    except (UnicodeDecodeError, ValueError) as exc:
        msg = f"body is not json: {exc}"
        raise EnvelopeError(msg) from exc

    if not isinstance(fields, dict):
        msg = f"body is a {type(fields).__name__}, expected an object"
        raise EnvelopeError(msg)

    _known_kind(fields.get("type"))
    _supported_schema(fields.get("schema"))

    return ArticleReindex(
        occurred_at=_moment(fields.get("occurred_at")),
        article_id=_identifier(fields, "article_id"),
        version_id=_identifier(fields, "version_id"),
        space_id=_identifier(fields, "space_id"),
        domain_id=_identifier(fields, "domain_id"),
    )


def _known_kind(value: Any) -> None:
    if value != TYPE_ARTICLE_REINDEX:
        msg = f"unknown type {value!r}"
        raise EnvelopeError(msg)


def _supported_schema(value: Any) -> None:
    if not _whole(value) or value < 1:
        msg = f"invalid schema {value!r}"
        raise EnvelopeError(msg)

    if value > SUPPORTED_SCHEMA:
        msg = f"schema {value} is newer than the supported {SUPPORTED_SCHEMA}"
        raise EnvelopeError(msg)


def _identifier(fields: dict[str, Any], name: str) -> int:
    value = fields.get(name)
    if not _whole(value) or value <= 0:
        msg = f"invalid {name} {value!r}"
        raise EnvelopeError(msg)

    return int(value)


def _moment(value: Any) -> datetime:
    """Read the edit time. It starts the clock of the indexing lag."""
    if not isinstance(value, str):
        msg = f"invalid occurred_at {value!r}"
        raise EnvelopeError(msg)

    try:
        moment = datetime.fromisoformat(value)
    except ValueError as exc:
        msg = f"invalid occurred_at {value!r}: {exc}"
        raise EnvelopeError(msg) from exc

    if moment.tzinfo is None:
        msg = f"occurred_at {value!r} carries no time zone"
        raise EnvelopeError(msg)

    return moment.astimezone(UTC)


def _whole(value: Any) -> TypeGuard[int]:
    """A json number that is an integer."""
    return isinstance(value, int) and not isinstance(value, bool)
