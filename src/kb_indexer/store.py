"""The database side of indexing: what one job reads and what it writes."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import psycopg
from psycopg import Connection
from psycopg.rows import TupleRow, class_row
from psycopg_pool import ConnectionPool

from kb_indexer.handler import PermanentError, TransientError

log = logging.getLogger(__name__)

# `kb.article.index_state`, written by this worker. The codes are the contract.
INDEXING = 2
INDEXED = 3
FAILED = 4

# One delivery at a time, plus a spare: a broker reconnect leaves the previous
# delivery running while its redelivery starts.
POOL_MIN = 1
POOL_MAX = 2

JOB_SQL = """
SELECT v.id AS version_id,
       v.article_id,
       v.version_number,
       v.subject,
       v.body_markdown,
       a.deleted_at IS NOT NULL AS article_deleted,
       s.chunking_strategy
FROM kb.article_version v
JOIN kb.article a ON a.id = v.article_id
JOIN kb.space s ON s.id = a.space_id
WHERE v.id = %(version_id)s
"""

STATE_SQL = """
UPDATE kb.article SET index_state = %(state)s WHERE id = %(article_id)s
"""

# Chunks whose text changed leave their vectors describing the text that was.
DROP_STALE_EMBEDDINGS_SQL = """
DELETE FROM kb.chunk_embedding e
USING kb.chunk c, unnest(%(contents)s::text[]) WITH ORDINALITY AS incoming(content, position)
WHERE e.chunk_id = c.id
  AND c.version_id = %(version_id)s
  AND c.chunk_index = incoming.position - 1
  AND c.content IS DISTINCT FROM incoming.content
"""

# The chunk keeps its id across a repeat, so the vectors of unchanged chunks
# survive it.
WRITE_CHUNKS_SQL = """
INSERT INTO kb.chunk (version_id, chunk_index, content)
SELECT %(version_id)s, incoming.position - 1, incoming.content
FROM unnest(%(contents)s::text[]) WITH ORDINALITY AS incoming(content, position)
ON CONFLICT (version_id, chunk_index) DO UPDATE SET content = EXCLUDED.content
RETURNING id, chunk_index
"""

# A shorter body leaves the chunks of the longer one behind it.
DROP_TAIL_SQL = """
DELETE FROM kb.chunk WHERE version_id = %(version_id)s AND chunk_index >= %(count)s
"""

# The pointer moves only forward, so a job that arrives late cannot publish an
# older version; the state is recorded either way, or a late job would leave
# the article indexing for good.
PUBLISH_SQL = """
UPDATE kb.article a
SET published_version_id = CASE
        WHEN a.published_version_id IS NULL
          OR %(version_number)s > (
                SELECT published.version_number
                FROM kb.article_version published
                WHERE published.id = a.published_version_id
             )
        THEN %(version_id)s
        ELSE a.published_version_id
    END,
    index_state = %(state)s
WHERE a.id = %(article_id)s
RETURNING published_version_id
"""

# Only the published version keeps its chunks. Versions above it may belong to
# a job still running, and are left alone.
RETENTION_SQL = """
DELETE FROM kb.chunk c
USING kb.article_version v, kb.article a
WHERE c.version_id = v.id
  AND v.article_id = %(article_id)s
  AND a.id = %(article_id)s
  AND a.published_version_id IS NOT NULL
  AND v.version_number < (
        SELECT published.version_number
        FROM kb.article_version published
        WHERE published.id = a.published_version_id
      )
"""


@dataclass(frozen=True, slots=True)
class Job:
    """The article version to index, as the database holds it."""

    version_id: int
    article_id: int
    version_number: int
    subject: str
    body_markdown: str
    article_deleted: bool
    chunking_strategy: str


class Store:
    """The `kb.*` tables, as the pipeline uses them."""

    def __init__(self, pool: ConnectionPool[Connection[TupleRow]]) -> None:
        self._pool = pool

    def job(self, version_id: int) -> Job | None:
        """Read the version to index, or nothing when it is gone."""
        with self._connection() as connection, connection.cursor(row_factory=class_row(Job)) as cursor:
            cursor.execute(JOB_SQL, {"version_id": version_id})
            found: Job | None = cursor.fetchone()

        return found

    def mark_indexing(self, article_id: int) -> None:
        """Record that the article is being indexed."""
        self._mark(article_id, INDEXING)

    def mark_failed(self, article_id: int) -> None:
        """Record that the article was not indexed."""
        self._mark(article_id, FAILED)

    def write_chunks(self, version_id: int, contents: list[str]) -> list[int]:
        """Replace the chunks of a version, returning their ids in reading order."""
        arguments = {"version_id": version_id, "contents": contents}

        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(DROP_STALE_EMBEDDINGS_SQL, arguments)
            cursor.execute(WRITE_CHUNKS_SQL, arguments)
            written = cursor.fetchall()
            cursor.execute(DROP_TAIL_SQL, {"version_id": version_id, "count": len(contents)})

        return [chunk_id for chunk_id, _ in sorted(written, key=lambda row: row[1])]

    def publish(self, article_id: int, version_id: int, version_number: int) -> bool:
        """Point the article at the version and drop what is no longer published.

        Returns whether this version is the published one.
        """
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                PUBLISH_SQL,
                {
                    "article_id": article_id,
                    "version_id": version_id,
                    "version_number": version_number,
                    "state": INDEXED,
                },
            )
            published = cursor.fetchone()
            cursor.execute(RETENTION_SQL, {"article_id": article_id})

        return published is not None and published[0] == version_id

    def _mark(self, article_id: int, state: int) -> None:
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(STATE_SQL, {"article_id": article_id, "state": state})

    @contextmanager
    def _connection(self) -> Iterator[Connection[TupleRow]]:
        """One transaction, with the failure told apart from the bug."""
        try:
            with self._pool.connection() as connection:
                yield connection
        except psycopg.OperationalError as unavailable:
            msg = f"the database is unavailable: {unavailable}"
            raise TransientError(msg) from unavailable
        except psycopg.Error as refused:
            msg = f"the database refused the statement: {refused}"
            raise PermanentError(msg) from refused


@contextmanager
def connect(dsn: str) -> Iterator[Store]:
    """Hold the connection pool for as long as the process runs.

    The pool opens without dialing: a database that is down must not stop the
    worker from starting, it makes the deliveries fail and be retried.
    """
    pool = ConnectionPool(
        dsn,
        min_size=POOL_MIN,
        max_size=POOL_MAX,
        open=False,
        check=ConnectionPool.check_connection,
    )
    pool.open(wait=False)
    log.info("database pool opened", extra={"min_size": POOL_MIN, "max_size": POOL_MAX})

    try:
        yield Store(pool)
    finally:
        pool.close()
