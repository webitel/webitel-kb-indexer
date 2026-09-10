from typing import ClassVar, cast

import psycopg
import pytest
from psycopg import Connection
from psycopg.rows import TupleRow
from psycopg_pool import ConnectionPool

from kb_indexer import store as store_module
from kb_indexer.handler import PermanentError, TransientError
from kb_indexer.store import FAILED, INDEXED, INDEXING, Job, Store, connect, vector_literal

JOB = Job(
    version_id=7,
    article_id=3,
    space_id=5,
    domain_id=1,
    version_number=2,
    subject="Скидання пароля",
    body_markdown="# Скидання\n\nТекст.",
    article_deleted=False,
    chunking_strategy="recursive_markdown",
)


def compact(sql):
    """The statement as one line, so a test reads the way the code does."""
    return " ".join(sql.split())


class FakeCursor:
    """Records every statement and serves the rows prepared for it."""

    def __init__(self, results):
        self.calls = []
        self._results = list(results)
        self._rows = []

    def __enter__(self):
        return self

    def __exit__(self, *_exception):
        return False

    def execute(self, sql, params=None):
        self.calls.append((compact(sql), params))
        self._rows = self._results.pop(0) if self._results else []

    def executemany(self, sql, params_seq):
        self.calls.append((compact(sql), list(params_seq)))

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)

    def statements(self):
        return [sql for sql, _ in self.calls]


class FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.row_factories = []

    def cursor(self, row_factory=None):
        self.row_factories.append(row_factory)

        return self._cursor


class FakePool:
    """A pool that hands out one connection, or fails the way psycopg does."""

    def __init__(self, cursor, failure=None):
        self.connection_calls = 0
        self._cursor = cursor
        self._failure = failure

    def connection(self):
        self.connection_calls += 1

        return _Handing(self._cursor, self._failure)


class _Handing:
    def __init__(self, cursor, failure):
        self._cursor = cursor
        self._failure = failure

    def __enter__(self):
        if self._failure is not None:
            raise self._failure

        return FakeConnection(self._cursor)

    def __exit__(self, *_exception):
        return False


def build(results=(), failure=None):
    """A store over a pool that only pretends to be one."""
    cursor = FakeCursor(results)
    pool = cast(ConnectionPool[Connection[TupleRow]], FakePool(cursor, failure))

    return Store(pool), cursor


def test_the_job_is_read_by_version_alone():
    store, cursor = build(results=[[JOB]])

    assert store.job(7) == JOB

    sql, params = cursor.calls[0]
    assert params == {"version_id": 7}
    assert "SELECT v.id AS version_id, v.article_id, a.space_id, s.domain_id," in compact(sql)
    assert "JOIN kb.article a ON a.id = v.article_id" in sql
    assert "JOIN kb.space s ON s.id = a.space_id" in sql
    assert "WHERE v.id = %(version_id)s" in sql


def test_a_version_that_is_gone_reads_as_nothing():
    store, _cursor = build(results=[[]])

    assert store.job(7) is None


@pytest.mark.parametrize(
    ("mark", "state"),
    [
        ("mark_indexing", INDEXING),
        ("mark_failed", FAILED),
    ],
)
def test_the_state_of_the_article_is_recorded(mark, state):
    store, cursor = build()

    getattr(store, mark)(3)

    sql, params = cursor.calls[0]
    assert params == {"article_id": 3, "state": state}
    assert "UPDATE kb.article SET index_state = %(state)s WHERE id = %(article_id)s" in sql


def test_chunks_are_written_without_duplicating_what_is_there():
    store, cursor = build(results=[[], [(11, 1), (10, 0)], []])

    written = store.write_chunks(7, ["first", "second"])

    # The ids come back in reading order, whatever order the write returned.
    assert written == [10, 11]

    stale, upsert, tail = cursor.calls
    assert "DELETE FROM kb.chunk_embedding" in stale[0]
    assert "c.content IS DISTINCT FROM incoming.content" in stale[0]
    assert "ON CONFLICT (version_id, chunk_index) DO UPDATE SET content = EXCLUDED.content" in upsert[0]
    assert upsert[1] == {"version_id": 7, "contents": ["first", "second"]}
    assert "DELETE FROM kb.chunk WHERE version_id = %(version_id)s AND chunk_index >= %(count)s" in tail[0]
    assert tail[1] == {"version_id": 7, "count": 2}


def test_a_body_that_yields_nothing_leaves_no_chunks_behind():
    store, cursor = build()

    assert store.write_chunks(7, []) == []

    _stale, _upsert, tail = cursor.calls
    assert tail[1] == {"version_id": 7, "count": 0}


def test_the_chunks_that_already_carry_a_vector_are_named():
    store, cursor = build(results=[[(11,), (13,)]])

    assert store.embedded(4, [11, 12, 13]) == {11, 13}

    sql, params = cursor.calls[0]
    assert params == {"model_id": 4, "chunk_ids": [11, 12, 13]}
    assert "SELECT chunk_id FROM kb.chunk_embedding" in sql
    assert "WHERE model_id = %(model_id)s AND chunk_id = ANY(%(chunk_ids)s)" in sql


def test_a_vector_replaces_the_one_the_chunk_carried():
    store, cursor = build()

    store.write_embeddings(JOB, 4, [(11, [1.0, 0.0]), (12, [0.0, 1.0])])

    sql, rows = cursor.calls[0]
    assert "INSERT INTO kb.chunk_embedding (chunk_id, model_id, domain_id, space_id, embedding)" in sql
    assert "%(embedding)s::vector" in sql
    assert "ON CONFLICT (chunk_id, model_id) DO UPDATE" in sql
    assert rows == [
        {"chunk_id": 11, "model_id": 4, "domain_id": 1, "space_id": 5, "embedding": "[1.0,0.0]"},
        {"chunk_id": 12, "model_id": 4, "domain_id": 1, "space_id": 5, "embedding": "[0.0,1.0]"},
    ]


def test_the_vectors_of_one_job_are_written_in_one_transaction():
    pool = FakePool(FakeCursor([]))
    store = Store(cast(ConnectionPool[Connection[TupleRow]], pool))

    store.write_embeddings(JOB, 4, [(11, [1.0]), (12, [0.0])])

    assert pool.connection_calls == 1


def test_a_vector_is_written_the_way_pgvector_reads_one():
    assert vector_literal([1.5, -0.25, 0.0]) == "[1.5,-0.25,0.0]"
    assert vector_literal([]) == "[]"


@pytest.mark.parametrize(
    ("pointer", "published"),
    [
        ([(7,)], True),
        ([(9,)], False),
        ([], False),
    ],
)
def test_publishing_reports_whether_this_version_is_the_published_one(pointer, published):
    store, _cursor = build(results=[pointer, []])

    assert store.publish(3, 7, 2) is published


def test_the_pointer_moves_only_forward_and_the_state_is_recorded_either_way():
    store, cursor = build(results=[[(7,)], []])

    store.publish(3, 7, 2)

    sql, params = cursor.calls[0]
    assert params == {"article_id": 3, "version_id": 7, "version_number": 2, "state": INDEXED}
    assert "%(version_number)s > (" in sql
    assert "ELSE a.published_version_id" in sql
    assert "index_state = %(state)s" in sql


def test_only_versions_older_than_the_published_one_lose_their_chunks():
    store, cursor = build(results=[[(7,)], []])

    store.publish(3, 7, 2)

    sql, params = cursor.calls[1]
    assert params == {"article_id": 3}
    assert "DELETE FROM kb.chunk c" in sql
    assert "v.version_number < (" in sql
    assert "a.published_version_id IS NOT NULL" in sql


def test_the_swap_and_the_cleanup_are_one_transaction():
    pool = FakePool(FakeCursor([[(7,)], []]))
    store = Store(cast(ConnectionPool[Connection[TupleRow]], pool))

    store.publish(3, 7, 2)

    assert pool.connection_calls == 1


def test_a_database_that_is_down_is_worth_another_attempt():
    store, _cursor = build(failure=psycopg.OperationalError("connection refused"))

    with pytest.raises(TransientError, match="unavailable"):
        store.job(7)


def test_a_refused_statement_is_not_worth_another_attempt():
    store, _cursor = build(failure=psycopg.ProgrammingError("relation does not exist"))

    with pytest.raises(PermanentError, match="refused"):
        store.job(7)


def test_the_pool_is_closed_when_the_process_leaves(monkeypatch):
    class FakeConnectionPool:
        instances: ClassVar[list["FakeConnectionPool"]] = []

        def __init__(self, dsn, **kwargs):
            self.dsn = dsn
            self.kwargs = kwargs
            self.opened = None
            self.closed = False
            FakeConnectionPool.instances.append(self)

        check_connection = staticmethod(lambda _connection: None)

        def open(self, wait):
            self.opened = wait

        def close(self):
            self.closed = True

    monkeypatch.setattr(store_module, "ConnectionPool", FakeConnectionPool)

    with connect("postgres://kb@db/webitel") as store:
        assert isinstance(store, Store)

    pool = FakeConnectionPool.instances[-1]
    assert pool.dsn == "postgres://kb@db/webitel"
    # Starting must not depend on the database being up.
    assert pool.opened is False
    assert pool.closed
