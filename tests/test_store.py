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
    model_id=4,
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
    assert "CASE WHEN s.vector_search_enabled THEN COALESCE(s.embedding_model_id, 0) ELSE 0 END AS model_id" in sql
    assert "JOIN kb.space s ON s.id = a.space_id" in sql
    assert "WHERE v.id = %(version_id)s" in sql


def test_a_version_that_is_gone_reads_as_nothing():
    store, _cursor = build(results=[[]])

    assert store.job(7) is None


LOCKED = [(1,)]
NEWEST = [(False,)]
SUPERSEDED = [(True,)]


@pytest.mark.parametrize(
    ("mark", "state"),
    [
        ("mark_indexing", INDEXING),
        ("mark_failed", FAILED),
    ],
)
def test_the_state_of_the_newest_version_is_recorded_under_the_article_lock(mark, state):
    store, cursor = build(results=[LOCKED, NEWEST, []])

    getattr(store, mark)(3, 7, 4)

    lock, newer, update = cursor.calls
    assert lock == ("SELECT 1 FROM kb.article WHERE id = %(article_id)s FOR UPDATE", {"article_id": 3})
    assert newer[1] == {"version_id": 7, "model_id": 4}
    assert update[1] == {"article_id": 3, "version_id": 7, "state": state, "indexed": INDEXED}
    assert "UPDATE kb.article SET index_state = %(state)s WHERE id = %(article_id)s" in update[0]
    # A repeat of an indexed version must not take the article back; a reindex
    # of the published version still moves it.
    assert "published_version_id IS DISTINCT FROM %(version_id)s OR index_state <> %(indexed)s" in update[0]


@pytest.mark.parametrize("mark", ["mark_indexing", "mark_failed"])
def test_a_superseded_version_leaves_the_state_to_the_newer_one(mark):
    store, cursor = build(results=[LOCKED, SUPERSEDED])

    getattr(store, mark)(3, 7, 4)

    assert len(cursor.calls) == 2
    assert not any(sql.startswith("UPDATE") for sql in cursor.statements())


def test_the_state_check_and_write_are_one_transaction():
    pool = FakePool(FakeCursor([LOCKED, NEWEST, []]))
    store = Store(cast(ConnectionPool[Connection[TupleRow]], pool))

    store.mark_failed(3, 7, 4)

    assert pool.connection_calls == 1


@pytest.mark.parametrize(("answer", "superseded"), [(SUPERSEDED, True), (NEWEST, False), ([], False)])
def test_a_version_is_superseded_by_any_newer_version_of_its_article(answer, superseded):
    store, cursor = build(results=[answer])

    assert store.superseded(7, 4) is superseded

    sql, params = cursor.calls[0]
    assert params == {"version_id": 7, "model_id": 4}
    assert "JOIN kb.article_version newer" in sql
    assert "newer.article_id = mine.article_id AND newer.version_number > mine.version_number" in sql
    assert "WHERE mine.id = %(version_id)s" in sql


def test_a_version_is_superseded_once_the_space_searches_with_another_model():
    store, cursor = build(results=[NEWEST])

    store.superseded(7, 4)

    sql, _params = cursor.calls[0]
    assert "OR EXISTS (" in sql
    assert "JOIN kb.space s ON s.id = a.space_id WHERE mine.id = %(version_id)s" in sql
    assert (
        "CASE WHEN s.vector_search_enabled THEN COALESCE(s.embedding_model_id, 0) ELSE 0 END <> %(model_id)s::bigint"
    ) in sql


def test_giving_up_checks_only_for_a_newer_version():
    store, cursor = build(results=[LOCKED, NEWEST, []])

    store.mark_failed(3, 7, None)

    _lock, newer, update = cursor.calls
    assert newer[1] == {"version_id": 7, "model_id": None}
    assert "%(model_id)s::bigint IS NOT NULL" in newer[0]
    assert update[1]["state"] == FAILED


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


def test_chunks_carry_a_lexical_vector_under_the_space_configuration():
    store, cursor = build(results=[[], [(11, 0)], []])

    store.write_chunks(7, ["оплата картой"])

    _stale, upsert, _tail = cursor.calls
    assert "INSERT INTO kb.chunk (version_id, chunk_index, content, tsv)" in upsert[0]
    assert "to_tsvector(config.text_search_config::regconfig, incoming.content)" in upsert[0]
    assert "JOIN kb.space s ON s.id = a.space_id WHERE v.id = %(version_id)s" in upsert[0]
    assert "DO UPDATE SET content = EXCLUDED.content, tsv = EXCLUDED.tsv" in upsert[0]


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


@pytest.mark.parametrize("dimensions", [768, 1024])
def test_a_vector_is_written_once_in_the_column_of_its_size(dimensions):
    store, cursor = build()

    store.write_embeddings(JOB, 4, dimensions, [(11, [1.0, 0.0]), (12, [0.0, 1.0])])

    column = f"embedding_{dimensions}"
    sql, rows = cursor.calls[0]
    assert f"INSERT INTO kb.chunk_embedding (chunk_id, model_id, domain_id, space_id, {column})" in sql
    assert "%(embedding)s::vector" in sql
    assert "ON CONFLICT (chunk_id, model_id) DO NOTHING" in sql
    assert rows == [
        {"chunk_id": 11, "model_id": 4, "domain_id": 1, "space_id": 5, "embedding": "[1.0,0.0]"},
        {"chunk_id": 12, "model_id": 4, "domain_id": 1, "space_id": 5, "embedding": "[0.0,1.0]"},
    ]


def test_the_vectors_of_one_job_are_written_in_one_transaction():
    pool = FakePool(FakeCursor([]))
    store = Store(cast(ConnectionPool[Connection[TupleRow]], pool))

    store.write_embeddings(JOB, 4, 768, [(11, [1.0]), (12, [0.0])])

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
    store, _cursor = build(results=[LOCKED, NEWEST, pointer, [], []])

    assert store.publish(3, 7, 2, 4) is published


@pytest.mark.parametrize(("answer", "newest"), [(NEWEST, True), (SUPERSEDED, False)])
def test_the_pointer_moves_only_forward_and_only_the_newest_version_records_the_state(answer, newest):
    store, cursor = build(results=[LOCKED, answer, [(7,)], [], []])

    store.publish(3, 7, 2, 4)

    lock, newer, update, _retention, _foreign = cursor.calls
    assert "FOR UPDATE" in lock[0]
    assert newer[1] == {"version_id": 7, "model_id": 4}
    sql, params = update
    assert params == {"article_id": 3, "version_id": 7, "version_number": 2, "state": INDEXED, "newest": newest}
    assert "%(version_number)s > (" in sql
    assert "ELSE a.published_version_id" in sql
    assert "index_state = CASE WHEN %(newest)s THEN %(state)s ELSE a.index_state END" in sql


def test_only_versions_older_than_the_published_one_lose_their_chunks():
    store, cursor = build(results=[LOCKED, NEWEST, [(7,)], [], []])

    store.publish(3, 7, 2, 4)

    sql, params = cursor.calls[3]
    assert params == {"article_id": 3}
    assert "DELETE FROM kb.chunk c" in sql
    assert "v.version_number < (" in sql
    assert "a.published_version_id IS NOT NULL" in sql


def test_the_swap_and_the_cleanup_are_one_transaction():
    pool = FakePool(FakeCursor([LOCKED, NEWEST, [(7,)], [], []]))
    store = Store(cast(ConnectionPool[Connection[TupleRow]], pool))

    store.publish(3, 7, 2, 4)

    assert pool.connection_calls == 1


def test_pruning_drops_only_what_is_older_than_the_published_version():
    store, cursor = build()

    store.prune(3)

    (sql, params), _foreign = cursor.calls
    assert params == {"article_id": 3}
    assert "DELETE FROM kb.chunk c" in sql
    assert "v.version_number < (" in sql


@pytest.mark.parametrize(
    ("operation", "arguments", "position"),
    [("publish", (3, 7, 2, 4), 4), ("prune", (3,), 1)],
)
def test_the_vectors_of_a_model_the_space_no_longer_has_are_dropped(operation, arguments, position):
    store, cursor = build(results=[LOCKED, NEWEST, [(7,)], [], []])

    getattr(store, operation)(*arguments)

    assert len(cursor.calls) == position + 1
    sql, params = cursor.calls[position]
    assert params == {"article_id": 3}
    assert "DELETE FROM kb.chunk_embedding e" in sql
    assert "v.article_id = %(article_id)s" in sql
    assert "e.model_id IS DISTINCT FROM s.embedding_model_id" in sql


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
