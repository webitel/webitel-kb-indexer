from dataclasses import replace
from datetime import UTC, datetime

import pytest

from kb_indexer import chunking
from kb_indexer.events import ArticleReindex
from kb_indexer.handler import PermanentError
from kb_indexer.indexing import IndexingHandler
from kb_indexer.store import Job

EVENT = ArticleReindex(
    occurred_at=datetime(2026, 7, 27, 10, 30, tzinfo=UTC),
    article_id=3,
    version_id=7,
    space_id=5,
    domain_id=1,
)

JOB = Job(
    version_id=7,
    article_id=3,
    version_number=2,
    subject="Скидання пароля",
    body_markdown="# Через портал\n\nВідкрийте профіль і натисніть «Забули пароль».",
    article_deleted=False,
    chunking_strategy=chunking.STRATEGY,
)


class FakeStore:
    """Records what the pipeline asked of the database."""

    def __init__(self, job=JOB, published=True):
        self.calls = []
        self._job = job
        self._published = published

    def job(self, version_id):
        self.calls.append(("job", version_id))

        return self._job

    def mark_indexing(self, article_id):
        self.calls.append(("mark_indexing", article_id))

    def write_chunks(self, version_id, contents):
        self.calls.append(("write_chunks", version_id, contents))

        return list(range(len(contents)))

    def publish(self, article_id, version_id, version_number):
        self.calls.append(("publish", article_id, version_id, version_number))

        return self._published

    def mark_failed(self, article_id):
        self.calls.append(("mark_failed", article_id))

    def steps(self):
        return [call[0] for call in self.calls]


def test_a_version_is_chunked_and_published():
    store = FakeStore()

    IndexingHandler(store).handle(EVENT)

    assert store.steps() == ["job", "mark_indexing", "write_chunks", "publish"]
    assert store.calls[0] == ("job", 7)
    assert store.calls[3] == ("publish", 3, 7, 2)


def test_the_chunks_carry_the_subject_and_the_headings():
    store = FakeStore()

    IndexingHandler(store).handle(EVENT)

    _step, version_id, contents = store.calls[2]
    assert version_id == 7
    assert contents == chunking.split(JOB.body_markdown, title=JOB.subject)
    assert contents[0].startswith("Скидання пароля / Через портал")


def test_a_body_that_yields_nothing_still_replaces_what_is_stored():
    store = FakeStore(job=replace(JOB, body_markdown=""))

    IndexingHandler(store).handle(EVENT)

    assert store.calls[2] == ("write_chunks", 7, [])


@pytest.mark.parametrize(
    ("job", "reason"),
    [
        (None, "the version is gone"),
        (replace(JOB, article_deleted=True), "the article is deleted"),
    ],
)
def test_there_is_nothing_to_index(job, reason, caplog):
    store = FakeStore(job=job)

    with caplog.at_level("INFO"):
        IndexingHandler(store).handle(EVENT)

    assert store.steps() == ["job"]
    assert reason in caplog.records[-1].reason


@pytest.mark.parametrize(
    ("job", "message"),
    [
        (replace(JOB, article_id=9), "belongs to article 9"),
        (replace(JOB, chunking_strategy="semantic"), "unsupported chunking strategy 'semantic'"),
    ],
)
def test_a_job_this_worker_must_not_run_is_refused_before_anything_is_written(job, message):
    store = FakeStore(job=job)

    with pytest.raises(PermanentError, match=message):
        IndexingHandler(store).handle(EVENT)

    assert store.steps() == ["job"]


def test_a_late_job_finishes_without_moving_the_pointer(caplog):
    store = FakeStore(published=False)

    with caplog.at_level("INFO"):
        IndexingHandler(store).handle(EVENT)

    assert store.steps() == ["job", "mark_indexing", "write_chunks", "publish"]
    assert caplog.records[-1].published is False


def test_giving_up_marks_the_article_failed():
    store = FakeStore(job=None)

    IndexingHandler(store).give_up(EVENT)

    assert store.calls == [("mark_failed", 3)]


def test_the_chunk_policy_is_the_one_it_was_given():
    store = FakeStore(job=replace(JOB, body_markdown="абв. " * 200))

    IndexingHandler(store, policy=chunking.Policy(max_chars=200, overlap_chars=20)).handle(EVENT)

    _step, _version_id, contents = store.calls[2]
    assert len(contents) > 1
    assert max(len(chunk) for chunk in contents) <= 200
