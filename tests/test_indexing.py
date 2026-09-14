from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

import pytest

from kb_indexer import chunking
from kb_indexer.embedding import CredentialRefusedError
from kb_indexer.events import ArticleReindex
from kb_indexer.handler import PermanentError, TransientError
from kb_indexer.indexing import IndexingHandler
from kb_indexer.resolver import SpaceEmbedding
from kb_indexer.store import Job
from tests.conftest import Recorded

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
    space_id=5,
    domain_id=1,
    version_number=2,
    subject="Скидання пароля",
    body_markdown="# Через портал\n\nВідкрийте профіль і натисніть «Забули пароль».",
    article_deleted=False,
    chunking_strategy=chunking.STRATEGY,
)

MODEL = SpaceEmbedding(
    model_id=4,
    provider="e5",
    model_ref="multilingual-e5-large",
    dimensions=3,
    endpoint="http://embed.local",
    validated=True,
    api_key="",
)


class FakeStore:
    """Records what the pipeline asked of the database."""

    def __init__(self, journal, job=JOB, stored=(), published=True):
        self.calls = journal
        self._job = job
        self._stored = set(stored)
        self._published = published

    def job(self, version_id):
        self.calls.append(("job", version_id))

        return self._job

    def mark_indexing(self, article_id):
        self.calls.append(("mark_indexing", article_id))

    def write_chunks(self, version_id, contents):
        self.calls.append(("write_chunks", version_id, contents))

        return [100 + index for index in range(len(contents))]

    def embedded(self, model_id, chunk_ids):
        self.calls.append(("embedded", model_id, chunk_ids))

        return set(self._stored)

    def write_embeddings(self, job, model_id, vectors):
        self.calls.append(("write_embeddings", job, model_id, vectors))

    def publish(self, article_id, version_id, version_number):
        self.calls.append(("publish", article_id, version_id, version_number))

        return self._published

    def mark_failed(self, article_id):
        self.calls.append(("mark_failed", article_id))


class FakeResolver:
    """Records what the pipeline asked of kb-api."""

    def __init__(self, journal, model=MODEL, failure=None):
        self.calls = journal
        self.forgotten = []
        self._model = model
        self._failure = failure

    def resolve(self, space_id):
        self.calls.append(("resolve", space_id))

        if self._failure is not None:
            raise self._failure

        return self._model

    def forget(self, space_id):
        self.calls.append(("forget", space_id))
        self.forgotten.append(space_id)


class FakeEmbedder:
    """A provider that answers with a vector describing the text."""

    def __init__(self, journal, batch=10, failure=None):
        self.batch = batch
        self.calls = journal
        self.texts = []
        self._failure = failure

    def embed(self, model, texts):
        self.calls.append(("embed", texts))
        self.texts.extend(texts)

        if self._failure is not None:
            raise self._failure

        return [[float(len(text)), 0.0, 0.0] for text in texts]


class FakeProviders:
    def __init__(self, journal, embedder, failure=None):
        self.calls = journal
        self._embedder = embedder
        self._failure = failure

    def for_provider(self, provider):
        self.calls.append(("for_provider", provider))

        if self._failure is not None:
            raise self._failure

        return self._embedder


class Pipeline:
    """The handler over fakes that all record into one journal."""

    def __init__(self, **built):
        self.journal: list[tuple[Any, ...]] = []
        self.recorded = built.get("recorded") or Recorded()
        self.store = FakeStore(
            self.journal,
            job=built.get("job", JOB),
            stored=built.get("stored", ()),
            published=built.get("published", True),
        )
        self.resolver = FakeResolver(
            self.journal,
            model=built.get("model", MODEL),
            failure=built.get("unresolvable"),
        )
        self.embedder = FakeEmbedder(self.journal, batch=built.get("batch", 10), failure=built.get("failure"))
        self.handler = IndexingHandler(
            self.store,
            self.resolver,
            FakeProviders(self.journal, self.embedder, failure=built.get("unsupported")),
            self.recorded.metrics,
            policy=built.get("policy", chunking.DEFAULT),
        )

    def handle(self, event=EVENT):
        self.handler.handle(event)

    def steps(self):
        return [entry[0] for entry in self.journal]

    def entry(self, name):
        return next(entry for entry in self.journal if entry[0] == name)

    def entries(self, name):
        return [entry for entry in self.journal if entry[0] == name]


def test_a_version_is_chunked_embedded_and_published():
    pipeline = Pipeline()

    pipeline.handle()

    assert pipeline.steps() == [
        "job",
        "mark_indexing",
        "resolve",
        "for_provider",
        "write_chunks",
        "embedded",
        "embed",
        "write_embeddings",
        "publish",
    ]
    assert pipeline.entry("job") == ("job", 7)
    assert pipeline.entry("publish") == ("publish", 3, 7, 2)


def test_the_space_is_resolved_before_anything_is_written():
    pipeline = Pipeline()

    pipeline.handle()

    steps = pipeline.steps()
    assert pipeline.entry("resolve") == ("resolve", 5)
    assert steps.index("resolve") < steps.index("write_chunks")


def test_a_kb_api_that_is_down_stops_the_job_before_it_writes():
    pipeline = Pipeline(unresolvable=TransientError("kb-api is unavailable"))

    with pytest.raises(TransientError):
        pipeline.handle()

    assert pipeline.steps() == ["job", "mark_indexing", "resolve"]


def test_the_chunks_carry_the_subject_and_the_headings():
    pipeline = Pipeline()

    pipeline.handle()

    _step, version_id, contents = pipeline.entry("write_chunks")
    assert version_id == 7
    assert contents == chunking.split(JOB.body_markdown, title=JOB.subject)
    assert contents[0].startswith("Скидання пароля / Через портал")


def test_the_vectors_are_stored_against_the_chunks_they_describe():
    pipeline = Pipeline()

    pipeline.handle()

    _step, _version_id, contents = pipeline.entry("write_chunks")
    _step, job, model_id, vectors = pipeline.entry("write_embeddings")
    assert (job.domain_id, job.space_id, model_id) == (1, 5, 4)
    assert [chunk_id for chunk_id, _vector in vectors] == [100 + index for index in range(len(contents))]
    assert [vector[0] for _chunk_id, vector in vectors] == [float(len(content)) for content in contents]
    assert pipeline.entry("for_provider") == ("for_provider", "e5")


def test_a_space_without_vector_search_is_indexed_without_vectors(caplog):
    pipeline = Pipeline(model=None)

    with caplog.at_level("INFO"):
        pipeline.handle()

    assert pipeline.steps() == ["job", "mark_indexing", "resolve", "write_chunks", "publish"]
    assert "for_provider" not in pipeline.steps()
    assert caplog.records[-1].embedded == 0
    assert caplog.records[-1].model_id == 0


def test_the_chunks_that_already_carry_a_vector_are_not_paid_for_twice():
    pipeline = Pipeline(stored=(100,))

    pipeline.handle()

    _step, _model_id, contents = pipeline.entry("write_chunks")
    _step, model_id, chunk_ids = pipeline.entry("embedded")
    assert (model_id, chunk_ids) == (4, [100 + index for index in range(len(contents))])
    assert pipeline.embedder.texts == contents[1:]


def test_a_repeat_of_a_delivery_calls_no_provider_at_all():
    pipeline = Pipeline(stored=(100, 101, 102, 103, 104))

    pipeline.handle()

    assert "embed" not in pipeline.steps()
    assert pipeline.embedder.texts == []


def test_every_batch_is_stored_before_the_next_one_is_asked_for():
    long_body = "\n\n".join(f"## Розділ {number}\n\nТекст розділу." for number in range(6))
    pipeline = Pipeline(job=replace(JOB, body_markdown=long_body), batch=2)

    pipeline.handle()

    calls = [entry[0] for entry in pipeline.journal if entry[0] in {"embed", "write_embeddings"}]
    assert calls == ["embed", "write_embeddings"] * 3
    assert [len(entry[1]) for entry in pipeline.entries("embed")] == [2, 2, 2]


def test_a_refused_credential_is_forgotten_and_the_delivery_is_retried():
    pipeline = Pipeline(failure=CredentialRefusedError("the provider returned status 401"))

    with pytest.raises(CredentialRefusedError):
        pipeline.handle()

    assert pipeline.resolver.forgotten == [5]
    assert "write_embeddings" not in pipeline.steps()


def test_a_provider_that_fails_otherwise_leaves_the_resolved_model_alone():
    pipeline = Pipeline(failure=TransientError("the provider returned status 503"))

    with pytest.raises(TransientError):
        pipeline.handle()

    assert pipeline.resolver.forgotten == []


def test_a_body_that_yields_nothing_still_replaces_what_is_stored():
    pipeline = Pipeline(job=replace(JOB, body_markdown=""))

    pipeline.handle()

    assert pipeline.entry("write_chunks") == ("write_chunks", 7, [])
    # Nothing to embed means nothing to ask the database about either.
    assert "embedded" not in pipeline.steps()
    assert "embed" not in pipeline.steps()


def test_a_provider_this_worker_cannot_call_is_refused_before_anything_is_written():
    pipeline = Pipeline(unsupported=PermanentError("unsupported embedding provider 'openai'"))

    with pytest.raises(PermanentError, match="unsupported embedding provider"):
        pipeline.handle()

    assert pipeline.steps() == ["job", "mark_indexing", "resolve", "for_provider"]


@pytest.mark.parametrize(
    ("job", "reason"),
    [
        (None, "the version is gone"),
        (replace(JOB, article_deleted=True), "the article is deleted"),
    ],
)
def test_there_is_nothing_to_index(job, reason, caplog):
    pipeline = Pipeline(job=job)

    with caplog.at_level("INFO"):
        pipeline.handle()

    assert pipeline.steps() == ["job"]
    assert reason in caplog.records[-1].reason


@pytest.mark.parametrize(
    ("job", "message"),
    [
        (replace(JOB, article_id=9), "belongs to article 9"),
        (replace(JOB, chunking_strategy="semantic"), "unsupported chunking strategy 'semantic'"),
    ],
)
def test_a_job_this_worker_must_not_run_is_refused_before_anything_is_written(job, message):
    pipeline = Pipeline(job=job)

    with pytest.raises(PermanentError, match=message):
        pipeline.handle()

    assert pipeline.steps() == ["job"]


def test_a_late_job_finishes_without_moving_the_pointer(caplog):
    pipeline = Pipeline(published=False)

    with caplog.at_level("INFO"):
        pipeline.handle()

    assert pipeline.entry("publish") == ("publish", 3, 7, 2)
    assert caplog.records[-1].published is False


def test_the_lag_is_recorded_from_the_edit_once_the_version_is_published():
    pipeline = Pipeline()

    pipeline.handle()

    [(attributes, count)] = pipeline.recorded.points("kb_reindex_lag_seconds")
    assert (attributes, count) == ({"embedded": True}, 1)


def test_a_space_without_vector_search_reports_its_lag_apart():
    pipeline = Pipeline(model=None)

    pipeline.handle()

    assert pipeline.recorded.points("kb_reindex_lag_seconds") == [({"embedded": False}, 1)]


@pytest.mark.parametrize(
    "built",
    [
        {"published": False},
        {"job": None},
        {"job": replace(JOB, article_deleted=True)},
    ],
    ids=["late job", "version gone", "article deleted"],
)
def test_a_job_that_made_nothing_searchable_has_no_lag(built):
    pipeline = Pipeline(**built)

    pipeline.handle()

    assert pipeline.recorded.points("kb_reindex_lag_seconds") == []


def test_every_call_to_the_provider_is_timed_under_its_model():
    long_body = "\n\n".join(f"## Розділ {number}\n\nТекст розділу." for number in range(6))
    pipeline = Pipeline(job=replace(JOB, body_markdown=long_body), batch=2)

    pipeline.handle()

    assert len(pipeline.entries("embed")) == 3
    assert pipeline.recorded.points("kb_embedding_duration_seconds") == [
        ({"provider": "e5", "model": "multilingual-e5-large", "outcome": "ok"}, 3),
    ]


def test_a_call_the_provider_refused_is_timed_as_an_error():
    pipeline = Pipeline(failure=TransientError("the provider returned status 503"))

    with pytest.raises(TransientError):
        pipeline.handle()

    assert pipeline.recorded.points("kb_embedding_duration_seconds") == [
        ({"provider": "e5", "model": "multilingual-e5-large", "outcome": "error"}, 1),
    ]


def test_giving_up_marks_the_article_failed():
    pipeline = Pipeline(job=None)

    pipeline.handler.give_up(EVENT)

    assert pipeline.journal == [("mark_failed", 3)]


def test_the_chunk_policy_is_the_one_it_was_given():
    pipeline = Pipeline(
        job=replace(JOB, body_markdown="абв. " * 200),
        policy=chunking.Policy(max_chars=200, overlap_chars=20),
    )

    pipeline.handle()

    _step, _version_id, contents = pipeline.entry("write_chunks")
    assert len(contents) > 1
    assert max(len(chunk) for chunk in contents) <= 200
