import json
from datetime import UTC, datetime, timedelta, timezone

import pytest

from kb_indexer import events


def envelope(**overrides):
    body = {
        "type": "article.reindex",
        "schema": 1,
        "occurred_at": "2026-07-27T10:30:00Z",
        "article_id": 7,
        "version_id": 19,
        "space_id": 3,
        "domain_id": 1,
    }
    body.update(overrides)

    return json.dumps(body).encode()


def test_a_complete_envelope_is_read():
    event = events.parse(envelope())

    assert event == events.ArticleReindex(
        occurred_at=datetime(2026, 7, 27, 10, 30, tzinfo=UTC),
        article_id=7,
        version_id=19,
        space_id=3,
        domain_id=1,
    )


def test_an_unknown_field_is_ignored():
    """Adding a field is additive and keeps the schema number."""
    event = events.parse(envelope(published_by=42))

    assert event.article_id == 7


def test_an_offset_is_normalised_to_utc():
    event = events.parse(envelope(occurred_at="2026-07-27T13:30:00+03:00"))

    assert event.occurred_at == datetime(2026, 7, 27, 10, 30, tzinfo=UTC)
    assert event.occurred_at.tzinfo == UTC


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        (b"", "json"),
        (b"\xff\xfe", "json"),
        (b"[1, 2]", "expected an object"),
        (b'"article.reindex"', "expected an object"),
    ],
)
def test_a_body_that_is_not_an_envelope_is_refused(body, reason):
    with pytest.raises(events.EnvelopeError, match=reason):
        events.parse(body)


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"type": "article.published"}, "unknown type"),
        ({"type": None}, "unknown type"),
        ({"schema": 2}, "newer than the supported 1"),
        ({"schema": 0}, "invalid schema"),
        ({"schema": "1"}, "invalid schema"),
        ({"schema": True}, "invalid schema"),
        ({"article_id": 0}, "invalid article_id"),
        ({"version_id": -1}, "invalid version_id"),
        ({"space_id": None}, "invalid space_id"),
        ({"domain_id": "1"}, "invalid domain_id"),
        ({"article_id": 7.5}, "invalid article_id"),
        ({"occurred_at": "2026-07-27T10:30:00"}, "no time zone"),
        ({"occurred_at": "yesterday"}, "invalid occurred_at"),
        ({"occurred_at": 1753612200}, "invalid occurred_at"),
    ],
)
def test_an_envelope_the_worker_must_not_act_on_is_refused(overrides, reason):
    with pytest.raises(events.EnvelopeError, match=reason):
        events.parse(envelope(**overrides))


@pytest.mark.parametrize("name", ["article_id", "version_id", "space_id", "domain_id", "occurred_at", "schema", "type"])
def test_a_missing_field_is_refused(name):
    body = json.loads(envelope())
    del body[name]

    with pytest.raises(events.EnvelopeError):
        events.parse(json.dumps(body).encode())


def test_the_identifiers_are_exposed_for_logs_and_labels():
    fields = events.parse(envelope()).as_fields()

    assert fields == {"article_id": 7, "version_id": 19, "space_id": 3, "domain_id": 1}


def test_a_far_offset_still_lands_on_the_right_instant():
    far = timezone(timedelta(hours=-11))
    moment = datetime(2026, 7, 26, 23, 30, tzinfo=far)

    event = events.parse(envelope(occurred_at=moment.isoformat()))

    assert event.occurred_at == datetime(2026, 7, 27, 10, 30, tzinfo=UTC)
