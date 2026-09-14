from datetime import UTC, datetime, timedelta

import pytest

EDITED = datetime(2026, 7, 27, 10, 30, tzinfo=UTC)


@pytest.mark.parametrize(
    ("now", "expected"),
    [
        (EDITED + timedelta(seconds=12.5), 12.5),
        (EDITED, 0.0),
        # The edit time is stamped by kb-api on another clock.
        (EDITED - timedelta(seconds=3), 0.0),
    ],
    ids=["later", "same moment", "clock behind"],
)
def test_the_lag_is_measured_from_the_edit_and_never_negative(recorded, now, expected):
    assert recorded.metrics.indexed(EDITED, embedded=True, now=now) == expected

    assert recorded.points("kb_reindex_lag_seconds") == [({"embedded": True}, 1)]


def test_the_lag_is_measured_against_the_present_by_default(recorded):
    lag = recorded.metrics.indexed(datetime.now(UTC) - timedelta(seconds=1), embedded=False)

    assert 1.0 <= lag < 5.0


@pytest.mark.parametrize(("ok", "outcome"), [(True, "ok"), (False, "error")])
def test_an_embedding_call_is_recorded_with_its_outcome(recorded, ok, outcome):
    recorded.metrics.embedding("gemini", "gemini-embedding-001", 0.42, ok=ok)

    assert recorded.points("kb_embedding_duration_seconds") == [
        ({"provider": "gemini", "model": "gemini-embedding-001", "outcome": outcome}, 1),
    ]


def test_failures_are_counted_by_reason(recorded):
    recorded.metrics.failed("permanent")
    recorded.metrics.failed("permanent")
    recorded.metrics.failed("envelope")

    counted = {attributes["reason"]: count for attributes, count in recorded.points("kb_reindex_failed_total")}
    assert counted == {"envelope": 1, "permanent": 2}


@pytest.mark.parametrize(
    ("readings", "expected"),
    [
        ([], ([], [])),
        ([(5, 2)], ([({}, 5)], [({}, 2)])),
        ([(5, 2), (0, 3)], ([({}, 0)], [({}, 3)])),
        ([(5, 2), None], ([], [])),
    ],
    ids=["never read", "one reading", "latest reading wins", "forgotten"],
)
def test_the_depth_is_the_latest_reading_or_nothing(recorded, readings, expected):
    for reading in readings:
        if reading is None:
            recorded.metrics.forget_depth()
        else:
            recorded.metrics.depth(*reading)

    assert (recorded.points("kb_reindex_queue_depth"), recorded.points("kb_reindex_dlq_depth")) == expected
