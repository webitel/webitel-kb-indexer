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

    assert recorded.points("webitel.kb.article.index.duration") == [({"webitel.kb.article.index.embedded": True}, 1)]


def test_the_lag_is_measured_against_the_present_by_default(recorded):
    lag = recorded.metrics.indexed(datetime.now(UTC) - timedelta(seconds=1), embedded=False)

    assert 1.0 <= lag < 5.0


@pytest.mark.parametrize(
    ("error_type", "failure"),
    [(None, {}), ("TransientError", {"error.type": "TransientError"})],
    ids=["answered", "failed"],
)
def test_an_embedding_call_is_recorded_as_a_gen_ai_operation(recorded, error_type, failure):
    recorded.metrics.embedding("e5", "multilingual-e5-large", 0.42, error_type=error_type)

    assert recorded.points("gen_ai.client.operation.duration") == [
        (
            {
                "gen_ai.operation.name": "embeddings",
                "gen_ai.provider.name": "e5",
                "gen_ai.request.model": "multilingual-e5-large",
            }
            | failure,
            1,
        ),
    ]


def test_a_provider_known_to_the_gen_ai_conventions_is_reported_under_their_name(recorded):
    recorded.metrics.embedding("gemini", "gemini-embedding-001", 0.42)

    [(attributes, _)] = recorded.points("gen_ai.client.operation.duration")
    assert attributes["gen_ai.provider.name"] == "gcp.gemini"


def test_failures_are_counted_by_reason(recorded):
    recorded.metrics.failed("permanent")
    recorded.metrics.failed("permanent")
    recorded.metrics.failed("envelope")

    points = recorded.points("webitel.kb.article.index.job.failed")
    counted = {attributes["error.type"]: count for attributes, count in points}
    assert counted == {"envelope": 1, "permanent": 2}


@pytest.mark.parametrize(
    ("readings", "expected"),
    [
        ([], {}),
        ([(5, 2)], {"pending": 5, "failed": 2}),
        ([(5, 2), (0, 3)], {"pending": 0, "failed": 3}),
        ([(5, 2), None], {}),
    ],
    ids=["never read", "one reading", "latest reading wins", "forgotten"],
)
def test_the_depth_is_the_latest_reading_or_nothing(recorded, readings, expected):
    for reading in readings:
        if reading is None:
            recorded.metrics.forget_depth()
        else:
            recorded.metrics.depth(*reading)

    points = recorded.points("webitel.kb.article.index.job.count")
    assert {attributes["webitel.kb.article.index.state"]: count for attributes, count in points} == expected
