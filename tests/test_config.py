import pytest

from kb_indexer import config
from tests.conftest import AMQP, DSN


def test_load_reads_the_environment(complete_env):
    complete_env.setenv("LOG_LEVEL", "DEBUG")
    complete_env.setenv("LOG_JSON", "false")
    complete_env.setenv("CONSUL_ADDR", "consul:8500")

    settings = config.load(env_file=None)

    assert settings.log_level == "debug"
    assert settings.log_json is False
    assert settings.consul_addr == "consul:8500"


@pytest.mark.parametrize(
    ("present", "missing"),
    [
        (None, ["POSTGRES_DSN", "PUBSUB_URL"]),
        ("POSTGRES_DSN", ["PUBSUB_URL"]),
        ("PUBSUB_URL", ["POSTGRES_DSN"]),
    ],
)
def test_incomplete_configuration_names_every_missing_variable(monkeypatch, present, missing):
    if present:
        monkeypatch.setenv(present, {"POSTGRES_DSN": DSN, "PUBSUB_URL": AMQP}[present])

    with pytest.raises(config.ConfigError) as failure:
        config.load(env_file=None)

    message = str(failure.value)
    assert all(name in message for name in missing)
    if present:
        assert present not in message


def test_blank_value_counts_as_missing(complete_env):
    complete_env.setenv("PUBSUB_URL", "   ")

    with pytest.raises(config.ConfigError, match="PUBSUB_URL"):
        config.load(env_file=None)


def test_an_invalid_value_is_reported_under_its_variable_name(complete_env):
    complete_env.setenv("LOG_LEVEL", "chatty")

    with pytest.raises(config.ConfigError, match="LOG_LEVEL"):
        config.load(env_file=None)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("postgres://kb:secret@db:5432/webitel", "postgres://kb:***@db:5432/webitel"),
        ("postgres://db:5432/webitel", "postgres://db:5432/webitel"),
        # Shapes we cannot parse are replaced entirely rather than shown.
        ("host=db user=kb password=secret", "***"),
        ("postgres://[unclosed", "***"),
    ],
)
def test_mask_url_hides_only_the_password(value, expected):
    assert config.mask_url(value) == expected


def test_describe_reports_variable_names_and_no_credentials(monkeypatch):
    monkeypatch.setenv("POSTGRES_DSN", "postgres://[unclosed")
    monkeypatch.setenv("PUBSUB_URL", "amqp://webitel:secret@rabbit:5672/")

    described = config.load(env_file=None).describe()

    assert "secret" not in repr(described)
    assert described["POSTGRES_DSN"] == "***"
    assert described["LOG_LEVEL"] == "info"


@pytest.mark.parametrize(
    "value",
    [
        "rabbit-host:5672",
        "not a url",
        "amqp://rabbit:notaport/",
        "amqp://rabbit:5672/?heartbeat=often",
    ],
)
def test_a_broker_url_that_cannot_be_dialled_is_a_configuration_error(complete_env, value):
    """It must be reported at startup, not raised out of the running process."""
    complete_env.setenv("PUBSUB_URL", value)

    with pytest.raises(config.ConfigError, match="PUBSUB_URL"):
        config.load(env_file=None)
