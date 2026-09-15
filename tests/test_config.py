import os
import pathlib

import pytest

from kb_indexer import config
from tests.conftest import AMQP, CONSUL, DSN, TOKEN


def test_load_reads_the_environment(complete_env):
    complete_env.setenv("LOG_LEVEL", "DEBUG")
    complete_env.setenv("LOG_JSON", "false")
    settings = config.load(env_file=None)

    assert settings.log_level == "debug"
    assert settings.log_json is False
    assert settings.consul_addr == CONSUL
    assert settings.kb_api_service == "webitel-kb"


@pytest.mark.parametrize(
    ("present", "missing"),
    [
        (None, ["POSTGRES_DSN", "PUBSUB_URL", "CONSUL_ADDR", "KB_API_SERVICE_TOKEN"]),
        ("POSTGRES_DSN", ["PUBSUB_URL", "CONSUL_ADDR", "KB_API_SERVICE_TOKEN"]),
        ("PUBSUB_URL", ["POSTGRES_DSN", "CONSUL_ADDR", "KB_API_SERVICE_TOKEN"]),
        ("CONSUL_ADDR", ["POSTGRES_DSN", "PUBSUB_URL", "KB_API_SERVICE_TOKEN"]),
        ("KB_API_SERVICE_TOKEN", ["POSTGRES_DSN", "PUBSUB_URL", "CONSUL_ADDR"]),
    ],
)
def test_incomplete_configuration_names_every_missing_variable(monkeypatch, present, missing):
    if present:
        values = {"POSTGRES_DSN": DSN, "PUBSUB_URL": AMQP, "CONSUL_ADDR": CONSUL, "KB_API_SERVICE_TOKEN": TOKEN}
        monkeypatch.setenv(present, values[present])

    with pytest.raises(config.ConfigError) as failure:
        config.load(env_file=None)

    message = str(failure.value)
    assert all(name in message for name in missing)
    if present:
        assert present not in message


def test_kb_api_cannot_be_looked_up_without_a_name(complete_env):
    complete_env.setenv("KB_API_SERVICE", "   ")

    with pytest.raises(config.ConfigError, match="KB_API_SERVICE"):
        config.load(env_file=None)


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
        ("", ""),
        ("otlpgrpc", "otlpgrpc"),
        (" OTLPHTTP ", "otlphttp"),
        ("none", "none"),
    ],
)
def test_the_exporter_is_read_under_the_name_the_go_services_use(complete_env, value, expected):
    complete_env.setenv("OTEL_METRICS_EXPORTER", value)
    complete_env.setenv("OTEL_LOGS_EXPORTER", value)

    settings = config.load(env_file=None)

    assert (settings.otel_metrics_exporter, settings.otel_logs_exporter) == (expected, expected)


def test_the_exporter_variables_of_the_env_file_reach_the_exporters(complete_env, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OTEL_METRICS_EXPORTER=otlpgrpc\n"
        "OTEL_EXPORTER_OTLP_ENDPOINT=http://collector:4317\n"
        "OTEL_EXPORTER_OTLP_HEADERS=from-the-file\n"
        "LOG_LEVEL=debug\n",
    )
    for name in ("OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_EXPORTER_OTLP_HEADERS"):
        complete_env.setenv(name, "")
        complete_env.delenv(name)
    complete_env.setenv("OTEL_EXPORTER_OTLP_HEADERS", "from-the-environment")

    settings = config.load(env_file=str(env_file))

    assert settings.otel_metrics_exporter == "otlpgrpc"
    assert os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://collector:4317"
    assert os.environ["OTEL_EXPORTER_OTLP_HEADERS"] == "from-the-environment"
    # Settings of the worker itself stay in the settings.
    assert settings.log_level == "debug"
    assert os.environ.get("LOG_LEVEL") is None


@pytest.mark.parametrize("variable", ["OTEL_METRICS_EXPORTER", "OTEL_LOGS_EXPORTER"])
def test_an_exporter_the_worker_does_not_have_is_a_configuration_error(complete_env, variable):
    # A typo must stop the start, not leave the process silently unobserved.
    complete_env.setenv(variable, "prometheus")

    with pytest.raises(config.ConfigError, match=variable):
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


def test_describe_reports_variable_names_and_no_credentials(complete_env):
    complete_env.setenv("POSTGRES_DSN", "host=db user=kb password=secret")
    complete_env.setenv("PUBSUB_URL", "amqp://webitel:secret@rabbit:5672/")

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


@pytest.mark.parametrize(
    "value",
    [
        "not a dsn",
        "postgres://[unclosed",
    ],
)
def test_a_database_dsn_that_cannot_be_dialled_is_a_configuration_error(complete_env, value):
    """It must be reported at startup, not raised out of the running process."""
    complete_env.setenv("POSTGRES_DSN", value)

    with pytest.raises(config.ConfigError, match="POSTGRES_DSN"):
        config.load(env_file=None)


def test_the_kb_api_settings_have_defaults_that_do_not_have_to_be_set(complete_env):
    settings = config.load(env_file=None)

    assert settings.kb_api_service == "webitel-kb"
    assert settings.kb_api_service_token == TOKEN
    assert settings.kb_api_tls is False
    assert (settings.kb_api_timeout, settings.embedding_timeout, settings.embedding_cache_ttl) == (5.0, 30.0, 300.0)


def test_a_token_kb_api_would_refuse_is_refused_here(complete_env):
    complete_env.setenv("KB_API_SERVICE_TOKEN", "too-short")

    with pytest.raises(config.ConfigError) as failure:
        config.load(env_file=None)

    assert "KB_API_SERVICE_TOKEN" in str(failure.value)
    assert "at least 32" in str(failure.value)


def test_the_service_token_is_never_described(complete_env):
    described = config.load(env_file=None).describe()

    assert described["KB_API_SERVICE_TOKEN"] == config.MASKED
    assert TOKEN not in str(described)


def test_every_setting_is_in_the_example_environment():
    example = pathlib.Path(".env.example").read_text()
    missing = [name.upper() for name in config.Settings.model_fields if name.upper() not in example]

    assert missing == []
