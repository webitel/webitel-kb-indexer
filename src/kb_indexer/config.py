"""Configuration of the indexer, read from the environment."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit, urlunsplit

import pika
from psycopg import conninfo
from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LOG_LEVELS = frozenset({"debug", "info", "warn", "warning", "error"})

DEFAULT_ENV_FILE = ".env"

# Stands in for a value whose shape we cannot parse, so nothing leaks by default.
MASKED = "***"

# Fields the worker cannot run without, fields that may carry a password inside
# a url, and fields that are a secret whole.
REQUIRED = ("postgres_dsn", "pubsub_url", "consul_addr", "kb_api_service_token")
CREDENTIALS = ("postgres_dsn", "pubsub_url")
SECRETS = ("kb_api_service_token",)

# What kb-api itself refuses to run with. Checking it here names the variable
# rather than letting every call come back unauthenticated.
MIN_TOKEN_LENGTH = 32


class ConfigError(Exception):
    """The environment cannot produce a usable configuration."""


class Settings(BaseSettings):
    """The whole configuration of one process."""

    model_config = SettingsConfigDict(env_file_encoding="utf-8", extra="ignore")

    # The shared Webitel database. The indexer writes `kb.*` but owns no schema.
    postgres_dsn: str = ""
    # The broker carrying the re-indexing queue.
    pubsub_url: str = ""
    # Service discovery: where kb-api is looked up.
    consul_addr: str = ""

    # kb-api, which tells the worker how a space is embedded and hands over the
    # credential of that model, under the name it is registered with.
    kb_api_service: str = "webitel-kb"
    kb_api_service_token: str = ""
    kb_api_tls: bool = False
    kb_api_timeout: float = Field(default=5.0, gt=0)

    # How long a resolved model is reused, and how long one provider call may take.
    embedding_cache_ttl: float = Field(default=300.0, ge=0)
    embedding_timeout: float = Field(default=30.0, gt=0)

    # How stubborn the worker is with one message. The rest of the delivery
    # rules are contract, not configuration.
    consumer_retries: int = Field(default=5, ge=0)
    consumer_retry_backoff: float = Field(default=1.0, gt=0)
    consumer_shutdown_timeout: float = Field(default=30.0, gt=0)

    log_level: str = "info"
    log_json: bool = True
    log_console: bool = True
    log_file: str = ""
    log_otel: bool = False

    @field_validator("postgres_dsn")
    @classmethod
    def _usable_database_dsn(cls, value: str) -> str:
        if value.strip():
            try:
                conninfo.conninfo_to_dict(value)
            except Exception as exc:
                msg = f"unusable database dsn: {exc}"
                raise ValueError(msg) from exc

        return value

    @field_validator("kb_api_service_token")
    @classmethod
    def _long_enough_token(cls, value: str) -> str:
        if value.strip() and len(value) < MIN_TOKEN_LENGTH:
            msg = f"service token must be at least {MIN_TOKEN_LENGTH} characters"
            raise ValueError(msg)

        return value

    @field_validator("kb_api_service")
    @classmethod
    def _named_service(cls, value: str) -> str:
        if not value.strip():
            msg = "the name kb-api is registered under cannot be empty"
            raise ValueError(msg)

        return value

    @field_validator("pubsub_url")
    @classmethod
    def _usable_broker_url(cls, value: str) -> str:
        # Asking the client that will dial it, rather than restating its rules:
        # an unusable url must be reported here, not crash the running process.
        if value.strip():
            try:
                pika.URLParameters(value)
            except Exception as exc:
                msg = f"unusable broker url: {exc}"
                raise ValueError(msg) from exc

        return value

    @field_validator("log_level")
    @classmethod
    def _known_level(cls, value: str) -> str:
        level = value.strip().lower()
        if level not in LOG_LEVELS:
            msg = f"unknown level {value!r}, expected one of {', '.join(sorted(LOG_LEVELS))}"
            raise ValueError(msg)

        return level

    def require(self) -> None:
        """Fail on values the worker cannot run without."""
        missing = [name.upper() for name in REQUIRED if not str(getattr(self, name)).strip()]
        if missing:
            msg = f"configuration is incomplete, missing: {', '.join(missing)}"
            raise ConfigError(msg)

    def describe(self) -> dict[str, Any]:
        """The effective configuration under its variable names, credentials masked."""
        return {name.upper(): _shown(name, value) for name, value in self.model_dump().items()}


def load(env_file: str | None = DEFAULT_ENV_FILE) -> Settings:
    """Read the configuration and validate it once, at startup."""
    try:
        settings = Settings(_env_file=env_file)  # type: ignore[call-arg]
    except ValidationError as exc:
        raise ConfigError(_readable(exc)) from exc

    settings.require()

    return settings


def _shown(name: str, value: Any) -> Any:
    """One value of the dump: a secret whole, a password inside a url, or as it is."""
    if name in SECRETS:
        return MASKED if value else value

    if name in CREDENTIALS:
        return mask_url(value)

    return value


def mask_url(value: str) -> str:
    """Hide the password of a connection url, keeping the rest readable."""
    if not value:
        return value

    try:
        parts = urlsplit(value)
    except ValueError:
        return MASKED

    if not parts.netloc:
        return MASKED

    if not parts.password:
        return value

    userinfo = f"{parts.username or ''}:{MASKED}"
    netloc = f"{userinfo}@{parts.hostname or ''}"
    if parts.port:
        netloc = f"{netloc}:{parts.port}"

    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def _readable(exc: ValidationError) -> str:
    """Turn a pydantic error into one line naming the variables at fault."""
    problems = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "configuration"
        problems.append(f"{location.upper()}: {error['msg']}")

    return "invalid configuration: " + "; ".join(problems)
