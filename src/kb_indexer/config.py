"""Configuration of the indexer, read from the environment."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit, urlunsplit

from pydantic import ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LOG_LEVELS = frozenset({"debug", "info", "warn", "warning", "error"})

DEFAULT_ENV_FILE = ".env"

# Stands in for a value whose shape we cannot parse, so nothing leaks by default.
MASKED = "***"

# Fields the worker cannot run without, and fields that may carry a password.
REQUIRED = ("postgres_dsn", "pubsub_url")
CREDENTIALS = ("postgres_dsn", "pubsub_url")


class ConfigError(Exception):
    """The environment cannot produce a usable configuration."""


class Settings(BaseSettings):
    """The whole configuration of one process."""

    model_config = SettingsConfigDict(env_file_encoding="utf-8", extra="ignore")

    # The shared Webitel database. The indexer writes `kb.*` but owns no schema.
    postgres_dsn: str = ""
    # The broker carrying the re-indexing queue.
    pubsub_url: str = ""
    # Service discovery, used once the consumer registers the instance.
    consul_addr: str = ""

    log_level: str = "info"
    log_json: bool = True
    log_console: bool = True
    log_file: str = ""
    log_otel: bool = False

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
        return {
            name.upper(): mask_url(value) if name in CREDENTIALS else value for name, value in self.model_dump().items()
        }


def load(env_file: str | None = DEFAULT_ENV_FILE) -> Settings:
    """Read the configuration and validate it once, at startup."""
    try:
        settings = Settings(_env_file=env_file)  # type: ignore[call-arg]
    except ValidationError as exc:
        raise ConfigError(_readable(exc)) from exc

    settings.require()

    return settings


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
