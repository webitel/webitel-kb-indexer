"""Log sinks and their format."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARNING,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}

# Attributes every LogRecord carries; anything else came from the call site.
_BUILTIN_FIELDS = frozenset(
    vars(logging.LogRecord(name="", level=0, pathname="", lineno=0, msg="", args=(), exc_info=None)),
) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    """One json object per line, with the field names the Webitel services use."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "time": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "msg": record.getMessage(),
            "logger": record.name,
        }
        payload.update(extras(record))

        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, default=str)


class ConsoleFormatter(logging.Formatter):
    """Human readable lines for a local run."""

    def __init__(self) -> None:
        super().__init__(fmt="%(asctime)s %(levelname)-5s %(name)s %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        fields = extras(record)
        if fields:
            line += " " + " ".join(f"{key}={value}" for key, value in fields.items())

        return line


def extras(record: logging.LogRecord) -> dict[str, Any]:
    """Structured fields passed through `extra=`."""
    return {key: value for key, value in vars(record).items() if key not in _BUILTIN_FIELDS}


def configure(*, level: str = "info", json_format: bool = True, console: bool = True, file: str = "") -> None:
    """Install the configured sinks on the root logger, replacing any previous ones.

    Called once at startup; the replaced handlers are closed so a reconfigured
    file sink cannot leak its descriptor.
    """
    if level not in _LEVELS:
        msg = f"unknown log level {level!r}, expected one of {', '.join(sorted(_LEVELS))}"
        raise ValueError(msg)

    formatter: logging.Formatter = JsonFormatter() if json_format else ConsoleFormatter()

    handlers: list[logging.Handler] = []
    if console:
        handlers.append(logging.StreamHandler(sys.stdout))
    if file:
        handlers.append(logging.FileHandler(Path(file), encoding="utf-8"))

    # A configuration with no sink at all would silence the process; stdout is
    # a better answer than silence.
    silenced = not handlers
    if silenced:
        handlers.append(logging.StreamHandler(sys.stdout))

    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()

    for handler in handlers:
        handler.setFormatter(formatter)
        root.addHandler(handler)

    root.setLevel(_LEVELS[level])

    if silenced:
        root.warning("no log sink configured, falling back to stdout")
