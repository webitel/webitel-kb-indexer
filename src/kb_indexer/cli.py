"""Command line of the indexer."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from kb_indexer import SERVICE_NAME, SERVICE_VERSION, app, config
from kb_indexer.log import configure as configure_logging

EXIT_CONFIG = 2


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the `kb-indexer` command."""
    parser = argparse.ArgumentParser(prog="kb-indexer", description="Webitel Knowledge Base indexer")
    parser.add_argument("--version", action="version", version=f"{SERVICE_NAME} {SERVICE_VERSION}")

    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("run", help="run the worker")
    commands.add_parser("config", help="print the effective configuration with credentials masked")

    args = parser.parse_args(argv)

    try:
        settings = config.load()
    except config.ConfigError as exc:
        sys.stderr.write(f"{exc}\n")

        return EXIT_CONFIG

    if args.command == "config":
        sys.stdout.write(json.dumps(settings.describe(), indent=2) + "\n")

        return 0

    try:
        configure_logging(
            level=settings.log_level,
            json_format=settings.log_json,
            console=settings.log_console,
            file=settings.log_file,
        )
    except OSError as exc:
        # An unusable log sink is a configuration problem, not a crash.
        sys.stderr.write(f"cannot install the log sinks: {exc}\n")

        return EXIT_CONFIG

    return app.run(settings)
