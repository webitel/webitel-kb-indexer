"""Where kb-api is, as consul knows it."""

from __future__ import annotations

import ipaddress
import logging
from dataclasses import dataclass
from typing import Any

import httpx

from kb_indexer.handler import TransientError

log = logging.getLogger(__name__)

# Seconds one lookup may take. The agent is local, so it answers at once or it
# is down.
LOOKUP_TIMEOUT = 5.0

# What a service registers when it listens on every interface: the catalogue
# keeps the address as it was given, and no client can dial it.
WILDCARD_HOSTS = frozenset({"", "0.0.0.0", "::", "[::]"})  # noqa: S104 - matched, not bound


@dataclass(frozen=True, slots=True)
class Instance:
    """One healthy instance of a service."""

    host: str
    port: int


def lookup(consul_addr: str, service: str, timeout: float = LOOKUP_TIMEOUT) -> list[Instance]:
    """The healthy instances of the service."""
    try:
        response = httpx.get(
            f"{_base_url(consul_addr)}/v1/health/service/{service}",
            params={"passing": "true"},
            timeout=timeout,
        )
        response.raise_for_status()
        entries = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        msg = f"consul did not answer where {service} is: {exc}"
        raise TransientError(msg) from exc

    instances = [found for entry in entries if (found := _instance(entry)) is not None]
    if not instances:
        msg = f"consul knows no healthy instance of {service}"
        raise TransientError(msg)

    return instances


def target(instances: list[Instance]) -> str:
    """The grpc target of the instances."""
    if all(_literal(instance.host) for instance in instances):
        return "ipv4:" + ",".join(f"{instance.host}:{instance.port}" for instance in instances)

    first = instances[0]
    host = f"[{first.host}]" if ":" in first.host else first.host

    return f"dns:///{host}:{first.port}"


def _instance(entry: dict[str, Any]) -> Instance | None:
    """One catalogue entry, with an address a client can actually dial."""
    service = entry.get("Service") or {}
    node = entry.get("Node") or {}

    host = str(service.get("Address") or "").strip()
    if host in WILDCARD_HOSTS:
        host = str(node.get("Address") or "").strip()

    port = int(service.get("Port") or 0)
    if host in WILDCARD_HOSTS or port <= 0:
        log.warning("instance without a usable address", extra={"host": host, "port": port})

        return None

    return Instance(host, port)


def _literal(host: str) -> bool:
    """Whether the address is an ipv4 literal rather than a name."""
    try:
        ipaddress.IPv4Address(host)
    except ValueError:
        return False

    return True


def _base_url(addr: str) -> str:
    if addr.startswith(("http://", "https://")):
        return addr.rstrip("/")

    return f"http://{addr.rstrip('/')}"
