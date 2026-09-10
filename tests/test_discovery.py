import httpx
import pytest

from kb_indexer import discovery
from kb_indexer.discovery import Instance
from kb_indexer.handler import TransientError

CONSUL = "consul:8500"
SERVICE = "webitel-kb"


def entry(address="10.0.0.4", port=22106, node="10.0.0.9"):
    """One entry of the health catalogue, trimmed to what the lookup reads."""
    return {"Node": {"Address": node}, "Service": {"Address": address, "Port": port}}


class FakeConsul:
    """A consul agent that records the query and answers as it was told."""

    def __init__(self, answer=None, status=200, raises=None):
        self.url = ""
        self.params = None
        self.timeout = None
        self._answer = answer if answer is not None else [entry()]
        self._status = status
        self._raises = raises

    def __call__(self, url, *, params=None, timeout=None):
        self.url, self.params, self.timeout = url, params, timeout

        if self._raises is not None:
            raise self._raises

        return httpx.Response(self._status, json=self._answer, request=httpx.Request("GET", url))


@pytest.fixture
def consul(monkeypatch):
    def answering(**kwargs):
        agent = FakeConsul(**kwargs)
        monkeypatch.setattr(httpx, "get", agent)

        return agent

    return answering


def test_the_lookup_asks_for_passing_instances_only(consul):
    agent = consul()

    discovery.lookup(CONSUL, SERVICE, timeout=2.0)

    assert agent.url == f"http://{CONSUL}/v1/health/service/{SERVICE}"
    assert agent.params == {"passing": "true"}
    assert agent.timeout == 2.0


@pytest.mark.parametrize("addr", [CONSUL, f"http://{CONSUL}", f"http://{CONSUL}/"])
def test_the_agent_is_reachable_however_its_address_is_written(consul, addr):
    agent = consul()

    discovery.lookup(addr, SERVICE)

    assert agent.url == f"http://{CONSUL}/v1/health/service/{SERVICE}"


def test_the_address_of_the_service_is_the_one_to_dial(consul):
    consul(answer=[entry(address="10.0.0.4", port=22106)])

    assert discovery.lookup(CONSUL, SERVICE) == [Instance("10.0.0.4", 22106)]


@pytest.mark.parametrize("bound", ["", "0.0.0.0", "::", "[::]"])
def test_a_service_listening_on_every_interface_is_dialed_at_its_node(consul, bound):
    consul(answer=[entry(address=bound, node="10.0.0.9")])

    assert discovery.lookup(CONSUL, SERVICE) == [Instance("10.0.0.9", 22106)]


def test_every_healthy_instance_is_returned(consul):
    consul(answer=[entry(address="10.0.0.4"), entry(address="10.0.0.5")])

    assert discovery.lookup(CONSUL, SERVICE) == [Instance("10.0.0.4", 22106), Instance("10.0.0.5", 22106)]


@pytest.mark.parametrize(
    "broken",
    [
        {"Node": {"Address": ""}, "Service": {"Address": "0.0.0.0", "Port": 22106}},
        {"Service": {"Address": "10.0.0.4", "Port": 0}},
        {},
    ],
)
def test_an_instance_without_a_usable_address_is_skipped(consul, broken):
    consul(answer=[broken, entry(address="10.0.0.4")])

    assert discovery.lookup(CONSUL, SERVICE) == [Instance("10.0.0.4", 22106)]


def test_a_service_nobody_registered_is_worth_another_attempt(consul):
    consul(answer=[])

    with pytest.raises(TransientError, match="no healthy instance"):
        discovery.lookup(CONSUL, SERVICE)


def test_a_consul_that_is_down_is_worth_another_attempt(consul):
    consul(raises=httpx.ConnectError("connection refused"))

    with pytest.raises(TransientError, match="consul did not answer"):
        discovery.lookup(CONSUL, SERVICE)


def test_a_consul_that_broke_is_worth_another_attempt(consul):
    consul(status=500)

    with pytest.raises(TransientError, match="consul did not answer"):
        discovery.lookup(CONSUL, SERVICE)


def test_literal_addresses_are_handed_over_as_a_set():
    instances = [Instance("10.0.0.4", 22106), Instance("10.0.0.5", 22107)]

    assert discovery.target(instances) == "ipv4:10.0.0.4:22106,10.0.0.5:22107"


def test_a_name_goes_through_the_resolver_of_grpc():
    assert discovery.target([Instance("kb-api.service.consul", 22106)]) == "dns:///kb-api.service.consul:22106"


def test_an_address_that_is_not_ipv4_keeps_its_brackets():
    assert discovery.target([Instance("::1", 22106)]) == "dns:///[::1]:22106"
