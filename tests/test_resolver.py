import threading
from concurrent import futures
from contextlib import contextmanager
from typing import ClassVar

import grpc
import pytest

from kb_indexer import config, discovery
from kb_indexer.discovery import Instance
from kb_indexer.handler import PermanentError, TransientError
from kb_indexer.kbapi import indexing_pb2, indexing_pb2_grpc
from kb_indexer.resolver import (
    CHANNEL_OPTIONS,
    SERVICE_TOKEN_HEADER,
    Cached,
    Reconnecting,
    Resolver,
    SpaceEmbedding,
    connect,
)
from tests.conftest import TOKEN

ANSWER = indexing_pb2.SpaceEmbedding(
    vector_search_enabled=True,
    model_id=4,
    provider="gemini",
    model_ref="gemini-embedding-001",
    dimensions=768,
    endpoint="",
    api_key="s3cr3t-gemini-key",
    validated=True,
)


class FakeIndexing(indexing_pb2_grpc.IndexingServicer):
    """A kb-api that records the call and answers as it was told."""

    def __init__(self, answer=ANSWER, code=None):
        self.spaces = []
        self.metadata = {}
        self.remaining = []
        self._answer = answer
        self._code = code

    def ResolveSpaceEmbedding(self, request, context):  # noqa: N802
        self.spaces.append(request.space_id)
        self.metadata = dict(context.invocation_metadata())
        self.remaining.append(context.time_remaining())

        if self._code is not None:
            context.abort(self._code, "the call was refused")

        return self._answer


@contextmanager
def serving(servicer, timeout=5.0):
    """A resolver talking to a real server over a real channel."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    indexing_pb2_grpc.add_IndexingServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")

    try:
        yield Resolver(channel, TOKEN, timeout)
    finally:
        channel.close()
        server.stop(None)


def test_the_call_carries_the_service_token_under_its_own_key():
    kb_api = FakeIndexing()

    with serving(kb_api) as resolver:
        resolver.resolve(5)

    assert kb_api.spaces == [5]
    assert kb_api.metadata[SERVICE_TOKEN_HEADER] == TOKEN
    assert "x-webitel-access" not in kb_api.metadata


def test_the_answer_becomes_the_model():
    with serving(FakeIndexing()) as resolver:
        found = resolver.resolve(5)

    assert found == SpaceEmbedding(
        model_id=4,
        provider="gemini",
        model_ref="gemini-embedding-001",
        dimensions=768,
        endpoint="",
        validated=True,
        api_key="s3cr3t-gemini-key",
    )


def test_a_space_that_is_not_embedded_resolves_to_nothing():
    answer = indexing_pb2.SpaceEmbedding(vector_search_enabled=False)

    with serving(FakeIndexing(answer=answer)) as resolver:
        assert resolver.resolve(5) is None


def test_the_credential_stays_out_of_the_repr():
    with serving(FakeIndexing()) as resolver:
        found = resolver.resolve(5)

    assert "s3cr3t" not in repr(found)
    assert found.api_key == "s3cr3t-gemini-key"


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (grpc.StatusCode.UNAVAILABLE, TransientError),
        (grpc.StatusCode.DEADLINE_EXCEEDED, TransientError),
        (grpc.StatusCode.RESOURCE_EXHAUSTED, TransientError),
        (grpc.StatusCode.INTERNAL, TransientError),
        (grpc.StatusCode.UNKNOWN, TransientError),
        (grpc.StatusCode.UNAUTHENTICATED, PermanentError),
        (grpc.StatusCode.PERMISSION_DENIED, PermanentError),
        (grpc.StatusCode.NOT_FOUND, PermanentError),
        (grpc.StatusCode.ABORTED, PermanentError),
        (grpc.StatusCode.INVALID_ARGUMENT, PermanentError),
        (grpc.StatusCode.UNIMPLEMENTED, PermanentError),
    ],
)
def test_a_refusal_says_whether_another_attempt_is_worth_making(code, expected):
    with serving(FakeIndexing(code=code)) as resolver, pytest.raises(expected, match=code.name.lower()):
        resolver.resolve(5)


def test_a_kb_api_that_is_not_listening_is_worth_another_attempt():
    channel = grpc.insecure_channel("127.0.0.1:1")
    resolver = Resolver(channel, TOKEN, timeout=1.0)

    with pytest.raises(TransientError, match="unavailable"):
        resolver.resolve(5)

    channel.close()


def test_the_deadline_travels_with_the_call():
    kb_api = FakeIndexing()

    with serving(kb_api, timeout=2.0) as resolver:
        resolver.resolve(5)

    with serving(kb_api, timeout=30.0) as resolver:
        resolver.resolve(5)

    # The server sees a deadline, and it is the one the caller was configured
    # with rather than a default of the library.
    short, long = kb_api.remaining
    assert 1.0 < short < 3.0
    assert 29.0 < long < 31.0


def test_a_resolved_model_is_reused_for_the_length_of_the_ttl():
    kb_api = FakeIndexing()
    now = [1000.0]

    with serving(kb_api) as resolver:
        cached = Cached(resolver, ttl=300, clock=lambda: now[0])

        assert cached.resolve(5) == cached.resolve(5)

        now[0] += 299
        cached.resolve(5)
        assert kb_api.spaces == [5]

        now[0] += 2
        cached.resolve(5)

    assert kb_api.spaces == [5, 5]


def test_a_space_without_vector_search_is_remembered_too():
    kb_api = FakeIndexing(answer=indexing_pb2.SpaceEmbedding(vector_search_enabled=False))

    with serving(kb_api) as resolver:
        cached = Cached(resolver, ttl=300, clock=lambda: 1000.0)

        assert cached.resolve(5) is None
        assert cached.resolve(5) is None

    assert kb_api.spaces == [5]


def test_forgetting_a_space_asks_kb_api_again():
    kb_api = FakeIndexing()

    with serving(kb_api) as resolver:
        cached = Cached(resolver, ttl=300, clock=lambda: 1000.0)
        cached.resolve(5)
        cached.forget(5)
        cached.resolve(5)

    assert kb_api.spaces == [5, 5]


def test_forgetting_a_space_nobody_asked_about_is_harmless():
    with serving(FakeIndexing()) as resolver:
        Cached(resolver, ttl=300).forget(5)


def test_each_space_is_remembered_on_its_own():
    kb_api = FakeIndexing()

    with serving(kb_api) as resolver:
        cached = Cached(resolver, ttl=300, clock=lambda: 1000.0)
        cached.resolve(5)
        cached.resolve(6)
        cached.resolve(5)

    assert kb_api.spaces == [5, 6]


ADDRESS = "10.0.0.4:22106"


class Refused(grpc.RpcError):
    """A call kb-api never answered."""

    def __init__(self, code):
        self._code = code

    def code(self):
        return self._code

    def details(self):
        return "the call was refused"


class FakeChannel:
    made: ClassVar[list["FakeChannel"]] = []
    raises: ClassVar[Exception | None] = None

    def __init__(self, target, credentials=None, options=None):
        self.target = target
        self.credentials = credentials
        self.options = options
        self.closed = False
        FakeChannel.made.append(self)

    def unary_unary(self, *_args, **_kwargs):
        """Enough of a channel for a stub to be built on it."""

        def call(*_call, **_options):
            if self.closed:
                # What grpc itself raises for a call on a closed channel.
                raise ValueError("Cannot invoke RPC on closed channel!")

            if FakeChannel.raises is not None:
                raise FakeChannel.raises

            return ANSWER

        return call

    def close(self):
        self.closed = True


class Racing(Reconnecting):
    """A resolver whose channel is given up between opening it and the call."""

    def _open(self):
        resolver, channel = super()._open()
        self._drop(channel)

        return resolver, channel


@pytest.fixture
def channels(monkeypatch):
    FakeChannel.made = []
    FakeChannel.raises = None
    monkeypatch.setattr(grpc, "insecure_channel", FakeChannel)
    monkeypatch.setattr(grpc, "secure_channel", FakeChannel)

    return FakeChannel


@pytest.fixture
def looked_up(monkeypatch):
    """kb-api, as consul hands it over."""

    def found(*instances):
        asked = []

        def lookup(consul_addr, service, *_args, **_kwargs):
            asked.append((consul_addr, service))

            return list(instances)

        monkeypatch.setattr(discovery, "lookup", lookup)

        return asked

    return found


@pytest.mark.parametrize("tls", [False, True])
def test_the_channel_is_closed_when_the_process_leaves(complete_env, channels, looked_up, tls):
    asked = looked_up(Instance("10.0.0.4", 22106))
    complete_env.setenv("KB_API_TLS", str(tls).lower())
    settings = config.load(env_file=None)

    with connect(settings) as resolver:
        resolver.resolve(5)

    assert asked == [(settings.consul_addr, "webitel-kb")]

    channel = channels.made[-1]
    assert channel.target == "ipv4:10.0.0.4:22106"
    assert (channel.credentials is not None) is tls
    assert channel.options == CHANNEL_OPTIONS
    assert channel.closed


def test_nothing_is_looked_up_or_dialed_before_the_first_call(complete_env, channels, looked_up):
    asked = looked_up(Instance("10.0.0.4", 22106))

    with connect(config.load(env_file=None)):
        pass

    assert asked == []
    assert channels.made == []


def test_every_instance_consul_knows_takes_calls(complete_env, channels, looked_up):
    looked_up(Instance("10.0.0.4", 22106), Instance("10.0.0.5", 22106))

    with connect(config.load(env_file=None)) as resolver:
        resolver.resolve(5)

    assert channels.made[-1].target == "ipv4:10.0.0.4:22106,10.0.0.5:22106"


def test_the_channel_is_opened_once_and_reused(channels):
    reconnecting = Reconnecting(lambda: ADDRESS, False, TOKEN, 5.0)

    reconnecting.resolve(5)
    reconnecting.resolve(6)
    reconnecting.close()

    assert len(channels.made) == 1


def test_a_failed_call_gives_up_the_channel_so_the_next_one_is_looked_up_again(channels):
    targets = iter(["10.0.0.4:22106", "10.0.0.5:22106"])
    reconnecting = Reconnecting(lambda: next(targets), False, TOKEN, 5.0)

    channels.raises = Refused(grpc.StatusCode.UNAVAILABLE)
    with pytest.raises(TransientError):
        reconnecting.resolve(5)

    channels.raises = None
    reconnecting.resolve(5)
    reconnecting.close()

    assert [channel.target for channel in channels.made] == ["10.0.0.4:22106", "10.0.0.5:22106"]
    assert channels.made[0].closed


def test_a_refusal_that_no_attempt_can_fix_keeps_the_channel(channels):
    reconnecting = Reconnecting(lambda: ADDRESS, False, TOKEN, 5.0)

    channels.raises = Refused(grpc.StatusCode.PERMISSION_DENIED)
    with pytest.raises(PermanentError):
        reconnecting.resolve(5)

    channels.raises = None
    reconnecting.resolve(5)
    reconnecting.close()

    assert len(channels.made) == 1


def test_a_consul_that_cannot_be_asked_leaves_nothing_open(channels):
    def lookup():
        raise TransientError("consul did not answer")

    reconnecting = Reconnecting(lookup, False, TOKEN, 5.0)

    with pytest.raises(TransientError):
        reconnecting.resolve(5)

    reconnecting.close()

    assert channels.made == []


def test_a_channel_given_up_under_a_call_is_worth_another_attempt(channels):
    # Two deliveries overlap: one fails and closes the channel while the call
    # of the other is on its way to it.
    racing = Racing(lambda: ADDRESS, False, TOKEN, 5.0)

    with pytest.raises(TransientError, match="closed while the call was in flight"):
        racing.resolve(5)


def test_a_failure_that_is_not_the_channel_is_not_disguised(channels):
    reconnecting = Reconnecting(lambda: ADDRESS, False, TOKEN, 5.0)

    channels.raises = ValueError("the request could not be built")
    with pytest.raises(ValueError, match="could not be built"):
        reconnecting.resolve(5)

    reconnecting.close()


class Slow:
    """A kb-api whose answer is still in flight when the caller forgets it."""

    def __init__(self, answers, started, released):
        self.calls = 0
        self._answers = answers
        self._started = started
        self._released = released

    def resolve(self, space_id):
        self.calls += 1

        if self.calls == 1:
            self._started.set()
            self._released.wait(5)

        return self._answers[min(self.calls, len(self._answers)) - 1]


def credential(key):
    return SpaceEmbedding(4, "gemini", "gemini-embedding-001", 768, "", True, key)


def test_forgetting_a_space_beats_an_answer_that_is_still_in_flight():
    started, released = threading.Event(), threading.Event()
    kb_api = Slow([credential("old"), credential("new")], started, released)
    cached = Cached(kb_api, ttl=300)

    answered = []
    asking = threading.Thread(target=lambda: answered.append(cached.resolve(5)))
    asking.start()
    started.wait(5)

    cached.forget(5)
    released.set()
    asking.join()

    forgotten = cached.resolve(5)
    assert answered[0] is not None
    assert answered[0].api_key == "old"
    assert forgotten is not None
    assert forgotten.api_key == "new"
    assert kb_api.calls == 2


def test_forgetting_one_space_does_not_discard_the_answer_of_another():
    started, released = threading.Event(), threading.Event()
    kb_api = Slow([credential("first"), credential("second")], started, released)
    cached = Cached(kb_api, ttl=300)

    answered = []
    asking = threading.Thread(target=lambda: answered.append(cached.resolve(5)))
    asking.start()
    started.wait(5)

    cached.forget(6)
    released.set()
    asking.join()

    kept = cached.resolve(5)
    assert kept is not None
    assert kept.api_key == "first"
    assert kb_api.calls == 1
