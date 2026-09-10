import json
import math

import httpx
import pytest

from kb_indexer.embedding import (
    ENDPOINT_BATCH,
    GEMINI_BATCH,
    CredentialRefusedError,
    Endpoint,
    Gemini,
    Providers,
    providers,
)
from kb_indexer.handler import PermanentError, TransientError
from kb_indexer.resolver import SpaceEmbedding

CLOUD = SpaceEmbedding(
    model_id=4,
    provider="gemini",
    model_ref="gemini-embedding-001",
    dimensions=3,
    endpoint="",
    validated=True,
    api_key="s3cr3t",
)

SELF_HOSTED = SpaceEmbedding(
    model_id=5,
    provider="e5",
    model_ref="multilingual-e5-large",
    dimensions=3,
    endpoint="http://embed.local/",
    validated=True,
    api_key="",
)


class Recorded:
    """A provider that records what it was asked and answers as it was told."""

    def __init__(self, answer, status=200):
        self.requests = []
        self._answer = answer
        self._status = status

    def __call__(self, request):
        self.requests.append(request)
        body = self._answer if isinstance(self._answer, str) else json.dumps(self._answer)

        return httpx.Response(self._status, content=body, headers={"content-type": "application/json"})


def client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler), timeout=1)


def sent(handler, index=0):
    return json.loads(handler.requests[index].content)


def test_gemini_is_called_the_way_kb_api_validates_a_model():
    handler = Recorded({"embeddings": [{"values": [1.0, 0.0, 0.0]}, {"values": [0.0, 1.0, 0.0]}]})

    vectors = Gemini(client(handler)).embed(CLOUD, ["перше", "друге"])

    assert vectors == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

    request = handler.requests[0]
    assert str(request.url).endswith("/v1beta/models/gemini-embedding-001:batchEmbedContents")
    assert request.headers["x-goog-api-key"] == "s3cr3t"

    body = sent(handler)
    assert [item["content"]["parts"][0]["text"] for item in body["requests"]] == ["перше", "друге"]
    assert body["requests"][0]["taskType"] == "RETRIEVAL_DOCUMENT"
    assert body["requests"][0]["outputDimensionality"] == 3
    assert body["requests"][0]["model"] == "models/gemini-embedding-001"


def test_a_model_ref_that_already_names_the_path_is_left_alone():
    handler = Recorded({"embeddings": [{"values": [1.0, 0.0, 0.0]}]})
    model = SpaceEmbedding(4, "gemini", "models/gemini-embedding-001", 3, "", True, "s3cr3t")

    Gemini(client(handler)).embed(model, ["текст"])

    assert sent(handler)["requests"][0]["model"] == "models/gemini-embedding-001"


def test_a_vector_shorter_than_the_native_one_is_scaled_to_unit_length():
    handler = Recorded({"embeddings": [{"values": [3.0, 4.0, 0.0]}]})

    vectors = Gemini(client(handler)).embed(CLOUD, ["текст"])

    assert vectors == [[0.6, 0.8, 0.0]]
    assert math.isclose(math.sqrt(sum(value * value for value in vectors[0])), 1.0)


def test_a_native_vector_is_taken_as_it_comes():
    native = SpaceEmbedding(4, "gemini", "gemini-embedding-001", 3072, "", True, "s3cr3t")
    handler = Recorded({"embeddings": [{"values": [3.0, 4.0] + [0.0] * 3070}]})

    vectors = Gemini(client(handler)).embed(native, ["текст"])

    assert vectors[0][:2] == [3.0, 4.0]


def test_a_zero_vector_survives_the_scaling():
    handler = Recorded({"embeddings": [{"values": [0.0, 0.0, 0.0]}]})

    assert Gemini(client(handler)).embed(CLOUD, ["текст"]) == [[0.0, 0.0, 0.0]]


def test_the_self_hosted_service_is_called_over_the_canonical_contract():
    handler = Recorded({"embeddings": [[1.0, 0.0, 0.0]]})

    vectors = Endpoint(client(handler)).embed(SELF_HOSTED, ["текст"])

    assert vectors == [[1.0, 0.0, 0.0]]
    assert str(handler.requests[0].url) == "http://embed.local/embed"
    assert sent(handler) == {
        "model": "multilingual-e5-large",
        "texts": ["текст"],
        "dimensions": 3,
        "task": "document",
    }


def test_an_endpoint_this_client_cannot_speak_to_is_refused_for_good():
    model = SpaceEmbedding(5, "e5", "multilingual-e5-large", 3, "ftp://embed.local", True, "")

    def unreachable(_request):
        raise httpx.UnsupportedProtocol("unsupported scheme")

    with pytest.raises(PermanentError, match="cannot be called"):
        Endpoint(client(unreachable)).embed(model, ["текст"])


def test_a_self_hosted_model_without_an_endpoint_cannot_be_served():
    model = SpaceEmbedding(5, "e5", "multilingual-e5-large", 3, "", True, "")
    handler = Recorded({"embeddings": []})

    with pytest.raises(PermanentError, match="no endpoint url"):
        Endpoint(client(handler)).embed(model, ["текст"])

    assert handler.requests == []


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, CredentialRefusedError),
        (403, CredentialRefusedError),
        (429, TransientError),
        (500, TransientError),
        (503, TransientError),
        (400, PermanentError),
        (404, PermanentError),
        (422, PermanentError),
    ],
)
def test_a_refusal_says_whether_another_attempt_is_worth_making(status, expected):
    handler = Recorded({"error": "no"}, status=status)

    with pytest.raises(expected, match=str(status)):
        Endpoint(client(handler)).embed(SELF_HOSTED, ["текст"])


def test_a_refused_credential_is_still_worth_another_attempt():
    assert issubclass(CredentialRefusedError, TransientError)


@pytest.mark.parametrize(
    "failure",
    [
        httpx.ConnectTimeout("too slow"),
        httpx.ReadTimeout("too slow"),
        httpx.ConnectError("refused"),
    ],
)
def test_a_provider_that_does_not_answer_is_worth_another_attempt(failure):
    def raising(_request):
        raise failure

    with pytest.raises(TransientError):
        Endpoint(client(raising)).embed(SELF_HOSTED, ["текст"])


@pytest.mark.parametrize(
    ("answer", "message"),
    [
        ({"embeddings": [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]}, "2 vectors for 1 texts"),
        ({"embeddings": [[1.0, 0.0]]}, "vector of 2 values, expected 3"),
        ({"embeddings": [[1.0, 0.0, float("nan")]]}, "not a number"),
        ({"embeddings": [["a", "b", "c"]]}, "not numeric"),
        ({"embeddings": [None]}, "where a vector was expected"),
        ({"embeddings": {}}, "without a embeddings array"),
        ({}, "without a embeddings array"),
        ("not json at all", "other than json"),
        ("[1, 2]", "list, expected an object"),
    ],
)
def test_an_answer_the_column_would_not_take_is_refused(answer, message):
    handler = Recorded(answer)

    with pytest.raises(PermanentError, match=message):
        Endpoint(client(handler)).embed(SELF_HOSTED, ["текст"])


def test_an_infinite_value_is_refused_as_well():
    handler = Recorded({"embeddings": [[1.0, 0.0, float("inf")]]})

    with pytest.raises(PermanentError, match="not a number"):
        Endpoint(client(handler)).embed(SELF_HOSTED, ["текст"])


@pytest.mark.parametrize(
    ("provider", "expected", "batch"),
    [
        ("gemini", Gemini, GEMINI_BATCH),
        ("bge-m3", Endpoint, ENDPOINT_BATCH),
        ("e5", Endpoint, ENDPOINT_BATCH),
        ("byom", Endpoint, ENDPOINT_BATCH),
    ],
)
def test_every_provider_kb_api_can_register_has_a_client(provider, expected, batch):
    embedder = Providers(client(Recorded({}))).for_provider(provider)

    assert isinstance(embedder, expected)
    assert embedder.batch == batch


@pytest.mark.parametrize("provider", ["openai", "cohere", "azure", "bge-reranker", ""])
def test_a_provider_this_worker_cannot_call_is_refused_for_good(provider):
    with pytest.raises(PermanentError, match="unsupported embedding provider"):
        Providers(client(Recorded({}))).for_provider(provider)


def test_the_client_is_closed_when_the_process_leaves():
    with providers(timeout=7) as built:
        embedder = built.for_provider("gemini")
        assert isinstance(embedder, Gemini)
        assert embedder._client.timeout.read == 7

    assert embedder._client.is_closed
