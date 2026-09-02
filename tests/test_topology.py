from kb_indexer import topology


class Recorder:
    """A channel that only remembers how it was asked to declare."""

    def __init__(self):
        self.calls = []

    def exchange_declare(self, exchange, **arguments):
        self.calls.append(("exchange", exchange, arguments))

    def queue_declare(self, queue, **arguments):
        self.calls.append(("queue", queue, arguments))

    def queue_bind(self, queue, exchange, **arguments):
        self.calls.append(("bind", queue, exchange, arguments))


# The contract, spelled out: kb-api declares the same objects with the same
# properties, and a redeclare that differs fails the channel.
EXPECTED = [
    ("exchange", "kb.reindex", {"exchange_type": "topic", "durable": True, "auto_delete": False, "internal": False}),
    (
        "exchange",
        "kb.reindex.dlx",
        {"exchange_type": "fanout", "durable": True, "auto_delete": False, "internal": False},
    ),
    (
        "queue",
        "kb.reindex",
        {
            "durable": True,
            "exclusive": False,
            "auto_delete": False,
            "arguments": {"x-dead-letter-exchange": "kb.reindex.dlx"},
        },
    ),
    ("bind", "kb.reindex", "kb.reindex", {"routing_key": "#"}),
    ("queue", "kb.reindex.dlq", {"durable": True, "exclusive": False, "auto_delete": False}),
    ("bind", "kb.reindex.dlq", "kb.reindex.dlx", {"routing_key": ""}),
]


def test_the_declared_objects_are_the_ones_in_the_contract():
    channel = Recorder()

    topology.declare(channel)

    assert channel.calls == EXPECTED


def test_an_object_is_declared_before_it_is_bound():
    channel = Recorder()

    topology.declare(channel)

    declared = [name for kind, name, *_ in channel.calls if kind in {"exchange", "queue"}]
    for _kind, queue, exchange, _ in (call for call in channel.calls if call[0] == "bind"):
        assert queue in declared
        assert exchange in declared


def test_one_delivery_at_a_time():
    """Per-article order in v1 rests on this and on a single relay."""
    assert topology.PREFETCH == 1
