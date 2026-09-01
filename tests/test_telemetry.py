import logging

from kb_indexer.telemetry import shutdown_quietly, telemetry


def test_without_an_endpoint_the_block_still_runs(caplog):
    entered = False

    with caplog.at_level(logging.INFO), telemetry("svc", "1.0", export_logs=True):
        entered = True

    assert entered
    assert "telemetry disabled" in caplog.text


def test_a_failing_provider_does_not_abort_the_shutdown(caplog):
    def refuse():
        msg = "collector is gone"
        raise RuntimeError(msg)

    with caplog.at_level(logging.ERROR):
        shutdown_quietly("metrics", refuse)

    assert "collector is gone" in caplog.text
