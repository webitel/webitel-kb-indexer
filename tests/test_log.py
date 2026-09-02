import json
import logging

import pytest

from kb_indexer.log import JsonFormatter, configure


@pytest.fixture(autouse=True)
def restore_root_logger():
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    for handler in handlers:
        root.addHandler(handler)
    root.setLevel(level)


def record(**fields):
    entry = logging.LogRecord(
        name="kb_indexer.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="article indexed",
        args=(),
        exc_info=None,
    )
    entry.__dict__.update(fields)

    return entry


def test_json_line_carries_the_field_names_the_go_services_use():
    line = json.loads(JsonFormatter().format(record(article_id=7)))

    assert line["level"] == "INFO"
    assert line["msg"] == "article indexed"
    assert line["logger"] == "kb_indexer.test"
    assert line["time"].endswith("+00:00")
    assert line["article_id"] == 7


def test_configure_installs_the_requested_sinks(tmp_path):
    destination = tmp_path / "indexer.log"
    configure(level="debug", console=False, file=str(destination))

    logging.getLogger("kb_indexer.test").debug("hello", extra={"article_id": 7})

    written = json.loads(destination.read_text().strip())
    assert written["msg"] == "hello"
    assert written["article_id"] == 7
    assert logging.getLogger().level == logging.DEBUG


def test_a_configuration_without_a_sink_still_logs(capsys):
    configure(console=False, file="")

    logging.getLogger("kb_indexer.test").info("hello")

    assert "hello" in capsys.readouterr().out


def test_an_unknown_level_is_rejected_by_name():
    with pytest.raises(ValueError, match="unknown log level"):
        configure(level="chatty")


@pytest.mark.parametrize(
    ("level", "expected"),
    [("info", logging.WARNING), ("warn", logging.WARNING), ("debug", logging.NOTSET)],
)
def test_a_library_narrating_itself_is_quiet_unless_we_are_debugging(level, expected):
    narrator = logging.getLogger("pika")
    restore = narrator.level
    try:
        configure(level=level, console=False, file="")

        assert narrator.level == expected
    finally:
        narrator.setLevel(restore)
