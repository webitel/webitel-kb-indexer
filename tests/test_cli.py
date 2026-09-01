import json

from kb_indexer import cli


def test_config_prints_the_effective_configuration(complete_env, capsys):

    assert cli.main(["config"]) == 0

    described = json.loads(capsys.readouterr().out)
    assert described["POSTGRES_DSN"] == "postgres://kb:***@db:5432/webitel"


def test_an_incomplete_configuration_exits_with_the_config_code(capsys):
    assert cli.main(["run"]) == cli.EXIT_CONFIG
    assert "POSTGRES_DSN" in capsys.readouterr().err


def test_an_unusable_log_file_exits_with_the_config_code(complete_env, capsys):
    complete_env.setenv("LOG_FILE", "/definitely/not/writable/indexer.log")

    assert cli.main(["run"]) == cli.EXIT_CONFIG
    assert "log sinks" in capsys.readouterr().err
