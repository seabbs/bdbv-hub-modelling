"""The German unit reads the workflow's fetched triangle; fetch writes it
where the readers look."""

import re
from pathlib import Path

ROOT = Path(__file__).parents[3]


def test_fetch_writes_to_the_data_directory_itself(monkeypatch):
    from nowcastgermany.cli import build_parser

    monkeypatch.setenv("BDBV_DATA_DIR", "/x")
    args = build_parser().parse_args(["fetch"])
    assert args.out == "/x/raw/germany"


def test_the_unit_reads_the_triangle_it_is_given(monkeypatch, tmp_path):
    from nowcastgermany.checkpoint import triangle_path
    from nowcastgermany.evaluation import load_germany_config

    config = load_germany_config(ROOT / "config" / "germany.toml")
    name = Path(config.triangle).name
    monkeypatch.delenv("GERMANY_TRIANGLE", raising=False)
    assert triangle_path(config) == Path("data/raw/germany") / name
    given = tmp_path / name
    monkeypatch.setenv("GERMANY_TRIANGLE", str(given))
    assert triangle_path(config) == given


def test_the_german_score_rules_take_the_fetched_triangle():
    text = (ROOT / "workflow" / "rules" / "score.smk").read_text()
    for rule in ("score_germany", "score_run_germany"):
        body = re.search(
            rf"^rule {rule}:\n(.*?)(?=^rule |\Z)", text, re.S | re.M
        )
        assert body, rule
        assert "germany=GERMANY_TRIANGLE" in body[1], rule


def test_every_command_reads_the_data_where_fetch_writes_it(monkeypatch):
    from nowcastgermany.cli import build_parser

    monkeypatch.setenv("BDBV_DATA_DIR", "/x")
    parser = build_parser()
    written = parser.parse_args(["fetch"]).out
    for command in ("backtest", "teams", "asof"):
        args = parser.parse_args([command])
        assert args.data == written, command
