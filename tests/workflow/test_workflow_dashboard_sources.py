"""Where the dashboard reads scores, forecasts and the mailbox from."""

from pathlib import Path

from flowlib.dashboard import sources


def test_the_scoring_cache_is_read_in_place(tmp_path):
    cfg = {"SCORE_CACHE": str(tmp_path / "s"), "MAILBOX": ""}
    src = sources.Sources(cfg, [{"key": "germany"}])
    assert src.scores == tmp_path / "s" / "tables" / "checkpoint-scores.csv"
    assert src.dataset("germany") == tmp_path / "s" / "germany"
    assert src.mailbox is None


def test_the_mailbox_is_this_machine_s(tmp_path):
    cfg = {"SCORE_CACHE": "/x", "MAILBOX": str(tmp_path / "box.md")}
    assert sources.Sources(cfg).mailbox == tmp_path / "box.md"


def test_a_dataset_reads_its_scoring_unit_s_folder(tmp_path):
    # The German units cache their forecasts in germany-days.
    cfg = {"SCORE_CACHE": str(tmp_path / "s"), "MAILBOX": ""}
    src = sources.Sources(cfg, [{"key": "germany", "unit": "germany-days"}])
    assert src.dataset("germany") == tmp_path / "s" / "germany-days"


def test_the_german_dataset_is_the_german_units_folder():
    import yaml

    root = Path(__file__).parents[2]
    cfg = yaml.safe_load((root / "workflow" / "config.yaml").read_text())
    (germany,) = [
        d for d in cfg["dashboard"]["datasets"] if d["key"] == "germany"
    ]
    assert germany["unit"] == "germany-days"
