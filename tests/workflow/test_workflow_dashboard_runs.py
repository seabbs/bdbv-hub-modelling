"""Dashboard runs and references come from configuration."""

import json

import pytest
from flowlib.dashboard import runs


def test_a_run_is_its_folder(tmp_path):
    r = runs.make_run("runa", tmp_path, live=True)
    assert (r.key, r.label, r.folder, r.live) == (
        "runa",
        "runa",
        tmp_path,
        True,
    )


def test_the_first_run_is_live(tmp_path):
    got = runs.load_runs([("a", tmp_path / "a"), ("b", tmp_path / "b")])
    assert [r.live for r in got] == [True, False]
    assert got[0].colour != got[1].colour


def test_a_bad_ref_spec_says_what_is_wrong():
    with pytest.raises(ValueError, match="name=VALUE"):
        runs.parse_reference("no-equals")
    with pytest.raises(ValueError, match="unknown"):
        runs.parse_reference("a=b,shade=3")


def test_a_checkpoint_label_maps_to_its_run(tmp_path):
    train = tmp_path / "train" / "n250"
    train.mkdir(parents=True)
    (train / "settings.json").write_text(json.dumps({"hash": "7d7c5d9f247e"}))
    which = runs.Classifier(runs.load_runs([("slots", tmp_path)]))
    assert which.run_of("slots-7d7c5d9f247e") == "slots"
    assert which.run_of("chronos-slots-7d7c5d9f247e") == "slots"
    assert which.run_of("7d7c5d9f247e") == "slots"
    assert which.run_of("ffffffffffff") is None


def test_references_match_rows():
    refs = runs.load_references(
        [
            {
                "key": "zero",
                "label": "zero-shot ragged",
                "colour": "#16a34a",
                "model": "^(?!randomwalknowcast$)(?!.*_complete$)",
                "checkpoint": "zero-shot",
            },
            {
                "key": "rw",
                "label": "randomwalknowcast",
                "colour": "#6b7280",
                "model": "^randomwalknowcast$",
            },
        ],
        ["complete=_complete$,checkpoint=zero-shot,colour=#9333ea"],
    )
    assert [r.key for r in refs] == ["zero", "rw", "complete"]
    which = runs.Classifier([], refs)
    row = {"model": "fm_chronos_daily_ragged_ft", "checkpoint": "zero-shot"}
    assert which.series(row) == "zero"
    row = {"model": "fm_chronos_daily_complete", "checkpoint": "zero-shot"}
    assert which.series(row) == "complete"
    row = {"model": "randomwalknowcast", "checkpoint": "abc"}
    assert which.series(row) == "rw"
