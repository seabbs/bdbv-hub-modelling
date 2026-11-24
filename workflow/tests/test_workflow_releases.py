"""Checkpoint labels from the repository's releases."""

from flowlib import releases

LISTING = [
    {"tagName": "chronos-ft-old", "publishedAt": "2026-11-10T00:00:00Z"},
    {"tagName": "chronos-ft-aaa", "publishedAt": "2026-11-16T12:00:00Z"},
    {"tagName": "chronos-slots-bbb", "publishedAt": "2026-11-16T15:00:00Z"},
    {"tagName": "chronos-slots-ccc", "publishedAt": "2026-11-16T10:00:00Z"},
    {"tagName": "sims-r90", "publishedAt": "2026-11-16T11:00:00Z"},
    {"tagName": "germany-ft-de", "publishedAt": "2026-11-16T11:00:00Z"},
]


def test_slots_checkpoints_first_then_oldest_first():
    got = releases.labels(LISTING, "2026-11-16T09:00:00Z")
    assert got == ["slots-ccc", "slots-bbb", "aaa"]


def test_earlier_releases_are_left_out():
    got = releases.labels(LISTING, "2026-11-01T00:00:00Z")
    assert got[-1] == "aaa"
    assert "old" in got


def test_a_batch_is_released_once_both_its_files_are():
    names = [
        "counts-30900.csv.gz",
        "settings-30900.csv",
        "counts-30901.csv.gz",
        "settings-30902.csv",
        "qc.csv",
    ]
    assert releases.released_seeds(names) == {30900}


def test_checkpoints_by_publication_time_across_kinds():
    got = releases.by_time(LISTING, "2026-11-16T09:00:00Z")
    assert got == ["slots-ccc", "aaa", "slots-bbb"]


def test_the_newest_checkpoint_has_the_highest_priority_within_its_band():
    order = ["slots-ccc", "aaa", "slots-bbb"]
    got = [releases.priority(order, label, 10) for label in order]
    assert got == sorted(got)
    assert all(10 < p < 11 for p in got)
    # Labels outside the order (the reference) sit at the band's base.
    assert releases.priority(order, "reference", 10) == 10


def test_only_r7_checkpoints_are_scored():
    # The shipped config scores only checkpoints released after
    # `published_from`.
    from pathlib import Path

    import yaml

    config = yaml.safe_load(
        (Path(__file__).parents[2] / "workflow" / "config.yaml").read_text()
    )["score"]
    assert config["labels"] is None
    last_r6 = {
        "tagName": "chronos-slots-deadbeef0002",
        "publishedAt": "2026-11-19T14:12:48Z",
    }
    assert releases.labels([*LISTING, last_r6], config["published_from"]) == []


def test_a_release_s_batches_and_none_before_it_is_made(monkeypatch):
    import json
    import subprocess

    assets = ["counts-9801.csv.gz", "settings-9801.csv", "counts-9802.csv.gz"]

    def view(args, **_):
        if "sims-r98" not in args:
            return subprocess.CompletedProcess(args, 1, "", "not found")
        out = json.dumps({"assets": [{"name": a} for a in assets]})
        return subprocess.CompletedProcess(args, 0, out, "")

    monkeypatch.setattr(releases.subprocess, "run", view)
    assert releases.release_seeds("o/r", 98) == {9801}
    assert releases.release_seeds("o/r", 97) == set()


def _cfg(**over):
    return {"labels": None, "published_from": "2026-11-16T09:00:00Z"} | over


def test_configured_labels_win_over_the_listing(tmp_path):
    cfg = _cfg(labels=["x"])
    assert releases.chosen(tmp_path / "none.json", cfg) == ["x"]
    assert releases.oldest_first(tmp_path / "none.json", cfg) == ["x"]


def test_listed_checkpoints_are_chosen_less_skipped(tmp_path):
    import json

    listing = tmp_path / "releases.json"
    listing.write_text(json.dumps(LISTING))
    got = releases.chosen(listing, _cfg(skip=["slots-bbb"]))
    assert got == ["slots-ccc", "aaa"]
    assert releases.oldest_first(listing, _cfg()) == [
        "slots-ccc",
        "aaa",
        "slots-bbb",
    ]


def test_no_order_before_the_releases_are_listed(tmp_path):
    assert releases.oldest_first(tmp_path / "none.json", _cfg()) == []
