"""Scoring units' commands and the German report's arguments."""

from flowlib import scoring


def test_a_unit_scores_and_stamps():
    got = scoring.unit_shell("germany", "{params.segment} ")
    assert got.startswith("mkdir -p {params.copy_back} && env {params.env} ")
    assert (
        f"{scoring.EVALUATE} germany {{params.cache}} {{params.label}} " in got
    )
    assert got.endswith("{params.segment} > {log} 2>&1 && touch {output}")


def test_the_report_names_the_references_and_saved_checkpoints():
    got = scoring.report_args("runa", [100, 200], ["reference"], "/c/g")
    assert got == (
        "reference=Zero-shot "
        "'randomwalknowcast_daily=Random walk nowcast (daily)' "
        "'n100=runa n100' 'n200=runa n200' --from reference=/c/g/reference"
    )


def _scoring(tmp_path, labels=None, listing=None):
    from types import SimpleNamespace

    from flowlib import rounds, runs, training

    cfg = runs.load_config()
    cfg["runs"] = {"x": {"prefixes": [100, 200], "outbreaks": 200}}
    cfg["train_runs"] = ["x"]
    cfg["score"] = {**cfg["score"], "labels": labels, "reference": ["ref"]}
    local = {
        "ROOT": str(tmp_path),
        "HOST_NAME": "box",
        "REMOTE": "",
        "REMOTE_ROOT": "",
        "SIM_SEEDS": "",
        "SIMS_FROM": "local",
        "PUBLISH": "",
        "SCORE_CACHE": str(tmp_path / "cache"),
        "MIN_FREE_GB": "1",
        "MIN_DISK_GB": "2",
        "FORECAST_THREADS": "",
    }
    trained = training.Training(cfg, local, rounds.Rounds(cfg, local, None))
    listed = SimpleNamespace(output=[listing])
    checkpoints = SimpleNamespace(
        release_list=SimpleNamespace(get=lambda: listed)
    )
    return scoring.Scoring(cfg, local, trained, checkpoints)


def test_the_configured_labels_are_scored_with_the_references(tmp_path):
    s = _scoring(tmp_path, labels=["abc123"])
    assert s.checkpoint_labels() == ["abc123"]
    cache = tmp_path / "cache"
    assert s.unit_stamps() == [
        f"{cache}/{scoring.GERMANY_UNIT}/ref.done",
        f"{cache}/{scoring.GERMANY_UNIT}/abc123.done",
        f"{cache}/heldout/abc123.done",
    ]
    assert s.references() == [f"{cache}/{scoring.GERMANY_UNIT}/ref.done"]


def test_unlisted_labels_come_from_the_release_listing(tmp_path, monkeypatch):
    from flowlib import releases

    seen = []
    monkeypatch.setattr(
        releases, "chosen", lambda listing, _sc: seen.append(listing) or ["b"]
    )
    s = _scoring(tmp_path, listing="releases.json")
    assert s.checkpoint_labels() == ["b"]
    assert seen == ["releases.json"]


def test_a_unit_runs_its_evaluation_with_no_wait_or_prefix(tmp_path):
    s = _scoring(tmp_path, labels=[])
    assert s.unit_threads == 2
    shell = s.unit_shell("germany")
    assert shell.startswith("mkdir -p {params.copy_back} && env {params.env} ")
    for gone in ("wait_for_room", "nice", "{params.prefix}"):
        assert gone not in shell, gone


def test_a_run_is_scored_on_its_saved_checkpoints(tmp_path):
    s = _scoring(tmp_path, labels=[])
    done = tmp_path / "runs" / "x" / "train" / "n100" / "settings.json"
    done.parent.mkdir(parents=True)
    done.write_text("{}")
    unit = f"{tmp_path}/runs/x/scores/{scoring.GERMANY_UNIT}"
    assert s.run_units("x") == [f"{unit}/n100.done"]
    assert s.run_scores() == [f"{tmp_path}/runs/x/reports/german/index.html"]
    assert "'n100=x n100'" in s.report_names(type("W", (), {"run": "x"}))


def test_the_reference_unit_reruns_when_its_models_code_changes():
    # The reference's forecasts are shared by every run and depend only on
    # its models' code and the German config, so those are its inputs.
    files = scoring.reference_code("reference")
    assert "config/germany.toml" in files
    assert any(f.endswith("nowcastbaselines/randomwalk.py") for f in files)
    assert any(f.endswith("nowcastfoundation/forecaster.py") for f in files)
    assert scoring.reference_code("abc123def456") == []
    # A digest of their contents, which the rule reruns on.
    key = scoring.reference_key("reference")
    assert len(key) == 12
    assert scoring.reference_key("abc123def456") == ""


def test_the_german_unit_is_given_its_fetched_triangle():
    got = scoring.unit_shell("germany")
    assert "GERMANY_TRIANGLE={input.germany} " in got
    assert "GERMANY_TRIANGLE" not in scoring.unit_shell("release")


def test_a_key_file_changes_only_when_its_key_does(tmp_path):
    # Every profile reruns on file times only (rerun-triggers: mtime), so
    # a code key reaches a rule as a file it reads, rewritten only when
    # the key moves: a checkout reruns nothing, a code change reruns it.
    import os

    path = tmp_path / "keys" / "reference.key"
    assert scoring.key_file(path, "abc") == str(path)
    assert path.read_text() == "abc\n"
    # A first key is older than any output made before it existed.
    assert path.stat().st_mtime == 0
    os.utime(path, (1, 1))
    scoring.key_file(path, "abc")
    assert path.stat().st_mtime == 1
    scoring.key_file(path, "def")
    assert path.read_text() == "def\n"
    assert path.stat().st_mtime > 1


def test_the_reference_unit_reads_its_code_key_as_a_file(tmp_path):
    s = _scoring(tmp_path, labels=[])
    ref = s.reference_key_input(scoring.REFERENCE)
    assert ref == [f"{tmp_path / 'cache'}/keys/{scoring.REFERENCE}.key"]
    assert s.reference_key_input("abc123") == []


def test_no_rule_keys_code_by_a_param_mtime_never_reads():
    from pathlib import Path

    rules = Path(__file__).parents[2] / "workflow" / "rules"
    text = "".join(p.read_text() for p in rules.glob("*.smk"))
    assert "code=lambda wc: scoring.reference_key" not in text
    assert "code=lambda wc: scoring.ft_check_key()" not in text
