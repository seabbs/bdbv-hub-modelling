"""Scoring units' commands and the German report's arguments."""

from pathlib import Path

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
        "'randomwalknowcast=Random walk nowcast' "
        "'n100=runa n100' 'n200=runa n200' --from reference=/c/g/reference"
    )


def _scoring(
    tmp_path, labels=None, listing=None, local_extra=None, score_extra=None
):
    from types import SimpleNamespace

    from flowlib import rounds, runs, training

    cfg = runs.load_config()
    cfg["runs"] = {"x": {"prefixes": [100, 200], "outbreaks": 200}}
    cfg["train_runs"] = ["x"]
    cfg["score"] = {
        **cfg["score"],
        "labels": labels,
        "reference": ["ref"],
        **(score_extra or {}),
    }
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
        **(local_extra or {}),
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
    assert "packages/nowcastgermany/config/germany.toml" in files
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


def test_no_rule_keys_code_by_a_param_mtime_never_reads():
    from pathlib import Path

    rules = Path(__file__).parents[2] / "workflow" / "rules"
    text = "".join(p.read_text() for p in rules.glob("*.smk"))
    assert "code=lambda wc: scoring.reference_key" not in text
    assert "code=lambda wc: scoring.ft_check_key()" not in text


def test_a_key_is_a_file_made_once_per_code(tmp_path):
    # Every profile reruns on file times only (rerun-triggers: mtime), so
    # a code key reaches a rule as a file named by the key.
    folder = tmp_path / "keys"
    first = scoring.key_file(folder, "reference", "abc")
    assert first == str(folder / "reference-abc.key")
    # The first key is older than any output made before keys existed.
    assert Path(first).stat().st_mtime == 0
    assert scoring.key_file(folder, "reference", "abc") == first
    # Changed code: a new file, dated now, so the rule reruns once.
    changed = scoring.key_file(folder, "reference", "def")
    assert Path(changed).stat().st_mtime > 0
    # Checkouts with either code share the folder without rewriting it.
    assert scoring.key_file(folder, "reference", "abc") == first
    assert Path(first).stat().st_mtime == 0
    assert not list(folder.glob(".*"))


def test_the_reference_unit_reads_its_code_key_as_a_file(tmp_path):
    s = _scoring(tmp_path, labels=[])
    (ref,) = s.reference_key_input(scoring.REFERENCE)
    assert Path(ref).parent == tmp_path / "cache" / "keys"
    assert Path(ref).name.startswith(f"{scoring.REFERENCE}-")
    assert s.reference_key_input("abc123") == []


def test_evaluation_runs_right_after_the_hand_offs():
    from types import SimpleNamespace

    from flowlib import compute, scoring

    s = scoring.Scoring.__new__(scoring.Scoring)
    new, old = SimpleNamespace(n="800"), SimpleNamespace(n="400")
    assert s.evaluation_priority(new) > s.evaluation_priority(old)
    assert s.evaluation_priority(new) < compute.NEXT_BATCH_PRIORITY
    assert s.evaluation_priority(old) > compute.REFERENCE_PRIORITY + 2
    # Step 0 first, which every checkpoint's evaluation needs.
    assert s.evaluation_priority(SimpleNamespace()) > s.evaluation_priority(new)
    assert not hasattr(compute, "BOUNDARY_PRIORITY")


def test_code_keys_ignore_modules_the_jobs_never_import():
    # A change to the trainer must not rerun the German reference or the
    # BDBV check, which never import it.
    from flowlib import scoring

    from nowcastutils import codekey

    for files in (
        scoring.reference_code("reference"),
        scoring.ft_check_code(),
    ):
        assert not any(f.endswith("finetune/stream.py") for f in files)
        assert not any(f.endswith("finetune/evaluate.py") for f in files)
    assert any(
        f.endswith("finetune/evaluate.py") for f in scoring.evaluator_code()
    )
    assert codekey.digest(scoring.evaluator_code())


def test_a_german_unit_splits_its_dates_over_its_threads():
    # Each of the rule's threads is a worker process of one thread, so
    # the scheduler counts the unit's cores as it runs them.
    got = scoring.unit_shell("germany")
    assert "SCORE_WORKERS={threads} " in got
    assert "SCORE_WORKERS" not in scoring.unit_shell("release")


def test_german_unit_workers_come_from_the_config_or_machine(tmp_path):
    s = _scoring(tmp_path, labels=[])
    assert s.score_workers == s.sc["workers"]
    s = _scoring(tmp_path, labels=[], local_extra={"SCORE_WORKERS": "3"})
    assert s.score_workers == 3
    # Each worker holds what one process held, and the coordinator its
    # own on top.
    assert (
        s.unit_memory_floor() == 3 * scoring.WORKER_MB + scoring.COORDINATOR_MB
    )


def test_a_german_unit_on_a_gpu_is_one_worker_s_memory(tmp_path):
    # On CUDA the unit forecasts in its own process (one GPU, one
    # process), so its floor is one worker's.
    import dataclasses

    s = _scoring(tmp_path, labels=[], local_extra={"SCORE_WORKERS": "3"})
    s.germany_forecast = dataclasses.replace(s.germany_forecast, device="cuda")
    assert s.unit_memory_floor() == scoring.WORKER_MB + scoring.COORDINATOR_MB


def test_german_units_stay_on_the_cpu_beside_a_free_gpu(tmp_path, monkeypatch):
    # The config's `score.device: cpu`: with a GPU no run trains on, the
    # held-out evaluation takes it and the German units keep their CPU
    # workers.
    monkeypatch.setattr(scoring.Scoring, "_gpus", lambda _self: 2)
    s = _scoring(tmp_path, labels=[], local_extra={"SCORE_WORKERS": "3"})
    assert s.sc["device"] == "cpu"
    assert s.checkpoint_forecast.device == "cuda"
    assert s.germany_forecast.device == "cpu"
    assert s.germany_forecast.gpu == 0
    assert (
        s.unit_memory_floor() == 3 * scoring.WORKER_MB + scoring.COORDINATOR_MB
    )
    env = s.germany_env(None, 3, None)
    assert "DEVICE=cpu" in env


def test_german_units_follow_the_scoring_device_when_unset(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(scoring.Scoring, "_gpus", lambda _self: 2)
    s = _scoring(
        tmp_path,
        labels=[],
        local_extra={"SCORE_WORKERS": "3"},
        score_extra={"device": None},
    )
    assert s.germany_forecast == s.checkpoint_forecast
    assert s.germany_forecast.device == "cuda"


def test_the_evaluation_leaves_a_german_unit_its_workers(tmp_path):
    from flowlib import machine

    s = _scoring(tmp_path, labels=[], local_extra={"SCORE_WORKERS": "3"})
    assert s.evaluate_threads == machine.scoring_threads(s.local, 3)
