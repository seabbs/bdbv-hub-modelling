from pathlib import Path
from types import SimpleNamespace

import pytest
from flowlib import releases, rounds, runs, training


def _training(
    tmp_path: Path, sims_from: str = "local", **run
) -> training.Training:
    cfg = runs.load_config()
    cfg["runs"] = {"x": {"prefixes": [100, 200], "outbreaks": 200, **run}}
    cfg["train_runs"] = ["x"]
    local = {
        "ROOT": str(tmp_path),
        "HOST_NAME": "box",
        "REMOTE": "",
        "REMOTE_ROOT": "",
        "SIM_SEEDS": "",
        "SIMS_FROM": sims_from,
        "PUBLISH": "",
    }
    return training.Training(cfg, local, rounds.Rounds(cfg, local, None))


def _wc(**kw: str) -> SimpleNamespace:
    return SimpleNamespace(run="x", **kw)


def test_a_run_trains_in_its_own_folder(tmp_path: Path) -> None:
    t = _training(tmp_path)
    assert t.dir == f"{tmp_path}/runs"
    assert t.train_dir("x") == f"{tmp_path}/runs/x/train"
    assert t.trained_runs() == ["x"]
    assert t.ft("prefixes")(_wc()) == [100, 200]
    assert t.first_prefix("x") == 100


def test_a_machine_setting_wins_over_the_run_s(tmp_path: Path) -> None:
    t = _training(tmp_path)
    t.local = {**t.local, "CHRONOS_BATCH": "64"}
    assert t.local_or_ft("CHRONOS_BATCH", "outbreaks")(_wc()) == "64"
    assert t.local_or_ft("UNSET", "outbreaks")(_wc()) == 200


def test_a_segment_resumes_from_the_checkpoint_before(tmp_path: Path) -> None:
    t = _training(tmp_path)
    train = t.train_dir("x")
    assert t.evaluate_prev(_wc(n="100")) == []
    assert t.evaluate_prev(_wc(n="200")) == f"{train}/n100/settings.json"
    assert t.earlier_settings(_wc(n="200")) == [f"{train}/n100/settings.json"]
    assert t.evaluate_prev_dir(SimpleNamespace(prev="a/b/settings.json")) == (
        Path("a/b")
    )
    assert t.evaluate_prev_dir(SimpleNamespace(prev=[])) is None


def test_held_out_losses_are_asked_for_only_when_offloaded(
    tmp_path: Path,
) -> None:
    off = _training(tmp_path, eval="offload")
    train = off.train_dir("x")
    assert off.offloaded("x", 100)[1] == f"{train}/evals/n100.done"
    assert len(off.heldout_curve_inputs(_wc())) == 6
    local = _training(tmp_path, eval="local")
    assert local.offloaded("x", 100) == []
    assert len(local.heldout_curve_inputs(_wc())) == 2


def test_a_worker_copies_back_what_publishing_reads(tmp_path: Path) -> None:
    t = _training(tmp_path)
    back = t.segment_copy_back(_wc(n="100"))
    assert back[0] == f"{t.train_dir('x')}/n100/outbreaks.txt"
    assert len(back) == len(training.SEGMENT_COPY_BACK)


def test_batches_come_from_local_rounds_only_when_the_run_reads_them(
    tmp_path: Path,
) -> None:
    t = _training(tmp_path, sims_from="local")
    seed = t.held_seeds("x")[0]
    got = t.fetch_source(_wc(seed=str(seed)))
    assert got["counts"].endswith(f"/sims/counts-{seed}.csv.gz")
    assert got["counts"].startswith(f"{tmp_path}/rounds/r")
    assert (
        _training(tmp_path, sims_from="release").fetch_source(
            _wc(seed=str(seed))
        )
        == []
    )


def test_the_held_out_seeds_are_one_range(tmp_path: Path) -> None:
    t = _training(tmp_path)
    seeds = t.held_seeds("x")
    assert t.held_range("x") == f"{seeds[0]} {seeds[-1]}"


def test_a_saved_checkpoint_is_final(tmp_path: Path) -> None:
    t = _training(tmp_path)
    done = Path(t.train_dir("x")) / "n100" / "settings.json"
    done.parent.mkdir(parents=True)
    done.write_text("{}")
    assert t.saved("x", 100)
    assert not t.saved("x", 200)
    assert t.segment_inputs(_wc(n="100")) == {}
    assert t.published_checkpoints(None) == [
        f"{t.train_dir('x')}/n100/published.json"
    ]


def test_released_batches_are_listed_once_per_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    def seeds(_repo: str, round_: int) -> set[int]:
        calls.append(round_)
        return {round_}

    monkeypatch.setattr(releases, "release_seeds", seeds)
    t = _training(tmp_path, sims_from="release")
    first = t.listed_batches("x")
    asked = len(calls)
    assert t.listed_batches("x") == first
    assert len(calls) == asked
    record = Path(t.runs["x"].dir) / "available-batches.txt"
    assert record.read_text().split() == [str(s) for s in sorted(first)]
