"""Places of the fine-tune's training blocks."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from sim_rounds import sim_dir

from nowcastfoundation.finetune import chronos as ft
from nowcastfoundation.finetune.chronos import (
    choose_places,
    plan_batches,
    size_class,
    target_levels,
)

STRATA = [f"P{p}-A{a}" for p in range(1, 5) for a in range(1, 4)]


def test_univariate_blocks_are_about_their_share():
    from nowcastfoundation.finetune import chronos as finetune_chronos

    rng = np.random.default_rng(0)
    mixes = []
    for _ in range(3000):
        places, mix = choose_places(STRATA, int(rng.integers(4, 20)), rng)
        levels = set(target_levels(places))
        expected = {
            "all_levels": {"national", "intermediate", "leaf"},
            "leaves_national": {"national", "leaf"},
            "leaves": {"leaf"},
            "national": {"national"},
            "single": {"leaf"},
        }[mix]
        assert levels == expected
        mixes.append(mix)
    univariate = sum(m in ("national", "single") for m in mixes) / len(mixes)
    assert finetune_chronos.UNIVARIATE_SHARE == 0.2
    assert abs(univariate - 0.2) < 0.03
    # The multivariate mixes share the rest about evenly.
    for m in ("all_levels", "leaves_national", "leaves"):
        assert abs(mixes.count(m) / len(mixes) - 0.8 / 3) < 0.04


def test_flat_strata_have_no_intermediate_level():
    rng = np.random.default_rng(1)
    flat = [f"P{i:02d}" for i in range(1, 9)]
    mixes = {choose_places(flat, 10, rng)[1] for _ in range(400)}
    assert mixes == {"leaves_national", "leaves", "national", "single"}


def test_places_stay_within_the_budget():
    rng = np.random.default_rng(1)
    for budget in range(1, 20):
        places, _ = choose_places(STRATA, budget, rng)
        assert len(places) <= budget + 1


def test_batches_use_every_block_once_and_pack_by_size():
    rng = np.random.default_rng(2)
    sizes = np.exp(rng.uniform(0, np.log(2000), 500)).astype(int) + 1
    plan = plan_batches(sizes, 256, rng)
    used = sorted(i for batch in plan for i in batch)
    assert used == list(range(len(sizes)))
    for batch in plan:
        # One size class per batch, and only a block of 256 series or more
        # makes a batch past twice the batch size.
        assert len({size_class(int(sizes[i])) for i in batch}) == 1
        total = sizes[batch].sum()
        assert total < 2 * 256 or len(batch) == 1


def test_a_mixed_batch_is_moved_aside_to_run_again(tmp_path):
    import pandas as pd

    from nowcastfoundation.finetune.stream import seed_done

    pd.DataFrame({"outbreak": ["5001-0001", "5001-0002"]}).to_csv(
        tmp_path / "settings-5001.csv", index=False
    )
    pd.DataFrame({"outbreak": ["5001-0001-1", "5001-0002-3"]}).to_csv(
        tmp_path / "counts-5001.csv.gz", index=False
    )
    # Draws of the same simulations: whole.
    assert seed_done(tmp_path, 5001)
    pd.DataFrame({"outbreak": ["5001-0001-1", "5001-0009-1"]}).to_csv(
        tmp_path / "counts-5001.csv.gz", index=False
    )
    # Another run's counts: moved aside, not done.
    assert not seed_done(tmp_path, 5001)
    assert not (tmp_path / "settings-5001.csv").exists()
    assert list((tmp_path / "bad").iterdir())


def test_a_non_finite_loss_or_gradient_leaves_the_weights():
    import pytest

    torch = pytest.importorskip("torch")
    from nowcastfoundation.finetune.stream import finite_step

    layer = torch.nn.Linear(2, 1)
    params = list(layer.parameters())
    optimiser = torch.optim.AdamW(params, lr=0.1)
    before = [p.detach().clone() for p in params]
    layer(torch.ones(1, 2)).sum().backward()
    for p in params:
        p.grad[...] = float("nan")
    assert not finite_step(params, optimiser, 1.0)
    # A skipped block leaves no gradients and a NaN loss.
    assert not finite_step(params, optimiser, float("nan"))
    assert all(torch.equal(a, b) for a, b in zip(before, params, strict=True))
    layer(torch.ones(1, 2)).sum().backward()
    assert finite_step(params, optimiser, 0.5)
    assert not torch.equal(before[0], params[0])


def test_the_trainer_has_no_way_to_skip_a_block():
    from nowcastfoundation.finetune import stream as finetune_stream

    # Skipping blocks biases training: none is ever left out.
    assert not hasattr(finetune_stream, "skipped_lines")


def test_no_batch_holds_more_series_than_the_budget_allows():
    rng = np.random.default_rng(3)
    sizes = rng.integers(100, 200, 50)
    for batch in plan_batches(sizes, 256, rng, most=300):
        assert sizes[batch].sum() <= 300 or len(batch) == 1


def test_flat_places_can_nest_in_regions_with_levels_recorded():
    import pandas as pd

    from nowcastfoundation.finetune.chronos import regions, target_levels

    rng = np.random.default_rng(4)
    places = [f"P{i:02d}" for i in range(1, 9)]
    mapping = regions(places, rng)
    assert set(mapping) == set(places)
    parents = {m.split("-")[0] for m in mapping.values()}
    assert 2 <= len(parents) <= 6
    assert target_levels(("CD", "R1", "R1-P01", "A1")) == (
        "national",
        "intermediate",
        "leaf",
        "intermediate",
    )
    assert target_levels(("CD", "P01", "P02")) == ("national", "leaf", "leaf")
    del pd


def test_ages_are_not_grouped_into_regions():
    import pandas as pd

    from nowcastfoundation.finetune.chronos import outbreak_blocks

    days = pd.date_range("2026-01-01", periods=140)
    rows = [
        (f"A{a}", str(d.date()), str((d + pd.Timedelta(days=2)).date()), 5)
        for a in range(1, 6)
        for d in days
    ]
    counts = pd.DataFrame(
        rows, columns=["location", "reference_date", "report_date", "count"]
    )
    blocks = outbreak_blocks(
        counts, [f"A{a}" for a in range(1, 6)], np.random.default_rng(0), 1, 4
    )
    assert blocks
    assert all("intermediate" not in b["levels"] for b in blocks)


def test_an_empty_batch_is_done_with_no_simulations(tmp_path):
    import pandas as pd

    from nowcastfoundation.finetune.stream import ready_outbreaks

    (tmp_path / "settings-5001.csv").write_text("")
    (tmp_path / "counts-5001.csv.gz").write_text("")
    pd_rows = "outbreak\n5002-0001\n"
    (tmp_path / "settings-5002.csv").write_text(pd_rows)
    pd.DataFrame({"outbreak": ["5002-0001-1"]}).to_csv(
        tmp_path / "counts-5002.csv.gz", index=False
    )
    assert ready_outbreaks(tmp_path, range(5001, 5003)) == ["5002-0001"]


def test_stop_rule_needs_two_checkpoints():
    from nowcastfoundation.finetune.stream import should_stop

    assert not should_stop([0.2], 8000, 8000, 0.01)
    assert should_stop([0.2, 0.199], 16000, 8000, 0.01)
    assert not should_stop([0.2, 0.15], 16000, 8000, 0.01)
    assert not should_stop([0.2, 0.199], 4000, 8000, 0.01)


def test_a_split_batch_starts_from_clear_gradients():
    from nowcastfoundation.finetune.stream import accumulate

    grads: list[int] = []

    def backward(part, _share):
        if len(part) > 2:
            grads.append(-1)  # a partial backward before running out
            raise RuntimeError("MPS backend out of memory")
        grads.extend(part)
        return 0.5

    def zero():
        grads.clear()

    value = accumulate([1, 2, 3, 4, 9], backward, zero)
    # The partial backward is cleared; every block is used once.
    assert sorted(grads) == [1, 2, 3, 4, 9]
    assert value == 0.5


def test_a_block_too_large_for_memory_stops_the_run():
    from nowcastfoundation.finetune.stream import OOM_WAITS, accumulate

    def backward(part, _share):
        if 9 in part:
            raise RuntimeError("CUDA out of memory")
        return 0.5

    waited: list[int] = []
    with pytest.raises(RuntimeError, match="too large for memory"):
        accumulate([1, 9], backward, lambda: None, wait=waited.append)
    # It waits for memory a set number of times before stopping.
    assert waited == list(range(1, OOM_WAITS + 1))


def test_a_block_waits_while_another_process_holds_the_memory():
    # Overseer 18:00: Run A died when a scoring job held the GPU. A block
    # that keeps running out waits and is run again; the run goes on.
    from nowcastfoundation.finetune.stream import accumulate

    calls: list[list[int]] = []
    waited: list[int] = []

    def backward(part, _share):
        calls.append(list(part))
        if len(calls) <= 4:
            raise RuntimeError("CUDA out of memory")
        return 0.5

    assert accumulate([7], backward, lambda: None, wait=waited.append) == 0.5
    assert calls == [[7]] * 5
    # The first retry is at once, then it waits before each run.
    assert waited == [1, 2, 3]


def test_a_block_that_runs_out_once_is_retried():
    from nowcastfoundation.finetune.stream import accumulate

    calls: list[list[int]] = []
    zeroed: list[int] = []

    def backward(part, _share):
        calls.append(list(part))
        # Out of memory the first time only, as when another process
        # held the memory for a moment.
        if len(calls) == 1:
            raise RuntimeError("CUDA out of memory")
        return 0.5

    value = accumulate([7], backward, lambda: zeroed.append(1))
    assert value == 0.5
    # Memory is freed (`zero`) before the retry.
    assert calls == [[7], [7]]
    assert zeroed == [1]


def _checkpoint(tmp_path):
    ckpt = tmp_path / "finetuned-ckpt"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text("{}")
    (ckpt / "adapter_model.safetensors").write_text("w")
    (tmp_path / "settings.json").write_text('{"n_outbreaks": 1000}')
    (tmp_path / "outbreaks.txt").write_text("1\n")
    return tmp_path


class _Gh:
    """Fake gh: `assets` maps the release's asset names to their content."""

    def __init__(self, assets=None):
        self.assets = assets
        self.calls = []

    def __call__(self, command, **_):
        import subprocess

        self.calls.append(command[:3])
        action = command[2]
        if action == "view":
            if self.assets is None:
                return subprocess.CompletedProcess(command, 1, "", "not found")
            names = [{"name": n} for n in self.assets]
            out = __import__("json").dumps({"assets": names})
            return subprocess.CompletedProcess(command, 0, out, "")
        if action == "download":
            sums = self.assets.get("sha256sums.txt", "")
            return subprocess.CompletedProcess(command, 0, sums, "")
        if action in ("create", "upload"):
            self.assets = self.assets or {}
            for f in command:
                if Path(f).is_file():
                    self.assets[Path(f).name] = Path(f).read_text()
            return subprocess.CompletedProcess(command, 0, "ok", "")
        raise AssertionError(command)


def test_publish_is_done_when_the_release_matches(tmp_path):
    from nowcastfoundation.finetune.stream import publish

    where = _checkpoint(tmp_path)
    gh = _Gh()
    publish(where, "abc", run=gh)
    assert [c[2] for c in gh.calls] == ["view", "create"]
    gh.calls.clear()
    publish(where, "abc", run=gh)
    assert "create" not in [c[2] for c in gh.calls]
    assert "upload" not in [c[2] for c in gh.calls]


def test_publish_releases_slots_checkpoints_apart(tmp_path):
    from nowcastfoundation.finetune.stream import publish

    where = _checkpoint(tmp_path)
    commands = []
    gh = _Gh()

    def run(command, **kwargs):
        commands.append(command)
        return gh(command, **kwargs)

    publish(where, "slots-abc", run=run)
    create = next(c for c in commands if c[2] == "create")
    assert create[3] == "chronos-slots-abc"


def test_publish_uploads_only_missing_assets(tmp_path):
    from nowcastfoundation.finetune.stream import publish

    where = _checkpoint(tmp_path)
    gh = _Gh()
    publish(where, "abc", run=gh)
    del gh.assets["outbreaks.txt"]
    gh.calls.clear()
    publish(where, "abc", run=gh)
    assert [c[2] for c in gh.calls if c[2] != "download"] == ["view", "upload"]
    assert "outbreaks.txt" in gh.assets


def test_publish_refuses_a_release_with_other_sums(tmp_path):
    import pytest

    from nowcastfoundation.finetune.stream import publish

    where = _checkpoint(tmp_path)
    gh = _Gh({"sha256sums.txt": "0  adapter_model.safetensors\n"})
    with pytest.raises(RuntimeError, match="differ"):
        publish(where, "abc", run=gh)
    assert "upload" not in [c[2] for c in gh.calls]


def test_a_finished_batch_is_read_once(tmp_path, monkeypatch):
    import pandas as pd

    from nowcastfoundation.finetune import stream as finetune_stream

    (tmp_path / "settings-5003.csv").write_text("outbreak\n5003-0001\n")
    pd.DataFrame({"outbreak": ["5003-0001-1"]}).to_csv(
        tmp_path / "counts-5003.csv.gz", index=False
    )
    first = finetune_stream.ready_outbreaks(tmp_path, range(5003, 5004))
    reads = []
    monkeypatch.setattr(
        finetune_stream, "read_ids", lambda p: reads.append(p) or pd.Series()
    )
    again = finetune_stream.ready_outbreaks(tmp_path, range(5003, 5004))
    assert again == first == ["5003-0001"]
    assert reads == []


@pytest.mark.slow
def test_strata_with_no_cases_give_no_empty_blocks():
    import pandas as pd

    from nowcastfoundation.finetune.chronos import outbreak_blocks

    rows = [
        ("P01", onset, onset + pd.Timedelta(days=d), 5)
        for onset in pd.date_range("2026-01-03", periods=140)
        for d in (0, 2, 6)
    ]
    counts = pd.DataFrame(
        rows, columns=["location", "reference_date", "report_date", "count"]
    )
    counts["reference_date"] = counts["reference_date"].dt.date.astype(str)
    counts["report_date"] = counts["report_date"].dt.date.astype(str)
    # P02 and P03 had no cases, so their leaves are never released.
    for seed in range(20):
        rng = np.random.default_rng(seed)
        for block in outbreak_blocks(counts, ["P01", "P02", "P03"], rng, 3):
            assert block["context"].shape[0] > 0


def test_blocks_cut_in_workers_match_blocks_cut_here(tmp_path):
    from nowcastfoundation.finetune.stream import cut, cut_in_workers

    ids = sim_dir(tmp_path)
    here = cut(tmp_path, ids, 1, 6)
    pairs, empty = cut_in_workers(None, tmp_path, ids, 1, 6, parts=3)
    assert [o for o, _ in pairs] == [o for o, _ in here]
    for (_, a), (_, b) in zip(pairs, here, strict=True):
        assert np.array_equal(
            np.asarray(a["context"]), np.asarray(b["context"]), equal_nan=True
        )
    assert empty == []


def test_blocks_cut_by_a_process_pool_match(tmp_path):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    from nowcastfoundation.finetune import chronos as finetune_chronos
    from nowcastfoundation.finetune.stream import (
        cut,
        cut_in_workers,
        init_worker,
    )

    ids = sim_dir(tmp_path)
    here = cut(tmp_path, ids, 1, 6)
    with ProcessPoolExecutor(
        2,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=init_worker,
        initargs=(
            finetune_chronos.MAX_SERIES,
            finetune_chronos.SOURCES,
        ),
    ) as pool:
        pairs, _ = cut_in_workers(pool, tmp_path, ids, 1, 6, parts=2)
    assert [o for o, _ in pairs] == [o for o, _ in here]


def test_a_chunk_waits_for_a_missing_batch_in_seed_order(tmp_path):
    from nowcastfoundation.finetune.stream import next_chunk

    ids = sim_dir(tmp_path)  # seeds 6001 and 6002, two simulations each
    # Seed 6002 finished; 6001 has not.
    for name in ("counts-6001.csv.gz", "settings-6001.csv"):
        (tmp_path / name).rename(tmp_path / f"late-{name}")
    seeds = range(6001, 6003)
    assert next_chunk(tmp_path, seeds, [], 2) is None
    # Once it finishes, the chunk is its outbreaks, then 6002's.
    for name in ("counts-6001.csv.gz", "settings-6001.csv"):
        (tmp_path / f"late-{name}").rename(tmp_path / name)
    assert next_chunk(tmp_path, seeds, [], 2) == ids[:2]
    assert next_chunk(tmp_path, seeds, ids[:2], 2) == ids[2:]


def test_blocks_are_cut_without_torch(tmp_path):
    import subprocess

    ids = sim_dir(tmp_path)
    # CI's test environment has no torch: cutting blocks must not need it.
    code = (
        "import builtins, sys\n"
        "real = builtins.__import__\n"
        "def no_torch(name, *args, **kwargs):\n"
        "    if name.split('.')[0] == 'torch':\n"
        "        raise ModuleNotFoundError(name)\n"
        "    return real(name, *args, **kwargs)\n"
        "builtins.__import__ = no_torch\n"
        "from nowcastfoundation.finetune import stream as fs\n"
        f"pairs = fs.cut(__import__('pathlib').Path({str(tmp_path)!r}), "
        f"{ids!r}, 1, 6)\nprint(len(pairs))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).parents[3],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr
    assert int(out.stdout.split()[-1]) > 0


def test_learning_rate_decays_linearly_to_zero_over_the_simulations():
    from nowcastfoundation.finetune.stream import learning_rate

    # Linear decay to zero over the run's simulations.
    assert learning_rate(0, 1e-5, 1000) == 1e-5
    assert abs(learning_rate(500, 1e-5, 1000) - 5e-6) < 1e-15
    assert learning_rate(1000, 1e-5, 1000) == 0.0
    assert learning_rate(1500, 1e-5, 1000) == 0.0


def _d(block) -> int:
    return block["n_covariates"] // block["n_targets"]


def test_one_source_cuts_only_that_source(tmp_path, monkeypatch):
    from nowcastfoundation.finetune import chronos as finetune_chronos
    from nowcastfoundation.finetune.stream import cut

    ids = sim_dir(tmp_path)
    monkeypatch.setattr(finetune_chronos, "SOURCES", ("weekly",))
    blocks = [b for _, b in cut(tmp_path, ids, 1, 32)]
    assert blocks
    assert all(b["source"] == "weekly" and _d(b) >= 1 for b in blocks)


def test_workers_take_the_trainer_block_settings():
    from nowcastfoundation.finetune import chronos as finetune_chronos
    from nowcastfoundation.finetune.stream import init_worker

    saved = (finetune_chronos.MAX_SERIES, finetune_chronos.SOURCES)
    try:
        init_worker(100, ("daily",), loss_days=7)
        assert finetune_chronos.PARENTS is not None
        assert finetune_chronos.LOSS_HORIZON_DAYS == 7
        assert finetune_chronos.MAX_SERIES == 100
        assert finetune_chronos.SOURCES == ("daily",)
    finally:
        finetune_chronos.MAX_SERIES, finetune_chronos.SOURCES = saved
        finetune_chronos.PARENTS = None
        finetune_chronos.LOSS_HORIZON_DAYS = None


@pytest.mark.slow
def test_blocks_reach_deployed_sizes_on_many_places():
    import pandas as pd

    from nowcastfoundation.finetune.chronos import outbreak_blocks

    places = [f"P{i:02d}" for i in range(1, 27)]
    rows = [
        (p, onset, onset + pd.Timedelta(days=d), 3)
        for p in places
        for onset in pd.date_range("2026-01-03", periods=280)
        for d in (0, 3, 9)
    ]
    counts = pd.DataFrame(
        rows, columns=["location", "reference_date", "report_date", "count"]
    )
    for col in ("reference_date", "report_date"):
        counts[col] = counts[col].dt.date.astype(str)
    # BDBV's daily block: 27 targets and 783 series. The search stops at
    # the first block that size (seed 0 gives one), not after every seed.
    assert any(
        b["n_targets"] >= 27 and b["context"].shape[0] >= 700
        for seed in range(30)
        for b in outbreak_blocks(counts, places, np.random.default_rng(seed), 3)
    )


def test_three_day_source_cuts_three_day_blocks(tmp_path, monkeypatch):
    from nowcastfoundation.finetune import chronos as finetune_chronos
    from nowcastfoundation.finetune.stream import cut

    ids = sim_dir(tmp_path)
    monkeypatch.setattr(finetune_chronos, "SOURCES", ("3day",))
    blocks = [b for _, b in cut(tmp_path, ids, 1, 32)]
    assert blocks
    for b in blocks:
        assert b["source"] == "3day"
        # D follows the lag rule in 3-day steps, at most the history.
        assert _d(b) >= 1
    assert max(_d(b) for b in blocks) >= 5


def test_blocks_predict_their_own_horizon_in_whole_patches(tmp_path):
    from nowcastfoundation.finetune.chronos import CONTEXT, OUTPUT_PATCH
    from nowcastfoundation.finetune.stream import cut

    ids = sim_dir(tmp_path)
    for _, b in cut(tmp_path, ids, 1, 32):
        truth = b["context"][0, CONTEXT:]
        horizon = int((~np.isnan(truth)).sum())
        length = b["prediction_length"]
        assert length % OUTPUT_PATCH == 0
        assert length - OUTPUT_PATCH < horizon <= length
        assert b["context"].shape[1] == CONTEXT + length


@pytest.mark.slow
def test_a_loss_horizon_stops_blocks_days_after_the_cut_off(
    tmp_path, monkeypatch
):
    from nowcastfoundation.finetune import chronos as finetune_chronos
    from nowcastfoundation.finetune.chronos import (
        CONTEXT,
        DAYS_PER_STEP,
        OUTPUT_PATCH,
    )
    from nowcastfoundation.finetune.stream import cut

    ids = sim_dir(tmp_path)
    full = [b for _, b in cut(tmp_path, ids, 1, 32)]
    monkeypatch.setattr(finetune_chronos, "LOSS_HORIZON_DAYS", 7)
    short = [b for _, b in cut(tmp_path, ids, 1, 32)]
    assert len(short) == len(full)
    shorter = 0
    for a, b in zip(full, short, strict=True):
        assert (a["source"], a["places"]) == (b["source"], b["places"])
        # The nowcast steps, up to the cut-off's, then a week of steps.
        week = -(-7 // DAYS_PER_STEP[b["source"]])
        h_full = int((~np.isnan(a["context"][0, CONTEXT:])).sum())
        h = int((~np.isnan(b["context"][0, CONTEXT:])).sum())
        assert h == min(h_full, b["nowcast"] + week)
        assert b["prediction_length"] == -(-h // OUTPUT_PATCH) * OUTPUT_PATCH
        # The context and the steps kept are as without the horizon.
        np.testing.assert_array_equal(
            b["context"][:, : CONTEXT + h], a["context"][:, : CONTEXT + h]
        )
        shorter += b["prediction_length"] < a["prediction_length"]
    assert shorter


def _raw(target_past, target_after, lags, places=("P01",)):
    """A raw block: start 3, nowcast 2, two lag series a place.

    `target_past` and `target_after` are a row a place (or one row for a
    single place); `lags` are the lag series, place by place, lag 0
    first, each a step's count as reported at that lag.
    """
    past = np.atleast_2d(np.array(target_past, dtype=float))
    after = np.atleast_2d(np.array(target_after, dtype=float))
    return {
        "context": np.concatenate(
            [np.concatenate([past, after], axis=1), np.array(lags, float)]
        ),
        "n_targets": len(places),
        "n_covariates": 2 * len(places),
        "places": tuple(places),
        "start": past.shape[1],
        "nowcast": 2,
    }


def test_reported_cases_count_what_is_known_at_the_cut_off():
    from nowcastfoundation.finetune.chronos import reported_cases

    nan = np.nan
    # Daily steps; the cut-off is step 4 (start 3 + nowcast 2 - 1). The
    # latest report of step 4 is its lag 0 value, of step 3 its lag 1.
    lags = [[0, 0, 0, 2, 3, nan, nan], [0, 0, 0, 4, nan, nan, nan]]
    # Each step once, at its latest report: 4 + 3, not 2 + 3 + 4. Cases
    # after the cut-off, in the target, do not count.
    assert reported_cases(_raw([0, 0, 0], [0, 0, 9, 9], lags), 1, 28) == 7
    # The targets' context counts within the window.
    assert reported_cases(_raw([5, 0, 1], [0, 0, 0, 0], lags), 1, 28) == 13
    # But not before it: 3 daily steps reach back to step 2.
    assert reported_cases(_raw([5, 0, 1], [0, 0, 0, 0], lags), 1, 3) == 8
    # The window is in days: 21 days of 7-day steps reach back to step 2,
    # 35 days to step 0.
    assert reported_cases(_raw([5, 0, 1], [0, 0, 0, 0], lags), 7, 21) == 8
    assert reported_cases(_raw([5, 0, 1], [0, 0, 0, 0], lags), 7, 35) == 13


def test_reported_cases_count_each_case_once_at_the_busiest_level():
    from nowcastfoundation.finetune.chronos import NATIONAL, reported_cases

    zeros = [[0] * 7] * 6
    # The country and its two places: the places' cases, not the country's
    # on top.
    block = _raw(
        [[0, 0, 10], [0, 0, 4], [0, 0, 6]],
        [[0] * 4] * 3,
        zeros,
        places=(NATIONAL, "P01", "P02"),
    )
    assert reported_cases(block, 1, 28) == 10
    # A block of some of the places is not their sum: the country's cases
    # count though the place in the block has none.
    some = _raw(
        [[0, 0, 12], [0, 0, 0]],
        [[0] * 4] * 2,
        zeros[:4],
        places=(NATIONAL, "P02"),
    )
    assert reported_cases(some, 1, 28) == 12
    # The country alone counts itself.
    alone = _raw([0, 0, 10], [0, 0, 0, 0], zeros[:2], places=(NATIONAL,))
    assert reported_cases(alone, 1, 28) == 10


def test_reported_cases_refuse_a_block_whose_lags_do_not_fit():
    from nowcastfoundation.finetune.chronos import NATIONAL, reported_cases

    block = _raw([0, 0, 0], [0, 0, 0, 0], [[0] * 7] * 2)
    # Lag series that do not divide over the places would be read across
    # places.
    two = _raw(
        [[0] * 3] * 2, [[0] * 4] * 2, [[0] * 7] * 3, places=(NATIONAL, "P01")
    )
    with pytest.raises(ValueError, match="lag series"):
        reported_cases({**two, "n_covariates": 3}, 1, 28)
    # A nowcast longer than the lags reads past a place's rows.
    with pytest.raises(ValueError, match="nowcast"):
        reported_cases({**block, "nowcast": 3}, 1, 28)


def test_a_block_is_quiet_below_the_threshold_whatever_follows():
    from nowcastfoundation.finetune.chronos import quiet

    nan = np.nan
    lags = [[0, 0, 0, 2, 3, nan, nan], [0, 0, 0, 4, nan, nan, nan]]
    block = _raw([0, 0, 0], [0, 0, 99, 99], lags)
    assert quiet(block, 1, 8)
    assert not quiet(block, 1, 7)
    assert quiet(_raw([0, 0, 0], [9, 9, 9, 9], [[0] * 7] * 2), 1, 1)


@pytest.mark.slow
def test_quiet_cut_offs_are_not_cut_when_set(tmp_path, monkeypatch):
    from nowcastfoundation.finetune import chronos as finetune_chronos
    from nowcastfoundation.finetune.stream import cut

    ids = sim_dir(tmp_path)
    full = [b for _, b in cut(tmp_path, ids, 1, 32)]
    # Unset, `quiet` is not asked.
    monkeypatch.setattr(finetune_chronos, "quiet", None)
    assert len([b for _, b in cut(tmp_path, ids, 1, 32)]) == len(full)
    # Set, exactly the blocks it calls quiet are dropped, with the days
    # per step of their source; the others are cut as before.
    asked = []

    def weekly_is_quiet(_block, days_per_step, min_cases):
        asked.append((days_per_step, min_cases))
        return days_per_step == 7

    monkeypatch.setattr(finetune_chronos, "quiet", weekly_is_quiet)
    monkeypatch.setattr(finetune_chronos, "QUIET_MIN_CASES", 10)
    kept = [b for _, b in cut(tmp_path, ids, 1, 32)]
    assert {d for d, _ in asked} >= {1, 7}
    assert {k for _, k in asked} == {10}
    rest = [b for b in full if b["source"] != "weekly"]
    assert len(kept) == len(rest) < len(full)
    for a, b in zip(rest, kept, strict=True):
        assert (a["source"], a["places"]) == (b["source"], b["places"])
        np.testing.assert_array_equal(a["context"], b["context"])


@pytest.mark.parametrize("start", [5, 40])
def test_padding_puts_each_step_at_its_place_after_the_context(start):
    # Each cell holds its step, so a step moved by padding or cropping
    # shows. Context of 16 steps: one block is padded, the other cropped.
    h = 9
    steps = np.arange(start + h, dtype=np.float32)
    block = {
        "context": np.tile(steps, (3, 1)),
        "start": start,
        "horizon": h,
        "nowcast": 2,
        "n_targets": 1,
        "n_covariates": 2,
        "n_future_covariates": 2,
    }
    out = ft._pad(block, 16)
    x = out["context"]
    keep = min(start, 16)
    np.testing.assert_array_equal(
        x[:, 16 - keep : 16 + h], block["context"][:, start - keep :]
    )
    assert np.isnan(x[:, : 16 - keep]).all()
    assert np.isnan(x[:, 16 + h :]).all()
    # The first predicted step is the block's first step after its context.
    assert (out["future_covariates"][:, 0] == start).all()


def test_batches_share_one_prediction_length():
    from nowcastfoundation.finetune.chronos import plan_by_length

    rng = np.random.default_rng(1)
    lengths = np.array([16] * 30 + [64] * 30 + [80] * 10)
    sizes = rng.integers(5, 300, len(lengths))
    plan = plan_by_length(lengths, sizes, 256, rng, 2000)
    used = sorted(i for length, batch in plan for i in batch)
    assert used == list(range(len(lengths)))
    for length, batch in plan:
        assert set(lengths[batch]) == {length}
        assert sizes[batch].sum() <= 2000 or len(batch) == 1


def test_a_seed_list_gives_the_training_seeds_in_its_order(tmp_path):
    from nowcastfoundation.finetune.stream import read_seed_list

    path = tmp_path / "seeds.txt"
    path.write_text("10005\n10001\n\n10003\n")
    assert read_seed_list(path) == [10005, 10001, 10003]


def test_a_round_continues_from_the_previous_adapter(tmp_path):
    import pytest

    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from peft import LoraConfig

    from nowcastfoundation.finetune.stream import lora_model

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for name in "qkvo":
                setattr(self, name, torch.nn.Linear(4, 4))

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attention = Attention()

    config = LoraConfig(r=2, lora_alpha=4, target_modules=["self_attention.q"])
    torch.manual_seed(0)
    first = lora_model(Net(), config, None)
    for p in first.parameters():
        if p.requires_grad:
            p.data.normal_()
    first.save_pretrained(tmp_path / "round0")
    torch.manual_seed(1)
    second = lora_model(Net(), config, tmp_path / "round0")
    trained = {n: p for n, p in first.named_parameters() if "lora" in n}
    loaded = {n: p for n, p in second.named_parameters() if "lora" in n}
    assert loaded.keys() == trained.keys()
    assert all(torch.equal(loaded[n], trained[n]) for n in trained)
    # The loaded adapter trains on.
    assert all(p.requires_grad for p in loaded.values())


def test_real_training_blocks_spread_evenly_over_the_chunks():
    from nowcastfoundation.finetune.stream import real_share

    real = list(range(10))
    parts = [real_share(real, chunk, 4) for chunk in range(4)]
    assert sorted(x for p in parts for x in p) == real
    assert {len(p) for p in parts} <= {2, 3}


def test_each_draw_is_cut_at_every_snapshot_once():
    from nowcastfoundation.finetune.chronos import CUTS, SOURCES
    from nowcastfoundation.finetune.stream import snapshots

    draws = [f"9001-0001-{d}" for d in range(1, 11)]
    picks = snapshots(draws, np.random.default_rng(0))
    # 10 draws x 32 cut-offs: 320 blocks, one source each.
    assert CUTS == 32
    assert len(picks) == 320
    assert len({(o, j) for o, _, j in picks}) == 320
    assert {s for _, s, _ in picks} <= set(SOURCES)


def test_level_mix_includes_single_series():
    rng = np.random.default_rng(0)
    mixes = [choose_places(STRATA, 10, rng) for _ in range(400)]
    single = [places for places, mix in mixes if mix == "single"]
    assert single
    assert all(len(places) == 1 and places[0] in STRATA for places in single)


def test_a_finished_batch_reads_every_outbreak(tmp_path, monkeypatch):
    from nowcastfoundation.finetune import stream as finetune_stream

    sims = tmp_path / "sims"
    sims.mkdir()
    sim_dir(sims)
    monkeypatch.setattr(finetune_stream, "_READY", {})
    assert finetune_stream.batch_outbreaks(sims, 6001) == [
        "6001-0001",
        "6001-0002",
    ]


def test_a_pass_samples_draws_across_every_snapshot():
    from nowcastfoundation.finetune.stream import snapshots

    draws = [f"9001-0001-{d}" for d in range(1, 11)]
    a = snapshots(draws, np.random.default_rng(0), n_draws=2, subset_seed=7)
    b = snapshots(draws, np.random.default_rng(5), n_draws=2, subset_seed=7)
    c = snapshots(draws, np.random.default_rng(0), n_draws=2, subset_seed=8)
    # 32 snapshots, two draws each: 64 blocks, spread over the 10 draws.
    assert len(a) == 64
    assert len({(o, j) for o, _, j in a}) == 64
    assert all(sum(j == k for _, _, j in a) == 2 for k in range(32))
    assert len({o for o, _, _ in a}) > 2
    # The subset seed alone picks the draws, so a later pass can take others.
    assert {(o, j) for o, _, j in b} == {(o, j) for o, _, j in a}
    assert {(o, j) for o, _, j in c} != {(o, j) for o, _, j in a}


def test_stale_block_files_fail_when_loaded(tmp_path):
    import pickle

    import pytest

    from nowcastfoundation.finetune.stream import load_blocks

    good = {
        "context": np.zeros((1, 4)),
        "n_targets": 1,
        "prediction_length": 16,
        "levels": ("national",),
    }
    stale = {k: v for k, v in good.items() if k != "prediction_length"}
    path = tmp_path / "blocks.pkl"
    path.write_bytes(pickle.dumps([good]))
    assert [b["prediction_length"] for b in load_blocks(path)] == [16]
    path.write_bytes(pickle.dumps([good, stale]))
    with pytest.raises(ValueError, match="prediction_length"):
        load_blocks(path)


def test_every_source_is_native_or_binned():
    ft.check_sources(("weekly", "daily", "3day"))
    # A source with no binning would be cut as daily: refuse it.
    with pytest.raises(ValueError, match="5day"):
        ft.check_sources(("weekly", "5day"))


def test_the_package_must_be_this_worktrees(tmp_path):
    ft.check_package(Path(__file__).parents[3])
    with pytest.raises(RuntimeError, match="not this worktree"):
        ft.check_package(tmp_path)


def _fake_torch(monkeypatch):
    import types

    torch = types.ModuleType("torch")
    torch.save = lambda obj, path: Path(path).write_text(json.dumps(obj))
    torch.load = lambda path, **_: json.loads(Path(path).read_text())
    # The random state the trainer saves with its state (CPU only).
    torch.get_rng_state = lambda: [0]
    torch.set_rng_state = lambda _: None
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: False)
    )
    peft = types.ModuleType("peft")
    peft.get_peft_model_state_dict = lambda model: model["adapter"]
    peft.set_peft_model_state_dict = lambda model, sd: model.update(adapter=sd)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "peft", peft)


class _Optimiser:
    def __init__(self, step):
        self.step = step

    def state_dict(self):
        return {"step": self.step}

    def load_state_dict(self, sd):
        self.step = sd["step"]


def test_a_crash_while_saving_state_keeps_the_last_state(tmp_path, monkeypatch):
    from nowcastfoundation.finetune.stream import load_state, save_state

    _fake_torch(monkeypatch)
    where = tmp_path / "state"
    save_state(where, {"adapter": [1]}, _Optimiser(1), {"step": 50})
    rename = Path.rename

    def crash(self, target):
        if self.name == "state.tmp":
            raise OSError("crash")
        return rename(self, target)

    monkeypatch.setattr(Path, "rename", crash)
    with pytest.raises(OSError, match="crash"):
        save_state(where, {"adapter": [2]}, _Optimiser(2), {"step": 100})
    monkeypatch.setattr(Path, "rename", rename)
    # The restarted trainer resumes from the last state that was saved.
    model, optimiser = {"adapter": None}, _Optimiser(0)
    assert load_state(where, model, optimiser) == {"step": 50}
    assert model["adapter"] == [1]
    assert optimiser.step == 1
    # The next save replaces it and leaves no old copy.
    save_state(where, {"adapter": [3]}, _Optimiser(3), {"step": 150})
    assert load_state(where, model, optimiser) == {"step": 150}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["state"]


def test_the_code_commit_is_the_trainers_repository(tmp_path, monkeypatch):
    from nowcastfoundation.finetune.stream import code_commit

    here = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    # Launched from another repository, it still names this one's commit.
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    monkeypatch.chdir(tmp_path)
    assert code_commit() == here


@pytest.mark.slow
def test_mixed_blocks_keep_each_target_rows_place_and_level(monkeypatch):
    import pandas as pd

    from nowcastfoundation.finetune import chronos as ft

    rows = [
        (place, onset, onset + pd.Timedelta(days=d), n)
        for onset in pd.date_range("2026-01-03", periods=140)
        for place, n in (("P01", 5), ("P02", 3))
        for d in (0, 2, 6)
    ]
    counts = pd.DataFrame(
        rows, columns=["location", "reference_date", "report_date", "count"]
    )
    counts["reference_date"] = counts["reference_date"].dt.date.astype(str)
    counts["report_date"] = counts["report_date"].dt.date.astype(str)
    monkeypatch.setattr(ft, "SOURCES", ("daily+weekly",))
    monkeypatch.setattr(ft, "UNIVARIATE_SHARE", 0.0)
    seen = 0
    for seed in range(5):
        rng = np.random.default_rng(seed)
        for b in ft.outbreak_blocks(counts, ["P01", "P02"], rng, 3):
            n = len(b["places"])
            assert list(b["target_places"]) == [*range(n), *range(n)]
            assert len(b["levels"]) == b["n_targets"] == 2 * n
            seen += 1
    assert seen


def test_block_options_vary_the_period_end_and_never_batch_reports():
    from nowcastfoundation.finetune import chronos as finetune_chronos
    from nowcastfoundation.finetune.chronos import block_options

    rng = np.random.default_rng(0)
    weekly = [block_options("daily+weekly", rng) for _ in range(3000)]
    # Weekly batches come from the grid's weekday reporting alone.
    assert all(set(o) == {"period_end"} for o in weekly)
    assert not hasattr(finetune_chronos, "REPORT_BATCH_SHARE")
    assert {o["period_end"] for o in weekly} == set(range(7))
    assert {
        block_options("daily+3day", rng)["period_end"] for _ in range(300)
    } == {0, 1, 2}
    assert block_options("weekly", rng)["period_end"] is None


def test_one_draw_of_a_simulation_gives_32_blocks_of_a_year_s_context():
    from nowcastfoundation.finetune.chronos import CONTEXT, CUTS
    from nowcastfoundation.finetune.stream import snapshots

    picks = snapshots(["9001-0001-1"], np.random.default_rng(0))
    assert CONTEXT == 364
    assert CUTS == 32
    assert len(picks) == 32


def test_a_batch_that_fits_is_not_cleared_first():
    from nowcastfoundation.finetune.stream import accumulate

    zeroed: list[int] = []

    def backward(part, _share):
        if len(part) > 2:
            raise RuntimeError("CUDA out of memory")
        return 0.5

    # The step before cleared the gradients and freed memory.
    assert accumulate([1, 2], backward, lambda: zeroed.append(1)) == 0.5
    assert zeroed == []
    # Each retry after running out of memory starts from clear gradients.
    assert accumulate([1, 2, 3], backward, lambda: zeroed.append(1))
    assert zeroed == [1]


def test_one_non_finite_gradient_skips_the_step():
    torch = pytest.importorskip("torch")
    from nowcastfoundation.finetune.stream import finite_step

    layer = torch.nn.Linear(2, 1)
    params = list(layer.parameters())
    optimiser = torch.optim.AdamW(params, lr=0.1)
    before = [p.detach().clone() for p in params]
    layer(torch.ones(1, 2)).sum().backward()
    params[1].grad[...] = float("inf")
    assert not finite_step(params, optimiser, 1.0)
    assert all(torch.equal(a, b) for a, b in zip(before, params, strict=True))
    assert all(p.grad is None for p in params)


def test_settled_objects_are_skipped_by_collections_until_the_next():
    import gc
    import weakref

    from nowcastfoundation.finetune.stream import settle_heap

    class Node:
        pass

    def cycle() -> weakref.ref:
        a, b = Node(), Node()
        a.other, b.other = b, a
        return weakref.ref(a)

    dead = cycle()
    kept = Node()
    try:
        settle_heap()
        # Garbage from before is collected; what survives is frozen.
        assert dead() is None
        assert gc.get_freeze_count() > 0
        assert all(o is not kept for o in gc.get_objects())
        # A frozen cycle that becomes garbage waits for the next settle.
        holder = [Node()]
        holder[0].me = holder[0]
        settle_heap()
        ref = weakref.ref(holder[0])
        holder.clear()
        gc.collect()
        assert ref() is not None
        settle_heap()
        assert ref() is None
    finally:
        gc.unfreeze()


def test_steps_in_a_row_do_not_sum_gradients():
    torch = pytest.importorskip("torch")
    from nowcastfoundation.finetune.stream import accumulate, finite_step

    def run(clear_between: bool) -> list:
        torch.manual_seed(0)
        layer = torch.nn.Linear(2, 1)
        params = list(layer.parameters())
        optimiser = torch.optim.SGD(params, lr=0.1)

        def backward(part, share):
            loss = layer(torch.full((len(part), 2), 1.0)).sum()
            (loss * share).backward()
            return float(loss)

        for _ in range(2):
            if clear_between:
                optimiser.zero_grad()
            value = accumulate([1, 2], backward, optimiser.zero_grad)
            finite_step(params, optimiser, value)
        return [p.detach().clone() for p in params]

    # finite_step leaves the gradients clear, so the next step starts
    # from none, as if cleared explicitly.
    for a, b in zip(run(False), run(True), strict=True):
        assert torch.equal(a, b)


def test_every_layout_has_cadences_whose_shares_sum_to_one():
    for source in ft.SOURCE_SHARES:
        mix = ft.CADENCES[ft.native_cadence(source)]
        assert sum(mix.values()) == pytest.approx(1)
    assert ft.native_cadence("daily+rolling7") == "daily"
    assert ft.native_cadence("3day") == "3day"
    assert ft.native_cadence("weekly") == "weekly"


@pytest.mark.parametrize("source", ["daily", "3day", "weekly"])
def test_each_layout_draws_its_cadences_at_their_shares(source):
    rng = np.random.default_rng(0)
    drawn = [ft.draw_cadence(source, rng) for _ in range(4000)]
    for cadence, share in ft.CADENCES[ft.native_cadence(source)].items():
        assert drawn.count(cadence) / len(drawn) == pytest.approx(
            share, abs=0.03
        )


def _lag_fill(b) -> float:
    n = b["n_targets"]
    ctx = b["context"][:, : ft.CONTEXT]
    seen = np.isfinite(ctx[0])
    return float(np.isfinite(ctx[n:, seen]).mean())


@pytest.mark.slow
def test_blocks_record_their_cadence_and_daily_releases_fill_lags(
    tmp_path, monkeypatch
):
    from nowcastfoundation.finetune import chronos as finetune_chronos
    from nowcastfoundation.finetune.stream import cut

    ids = sim_dir(tmp_path)
    monkeypatch.setattr(finetune_chronos, "SOURCES", ("daily",))
    fills = {}
    for cadence in ("daily", "weekly"):
        monkeypatch.setitem(finetune_chronos.CADENCES, "daily", {cadence: 1.0})
        blocks = [b for _, b in cut(tmp_path, ids, 1, 32)]
        assert blocks
        assert {b["cadence"] for b in blocks} == {cadence}
        fills[cadence] = np.mean([_lag_fill(b) for b in blocks])
    assert fills["daily"] > 0.8
    assert fills["weekly"] < 0.3


def test_evaluation_cadence_defaults_to_the_recipe_s():
    from nowcastfoundation.finetune import stream as fs

    args = fs.build_parser().parse_args(
        ["sims", "out", "--held-out-seeds", "1", "2"]
    )
    assert args.eval_every == fs.ft.EVAL_EVERY == 100
    assert args.by_source_every == fs.BY_SOURCE_EVERY == 200
    fs.check_cadence(500, 1000)
    # By source runs with the monitor, so on a multiple of its steps.
    with pytest.raises(ValueError, match="multiple"):
        fs.check_cadence(300, 1000)
    with pytest.raises(ValueError, match="positive"):
        fs.check_cadence(0, 0)


def test_a_missing_grid_file_is_refused(tmp_path):
    from nowcastfoundation.finetune import chronos as finetune_chronos

    with pytest.raises(FileNotFoundError, match=r"nowhere\.yaml"):
        finetune_chronos.use_grid(tmp_path / "nowhere.yaml")


def test_the_trainer_takes_its_design_from_the_grid():
    from nowcastfoundation.finetune import chronos as finetune_chronos
    from nowcastfoundation.finetune import stream as fs

    args = fs.build_parser().parse_args(
        ["sims", "out", "--held-out-seeds", "1", "2"]
    )
    assert args.sources is None
    try:
        finetune_chronos.use_grid(args.grid_file)
        assert tuple(args.sources or finetune_chronos.SOURCE_SHARES) == (
            "daily",
            "weekly",
            "3day",
            "daily+weekly",
            "daily+rolling7",
            "daily+3day",
            "daily+rolling3",
        )
    finally:
        finetune_chronos.PARENTS = None


def test_a_whole_system_block_holds_every_stratum_and_its_totals(monkeypatch):
    from nowcastfoundation.finetune.chronos import NATIONAL, whole_system

    monkeypatch.setattr(ft, "UNIVARIATE_SHARE", 0.0)
    rng = np.random.default_rng(0)
    # Places by ages: the country, each place's and each age's total, and
    # every leaf.
    places, mix = whole_system(STRATA, rng)
    assert mix == "all_levels"
    assert places == (
        NATIONAL,
        *[f"P{p}" for p in range(1, 5)],
        *[f"A{a}" for a in range(1, 4)],
        *sorted(STRATA),
    )
    # Places nested in regions: the regions' totals and every place.
    nested = ["R1-P01", "R1-P02", "R2-P03"]
    assert whole_system(nested, rng) == (
        (NATIONAL, "R1", "R2", *nested),
        "all_levels",
    )
    # Flat places: the country and every place.
    flat = ["P01", "P02", "P03"]
    assert whole_system(flat, rng) == ((NATIONAL, *flat), "leaves_national")
    # One stratum is the country.
    assert whole_system(["P01"], rng) == ((NATIONAL,), "national")
    # At the univariate share, the country alone.
    monkeypatch.setattr(ft, "UNIVARIATE_SHARE", 1.0)
    assert whole_system(STRATA, rng) == ((NATIONAL,), "national")


def _counts_of(places, days=140):
    import pandas as pd

    rows = [
        (p, onset, onset + pd.Timedelta(days=d), 3)
        for p in places
        for onset in pd.date_range("2026-01-03", periods=days)
        for d in (0, 3, 9)
    ]
    counts = pd.DataFrame(
        rows, columns=["location", "reference_date", "report_date", "count"]
    )
    for col in ("reference_date", "report_date"):
        counts[col] = counts[col].dt.date.astype(str)
    return counts


@pytest.mark.slow
def test_blocks_are_whole_systems_never_cut(monkeypatch):
    from nowcastfoundation.finetune.chronos import NATIONAL, outbreak_blocks

    ft.use_grid()
    try:
        monkeypatch.setattr(ft, "UNIVARIATE_SHARE", 0.2)
        places = [f"P{p}-A{a}" for p in range(1, 4) for a in range(1, 3)]
        counts = _counts_of(places)
        whole = (NATIONAL, "P1", "P2", "P3", "A1", "A2", *places)
        got = []
        for seed in range(6):
            for b in outbreak_blocks(
                counts, places, np.random.default_rng(seed), 3
            ):
                got.append(b["places"])
        assert got
        # The whole system, or the country alone; never a subset.
        assert set(got) == {whole, (NATIONAL,)}
        # A system over the series cap is still cut whole: no block is
        # ever skipped for its size.
        monkeypatch.setattr(ft, "MAX_SERIES", 10)
        empty: list[str] = []
        blocks = outbreak_blocks(
            counts, places, np.random.default_rng(0), 3, empty=empty
        )
        assert any(b["places"] == whole for b in blocks)
        assert not any("series cap" in e for e in empty)
    finally:
        ft.PARENTS = None


@pytest.mark.slow
def test_parents_come_from_the_grid_file(monkeypatch):
    import yaml

    from nowcastfoundation.finetune.chronos import NATIONAL, outbreak_blocks

    grid = yaml.safe_load(
        (Path(__file__).parents[3] / "config/finetune/grid.yaml").read_text()
    )
    ft.use_grid()
    try:
        # The grid's weights of each kind of parent, and its named
        # distributions of the share and the lag.
        weights = grid["aggregation"]["parents"]
        total = sum(weights.values())
        assert ft.PARENTS["shares"] == {
            k: w / total for k, w in weights.items()
        }
        dists = grid["distributions"]
        assert ft.PARENTS["unallocated_share"] == dists["unallocated_share"]
        assert ft.PARENTS["parent_lag_days"] == dists["parent_lag_days"]
        monkeypatch.setattr(ft, "UNIVARIATE_SHARE", 0.0)
        monkeypatch.setitem(
            ft.PARENTS,
            "shares",
            {"exact": 0.0, "unallocated": 1.0, "late_parent": 0.0},
        )
        places = [f"P{p}-A{a}" for p in range(1, 3) for a in range(1, 3)]
        counts = _counts_of(places)
        parents = {
            b["parent"]
            for seed in range(3)
            for b in outbreak_blocks(
                counts, places, np.random.default_rng(seed), 3
            )
            if len(b["places"]) > 1 and b["places"][0] == NATIONAL
        }
        assert parents == {"unallocated"}
    finally:
        ft.PARENTS = None
    assert ft.PARENTS is None


def test_early_cut_offs_take_as_many_lags_as_history_allows(monkeypatch):
    # Early cut-offs have no complete steps to read a delay from, so the
    # shared rule takes as many lags as the history allows: they are kept,
    # not skipped.
    from nowcastfoundation.finetune.chronos import NATIONAL, outbreak_blocks

    monkeypatch.setattr(ft, "SOURCES", ("daily",))
    monkeypatch.setattr(ft, "UNIVARIATE_SHARE", 1.0)
    counts = _counts_of(["P1", "P2"], days=63)
    blocks = outbreak_blocks(
        counts, ["P1", "P2"], np.random.default_rng(0), 1, 4
    )
    assert len(blocks) == 4
    for b in blocks:
        assert b["places"] == (NATIONAL,)
        lag = b["n_covariates"] // len(b["places"])
        # Every lag series and two steps of context fit in the history.
        assert 1 <= lag < 63
        assert b["nowcast"] == lag


def test_views_of_a_system_over_capacity_cover_every_leaf():
    from nowcastfoundation.finetune.chronos import NATIONAL, system_views

    nested = [f"P{p}-A{a}" for p in range(1, 4) for a in range(1, 3)]
    whole = (NATIONAL, "P1", "P2", "P3", "A1", "A2", *nested)
    # Within capacity: the whole system.
    assert system_views(whole, len(whole)) == [("whole", whole, [])]
    # Over it: the top levels aggregated up, and each place with its ages
    # and its own total.
    assert system_views(whole, 5) == [
        ("top", (NATIONAL, "P1", "P2", "P3"), ["strata"]),
        ("subtree", ("P1", "P1-A1", "P1-A2"), []),
        ("subtree", ("P2", "P2-A1", "P2-A2"), []),
        ("subtree", ("P3", "P3-A1", "P3-A2"), []),
    ]
    # A subtree over capacity is its total with as many leaves as fit.
    assert system_views(whole, 2)[1:] == [
        ("subtree", (f"P{p}", f"P{p}-A{a}"), [])
        for p in range(1, 4)
        for a in range(1, 3)
    ]
    # The top collapses further until it fits.
    assert system_views(whole, 2)[0] == (
        "top",
        (NATIONAL,),
        ["strata", "places"],
    )
    # Room for one place: the country, and each leaf alone.
    assert system_views(whole, 1) == [
        ("top", (NATIONAL,), ["strata", "places"]),
        *[("subtree", (p,), []) for p in nested],
    ]
    # Flat places: the country, and its places in groups with it.
    flat = (NATIONAL, "P01", "P02", "P03")
    assert system_views(flat, 3) == [
        ("top", (NATIONAL,), ["places"]),
        ("subtree", (NATIONAL, "P01", "P02"), []),
        ("subtree", (NATIONAL, "P03"), []),
    ]


@pytest.mark.slow
def test_a_system_over_capacity_is_cut_into_views_with_exact_totals(
    monkeypatch,
):
    from nowcastfoundation.finetune.chronos import NATIONAL, outbreak_blocks

    ft.use_grid()
    try:
        monkeypatch.setattr(ft, "UNIVARIATE_SHARE", 0.0)
        monkeypatch.setitem(
            ft.PARENTS,
            "shares",
            {"exact": 1.0, "unallocated": 0.0, "late_parent": 0.0},
        )
        places = [f"P{p}-A{a}" for p in range(1, 4) for a in range(1, 3)]
        counts = _counts_of(places)
        whole = (NATIONAL, "P1", "P2", "P3", "A1", "A2", *places)
        monkeypatch.setattr(ft, "BLOCK_LIMIT", None)
        full = outbreak_blocks(counts, places, np.random.default_rng(0), 3)
        assert full
        assert all(b["places"] == whole for b in full)
        assert all(b["view"] == {"kind": "whole"} for b in full)
        series = [b["context"].shape[0] for b in full]
        # Room for the country and its places in the largest blocks, not
        # for their whole system.
        limit = -(-max(series) * 5 // len(whole))
        over = [n > limit for n in series]
        assert any(over)
        monkeypatch.setattr(ft, "BLOCK_LIMIT", limit)
        cut = outbreak_blocks(counts, places, np.random.default_rng(0), 3)
        assert len(cut) == len(full) + 3 * sum(over)
        at = 0
        for k, before in enumerate(full):
            if not over[k]:
                # A system within the limit stays whole.
                assert cut[at]["places"] == whole
                assert cut[at]["view"] == {"kind": "whole"}
                at += 1
                continue
            views = cut[at : at + 4]
            at += 4
            kinds = [v["view"]["kind"] for v in views]
            assert kinds == ["top", "subtree", "subtree", "subtree"]
            assert views[0]["view"]["collapsed"] == ["strata"]
            # Every leaf is in a block.
            assert {p for v in views for p in v["places"] if "-" in p} == set(
                places
            )
            for after in views:
                n = after["context"].shape[0]
                assert n <= limit
                assert after["cut"] == before["cut"]
                assert after["view"]["series_before"] == series[k]
                assert after["view"]["series_after"] == n
                assert after["view"]["places_before"] == len(whole)
                assert after["view"]["places_after"] == len(after["places"])
                # Totals stay exact: each place's series is the whole
                # system's.
                for i, place in enumerate(after["places"]):
                    row = after["target_places"].index(i)
                    was = before["target_places"].index(whole.index(place))
                    np.testing.assert_array_equal(
                        after["context"][row], before["context"][was]
                    )
    finally:
        ft.PARENTS = None


def test_the_grid_file_has_a_default_outside_a_checkout(monkeypatch, tmp_path):
    # Installed alone, the package has no workflow/config.yaml to read.
    monkeypatch.setattr(ft, "ROOT", tmp_path)
    assert ft.workflow_grid_file() == ft.DEFAULT_GRID_FILE


def test_a_complete_block_is_the_ragged_block_without_its_lag_series():
    from nowcastfoundation.finetune.chronos import complete_block

    nan = np.nan
    lags = [[0, 0, 0, 2, 3, nan, nan], [0, 0, 0, 4, nan, nan, nan]]
    raw = _raw([5, 0, 1], [6, 7, 8, 9], lags)
    out = complete_block(raw)
    # The counts to the last complete step, then the truth: no lag series,
    # so the context ends at `start`.
    np.testing.assert_array_equal(out["context"], raw["context"][:1])
    assert (out["n_covariates"], out["n_future_covariates"]) == (0, 0)
    assert (out["start"], out["nowcast"]) == (3, 2)


@pytest.mark.slow
def test_the_complete_layout_cuts_the_ragged_blocks_without_lag_series(
    monkeypatch,
):
    from nowcastfoundation.finetune.chronos import outbreak_blocks

    ft.use_grid()
    counts = _counts_of(["P01", "P02"])
    ragged = outbreak_blocks(counts, ["P01", "P02"], np.random.default_rng(3))
    monkeypatch.setattr(ft, "LAYOUT", "complete")
    complete = outbreak_blocks(counts, ["P01", "P02"], np.random.default_rng(3))
    # The same blocks in the same order, so both runs see the same data.
    assert len(complete) == len(ragged) > 0
    for r, c in zip(ragged, complete, strict=True):
        assert (c["source"], c["cut"], c["places"]) == (
            r["source"],
            r["cut"],
            r["places"],
        )
        np.testing.assert_array_equal(
            c["context"], r["context"][: r["n_targets"]]
        )
        assert c["n_covariates"] == c["n_future_covariates"] == 0
