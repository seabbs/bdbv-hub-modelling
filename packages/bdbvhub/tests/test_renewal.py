from __future__ import annotations

import datetime as dt
import json
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from bdbvhub.models import get_model, list_models
from bdbvhub.submission.hub import model_abbr_for
from nowcastbdbv.backtest import make_tasks
from nowcasteval.backtest import BacktestConfig
from nowcastrenewal import forecaster as rn
from nowcastutils.base import (
    NATIONAL,
    ForecastModel,
    ForecastTask,
)
from nowcastutils.dates import epiweek_end

pytest.importorskip("pyrenew")


@pytest.fixture
def task(target: pd.DataFrame, delays: pd.DataFrame) -> ForecastTask:
    rows = target.loc[target["vintage"] == "2026-10-17"]
    return make_tasks(rows, delays, BacktestConfig(n_samples=20, seed=3))[0]


def test_variants_are_registered_as_fitted_models() -> None:
    specs = {s.name: s for s in list_models()}
    for name in ("renewal", "renewalsp"):
        assert specs[name].fitted
        model = get_model(name)
        assert isinstance(model, ForecastModel)
        assert model.name == name
    # Sam (lead 2026-11-24 12:32): renewalsp is the single-patch model
    # fitted per zone; the per-province fit and its zone split are gone.
    assert set(rn.VARIANTS) == {"renewal", "renewalsp", "renewalbvd"}
    assert "renewalsp_zones" not in specs
    assert not hasattr(rn, "with_zones")
    field_names = {f.name for f in fields(rn.Variant)}
    assert {"zones", "sampler_of"}.isdisjoint(field_names)
    # The pooled variants can forecast a province with its data held out;
    # renewalsp fits each province on its own data only, so it cannot.
    assert get_model("renewal").supports_mask
    assert get_model("renewalbvd").supports_mask
    assert not get_model("renewalsp").supports_mask
    assert not specs["randomwalk"].fitted
    # Hub model abbreviations: letters, digits and underscores, under 16.
    for name in specs:
        assert len(model_abbr_for(name)) < 16


def test_only_renewalsp_centres_its_rt_walk() -> None:
    # One place's data pin its Rt walk, so its levels are sampled.
    # Pooled, the walk's step SD mixes badly centred.
    centred = {
        name: rn.model_config(v).centred_rt for name, v in rn.VARIANTS.items()
    }
    assert centred == {
        "renewal": False,
        "renewalsp": True,
        "renewalbvd": False,
    }


def test_renewalsp_centres_only_where_the_data_pin_the_walk() -> None:
    # Centred, a province with few cases funnels.
    def data(cases: float) -> SimpleNamespace:
        known = np.array([[cases - 1.0], [np.nan]])
        return SimpleNamespace(known=known, imputed=np.array([[1.0], [0.0]]))

    sp = rn.VARIANTS["renewalsp"]
    rich = rn.fit_config(sp, data(rn.CENTRED_RT_MIN_CASES))
    poor = rn.fit_config(sp, data(rn.CENTRED_RT_MIN_CASES - 1))
    assert rich.centred_rt
    assert not poor.centred_rt
    assert poor == replace(rich, centred_rt=False)
    pooled = rn.fit_config(rn.VARIANTS["renewal"], data(1e6))
    assert not pooled.centred_rt


@pytest.mark.parametrize("name", ["renewal", "renewalsp", "renewalbvd"])
def test_renewal_model_fits_provinces_only_on_their_own(
    task: ForecastTask, name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def ran(*_: object) -> rn.RenewalRun:
        raise AssertionError("fitted")

    monkeypatch.setattr(rn.RenewalModel, "run", ran)
    # Every variant fits zones (`nowcastrenewal.zones`); the province
    # fits they replaced are gone.
    with pytest.raises(NotImplementedError, match="health zones"):
        rn.RenewalModel(rn.VARIANTS[name]).forecast_fit(task)


def test_the_variants_are_ablations_of_renewalbvd() -> None:
    # renewal drops the BVD Rt term. renewalsp also drops pooling and
    # importation by fitting each province alone (`national`), with its
    # walk centred where the data pin it (sampler geometry only).
    # Depletion and the Rt process are the same in all three.
    import dataclasses

    full = rn.model_config(rn.VARIANTS["renewalbvd"])

    def dropped(name: str) -> set[str]:
        config = rn.model_config(rn.VARIANTS[name])
        return {
            f.name
            for f in dataclasses.fields(config)
            if getattr(config, f.name) != getattr(full, f.name)
        }

    assert full == rn.engine().BVD
    assert full.depletion
    assert dropped("renewal") == {"bvd"}
    # renewalsp also keeps its centred walk's step SD off 0.
    assert dropped("renewalsp") == {
        "bvd",
        "national",
        "centred_rt",
        "rt_sd_lognormal",
    }
    assert rn.VARIANTS["renewalsp"].per_province
    assert rn.model_config(rn.VARIANTS["renewal"]) == rn.engine().MAIN


def test_sampler_settings_read_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(rn.SAMPLER_ENV, '{"num_chains": 1, "num_warmup": 5}')
    settings = rn.sampler_settings()
    assert settings["num_chains"] == 1
    assert settings["num_warmup"] == 5
    assert settings["num_samples"] == rn.DEFAULT_SAMPLER["num_samples"]
    assert rn.sampler_settings({"num_chains": 3})["num_chains"] == 3


def test_chain_diagnostics() -> None:
    rng = np.random.default_rng(1)
    n = 200
    extra = {
        "diverging": np.zeros((2, n), dtype=bool),
        "num_steps": np.full((2, n), 7),
        "accept_prob": np.full((2, n), 0.8),
        "energy": rng.normal(size=(2, n)).cumsum(axis=1),
        "adapt_state.step_size": np.full((2, n), 0.05),
    }
    extra["diverging"][1, :3] = True
    extra["num_steps"][0, :50] = 2**10 - 1
    chains = rn.chain_diagnostics(extra, max_tree_depth=10)
    assert [c["chain"] for c in chains] == [0, 1]
    assert chains[0]["draws"] == n
    assert chains[1]["divergences"] == 3
    assert chains[0]["tree_depth_hits"] == 50
    assert chains[0]["tree_depth_share"] == pytest.approx(0.25)
    assert chains[1]["tree_depth_share"] == 0.0
    assert chains[0]["mean_accept"] == pytest.approx(0.8)
    assert chains[0]["step_size"] == pytest.approx(0.05)
    # A random walk in energy has a small E-BFMI; white noise near 2.
    assert chains[0]["ebfmi"] < 0.3
    white = rn.ebfmi(rng.normal(size=4000))
    assert white == pytest.approx(2.0, rel=0.1)


def test_parameter_diagnostics_and_worst_rhat() -> None:
    rng = np.random.default_rng(2)
    trace = {
        "good": rng.normal(size=(2, 300, 3)),
        "bad": np.stack([rng.normal(size=300), rng.normal(5.0, 1.0, size=300)]),
        "fixed": np.ones((2, 300)),
    }
    params = rn.parameter_diagnostics(trace)
    names = [(p["name"], p["index"]) for p in params]
    assert names == [
        ("bad", 0),
        ("fixed", 0),
        ("good", 0),
        ("good", 1),
        ("good", 2),
    ]
    fixed = params[1]
    assert fixed["rhat"] is None
    assert fixed["ess_bulk"] is None
    assert params[0]["rhat"] > 1.5
    worst = rn.worst_rhat(params, n=2)
    assert [w["name"] for w in worst] == ["bad", "good"]
    assert all(set(p) >= {"rhat", "ess_bulk", "ess_tail"} for p in params)


def test_vintage_inputs_model_every_task_province(task: ForecastTask) -> None:
    config = rn.model_config(rn.VARIANTS["renewal"])
    data = rn.vintage_inputs(task, config)
    assert data.names == list(task.locations)
    assert data.start_date.weekday() == 6
    assert data.known.shape == data.imputed.shape
    assert data.fit_mask is not None
    assert data.fit_mask.all()
    assert data.horizon_weeks == rn.horizon_weeks(task)


def test_vintage_inputs_apply_the_mask(task: ForecastTask) -> None:
    config = rn.model_config(rn.VARIANTS["renewal"])
    full = rn.vintage_inputs(task, config)
    masked = rn.vintage_inputs(replace(task, mask=("CD-NK",)), config)
    j = masked.names.index("CD-NK")
    assert masked.fit_mask is not None
    assert not masked.fit_mask[j]
    assert masked.fit_mask.sum() == len(masked.names) - 1
    # Its delays leave the delay data; it stays in the latent process.
    assert masked.delay_counts.sum() < full.delay_counts.sum()
    assert masked.names == full.names


def _fake_draws(data: Any, n: int, rng: np.random.Generator) -> dict[str, Any]:
    weeks = data.n_weeks + data.horizon_weeks
    province = rng.poisson(5.0, size=(n, weeks, len(data.names))).astype(float)
    insample = {
        "known": province * 0.7,
        "imputed": province * 0.3,
    }
    return {
        "province": province,
        "national": province.sum(2),
        "insample": insample,
        "rt": _fake_rt(data, 40, rng),
    }


def _fake_rt(data: Any, n: int, rng: np.random.Generator) -> dict[str, Any]:
    """`weekly_rt` output on a 42-day seeding axis."""
    weeks = 6 + data.n_weeks + data.horizon_weeks
    shape = (n, weeks, len(data.names))
    return {
        "rt": rng.lognormal(0.0, 0.3, size=shape),
        "weight": rng.gamma(2.0, 5.0, size=shape),
        "start": data.start_date - dt.timedelta(days=42),
    }


def test_samples_table_is_coherent(task: ForecastTask) -> None:
    # Every province of the release: only those with cases are forecast.
    rows = task.target.loc[task.target["location"] != NATIONAL]
    task = replace(task, locations=tuple(sorted(set(rows["location"]))))
    config = rn.model_config(rn.VARIANTS["renewal"])
    data = rn.vintage_inputs(task, config)
    draws = _fake_draws(data, task.n_samples, np.random.default_rng(1))
    samples = rn.samples_table(task, data, draws)
    assert set(samples["location"]) == set(data.names)
    assert set(samples["sample"]) == set(range(1, task.n_samples + 1))


def test_ppc_rows_add_national(task: ForecastTask) -> None:
    config = rn.model_config(rn.VARIANTS["renewal"])
    data = rn.vintage_inputs(task, config)
    draws = _fake_draws(data, 30, np.random.default_rng(3))
    observed, ppc = rn.ppc_rows(data, draws["insample"])
    assert set(observed["stream"]) == {"observed", "imputed"}
    assert NATIONAL in set(observed["location"])
    assert ppc.shape == (30, len(observed))
    nat = observed.loc[observed["location"] == NATIONAL]
    assert len(nat) == 2 * data.n_weeks


def _fake_run(task: ForecastTask, variant: rn.Variant) -> rn.RenewalRun:
    config = rn.model_config(variant)
    data = rn.vintage_inputs(task, config)
    draws = _fake_draws(data, task.n_samples, np.random.default_rng(4))
    rng = np.random.default_rng(5)
    n = 50
    trace = {"sd": rng.normal(size=(2, n)), "z": rng.normal(size=(2, n, 2))}
    extra = {
        "diverging": np.zeros((2, n), dtype=bool),
        "num_steps": np.full((2, n), 15),
        "accept_prob": np.full((2, n), 0.9),
        "energy": rng.normal(size=(2, n)),
        "adapt_state.step_size": np.full((2, n), 0.1),
    }
    # The counts as reported by each later vintage: one case a province.
    draws["as_reported"] = {
        cut: {
            "province": np.ones_like(draws["province"]),
        }
        for cut in rn.report_cutoffs(task, data)
    }
    return rn.RenewalRun(
        variant=variant,
        data=data,
        draws=draws,
        trace=trace,
        extra=extra,
        seconds=1.5,
        cpu_seconds=2.9,
        delay=pd.concat(
            rn.delay_summary(
                rng.normal(-1, 0.1, size=(n, 20)),
                rng.normal(0, 0.1, size=(n, 14)),
                data.start_date,
                13,
            ).assign(place=place)
            for place in data.names
        ),
    )


def test_rt_quantiles_give_the_hub_levels_by_infection_week(
    task: ForecastTask,
) -> None:
    runs = {
        loc: _fake_run(
            replace(task, locations=(loc,)), rn.VARIANTS["renewalsp"]
        )
        for loc in task.locations[:2]
    }
    out = rn.rt_quantiles(task, runs)
    assert list(out.columns) == [
        "location",
        "target_end_date",
        "output_type_id",
        "value",
    ]
    assert set(out["location"]) == {*task.locations[:2], NATIONAL}
    assert set(out["target_end_date"]) == set(task.target_end_dates)
    assert tuple(sorted(set(out["output_type_id"]))) == rn.RT_QUANTILES
    assert len(rn.RT_QUANTILES) == 23
    assert (out["value"] >= 0).all()
    for _, group in out.groupby(["location", "target_end_date"]):
        assert group["value"].is_monotonic_increasing
    # National pools the provinces' draws one by one, weighted by their
    # infections: not a mix of their quantiles.
    # Each fit has its own axis, so each takes its own week of the date.
    date = task.target_end_dates[-1]
    a, b = (
        {
            k: v[:, (epiweek_end(date) - r["start"]).days // 7, 0]
            for k, v in r.items()
            if k != "start"
        }
        for r in (runs[loc].draws["rt"] for loc in task.locations[:2])
    )
    joint = (a["weight"] * a["rt"] + b["weight"] * b["rt"]) / (
        a["weight"] + b["weight"]
    )
    nat = out.loc[
        (out["location"] == NATIONAL)
        & (out["target_end_date"] == date)
        & (out["output_type_id"] == 0.5)
    ]
    assert float(nat["value"].iloc[0]) == pytest.approx(np.median(joint))


def test_chains_start_from_their_own_prior_draws() -> None:
    from numpyro.infer import init_to_sample

    kernel = rn.nuts_kernel(lambda: None, rn.DEFAULT_SAMPLER)
    assert kernel._init_strategy is init_to_sample


def test_bvd_variant_reads_the_release_for_its_vintage(
    task: ForecastTask, tmp_path: Path
) -> None:
    import shutil

    from nowcastrenewal import bvd

    fixtures = (
        Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "bvd"
    )
    releases = bvd.parse_releases(
        json.loads((fixtures / "releases.json").read_text())
    )
    bvd.save_releases(
        releases.loc[releases["tag"].isin(["results-3151", "results-3086"])],
        tmp_path,
    )
    for tag in ("results-3151", "results-3086"):
        shutil.copytree(fixtures / tag, tmp_path / tag)
    rows = rn.bvd_rows(task, tmp_path)
    assert rows is not None
    assert rows["as_of"].iloc[0] == "2026-10-15"  # results-3086 for 17 Oct
    config = rn.model_config(rn.VARIANTS["renewalbvd"])
    data = rn.vintage_inputs(task, config, bvd=rows)
    assert data.bvd is not None
    assert rn.vintage_inputs(task, config).bvd is None
    assert "renewalbvd" in {s.name for s in list_models()}


def test_chain_method_defaults_to_vectorized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpyro.infer

    from nowcastrenewal.forecaster import SAMPLER_ENV

    # The backtest's config may have set the sampler for the process.
    monkeypatch.delenv(SAMPLER_ENV, raising=False)
    seen: dict[str, Any] = {}

    class FakeMCMC:
        def __init__(self, kernel: Any, **kw: Any) -> None:
            del kernel
            seen.update(kw)

        def run(self, *args: Any, **kw: Any) -> None:
            del args, kw

    class FakeModel:
        def validate_data(self, **kw: Any) -> None:
            del kw

        def sample(self) -> None:
            return None

    import jax

    cleared: list[bool] = []
    monkeypatch.setattr(jax, "clear_caches", lambda: cleared.append(True))
    monkeypatch.setattr(numpyro.infer, "MCMC", FakeMCMC)
    monkeypatch.setattr(rn.engine(), "model_args", lambda *_: {})
    settings = rn.sampler_settings()
    assert settings["chain_method"] == "vectorized"
    rn.run_nuts(FakeModel(), None, settings)
    assert seen["chain_method"] == "vectorized"
    # Each fit compiles its own model, so earlier executables are dropped.
    assert cleared == [True]
    rn.run_nuts(FakeModel(), None, {**settings, "chain_method": "sequential"})
    assert seen["chain_method"] == "sequential"


def test_inputs_carry_the_release_pairs_of_earlier_files(
    target: pd.DataFrame, delays: pd.DataFrame
) -> None:
    from nowcastbdbv.backtest import make_tasks
    from nowcasteval.backtest import BacktestConfig
    from nowcastrenewal.forecaster import vintage_inputs
    from nowcastrenewal.model import MAIN

    tasks = {
        t.vintage: t
        for t in make_tasks(
            target.loc[target["kind"] == "real"],
            delays.assign(kind="real"),
            BacktestConfig(seed=3),
        )
    }
    late = vintage_inputs(tasks["2026-10-17"], MAIN)
    assert late.release_pairs is not None
    # One pair of delay files, 10-09 and 10-17.
    assert late.release_pairs["pair"].shape[1] == 1
    assert vintage_inputs(tasks["2026-10-09"], MAIN).release_pairs is None


def test_report_cutoffs_are_days_from_the_model_start(
    task: ForecastTask,
) -> None:
    last = {"2026-10-24": "2026-10-24", "2026-11-01": "2026-10-31"}
    task = replace(task, report_by=last)
    data = rn.vintage_inputs(task, rn.model_config(rn.VARIANTS["renewal"]))
    cuts = rn.report_cutoffs(task, data)
    start = pd.Timestamp(data.start_date)
    # Each later release is cut at its latest onset day.
    assert cuts == {(pd.Timestamp(last[d]) - start).days: d for d in last}


def test_delay_summary_reads_shares_and_medians_by_onset_week() -> None:
    import datetime as dt

    from nowcastrenewal.forecaster import delay_summary

    draws, dmax, days = 200, 20, 21
    # A constant hazard of 0.5 on every delay, shifted later in the last
    # week so its delays are longer.
    beta = np.zeros((draws, dmax))
    shift = np.zeros((draws, days))
    shift[:, 14:] = -2.0
    # The axis starts on a Sunday, so its weeks end on days 6, 13, 20.
    out = delay_summary(beta, shift, dt.date(2026, 10, 4), days - 1)
    weeks = out.loc[out["quantity"] == "median_delay"]
    assert list(weeks["x"]) == ["2026-10-10", "2026-10-17", "2026-10-24"]
    # With hazard h, P(delay <= d) = 1 - (1 - h)^(d + 1). For h = 0.5 the
    # median is day 0; for h = sigmoid(-2) it is day 5.
    assert list(weeks["q50"]) == [0.0, 0.0, 5.0]
    # The mean of a geometric delay with h = 0.5 is (1 - h) / h = 1.
    mean = out.loc[out["quantity"] == "mean_delay", "q50"].to_numpy()
    assert mean[0] == pytest.approx(1.0, abs=1e-3)
    assert mean[2] > mean[0]
    share = out.loc[out["quantity"] == "share_within_7", "q50"].to_numpy()
    assert share[0] == pytest.approx(1 - 0.5**8)
    assert share[2] < share[0]
    cum = out.loc[out["kind"] == "cumulative"]
    assert len(cum) == dmax + 1
    assert cum["q50"].iloc[-1] == pytest.approx(1.0)
    assert cum["q50"].is_monotonic_increasing


def test_the_model_cut_off_is_the_release_s_latest_onset_day(
    target: pd.DataFrame, delays: pd.DataFrame
) -> None:
    import datetime as dt

    config = rn.model_config(rn.VARIANTS["renewal"])
    for label, last in (
        ("2026-10-09", "2026-10-07"),
        ("2026-10-17", "2026-10-16"),
        ("2026-10-24", "2026-10-24"),
        ("2026-11-01", "2026-10-31"),
    ):
        rows = target.loc[target["vintage"] == label]
        task = make_tasks(rows, delays, BacktestConfig(n_samples=20, seed=3))[0]
        data = rn.vintage_inputs(task, config)
        # The cut-off is the latest onset day and the last data week is
        # the week that holds it.
        cut = data.start_date + dt.timedelta(days=data.cutoff)
        end = data.start_date + dt.timedelta(days=7 * data.n_weeks - 1)
        assert cut.isoformat() == last
        assert end == epiweek_end(last)


def test_a_release_is_cut_at_its_stored_cut_off(
    target: pd.DataFrame, delays: pd.DataFrame
) -> None:
    # The 2026-11-08 release holds onsets up to its cut-off, three days
    # before it.
    rows = target.loc[target["vintage"] == "2026-11-08"]
    task = make_tasks(rows, delays, BacktestConfig(n_samples=20, seed=3))[0]
    assert task.cutoff == "2026-11-05"
    data = rn.vintage_inputs(task, rn.model_config(rn.VARIANTS["renewal"]))
    cut = data.start_date + dt.timedelta(days=data.cutoff)
    end = data.start_date + dt.timedelta(days=7 * data.n_weeks - 1)
    assert cut == dt.date(2026, 11, 5)
    # The grid ends with the week holding the cut-off.
    assert end == epiweek_end("2026-11-05")


def test_province_workers_fill_the_cores_with_chains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(rn.PROVINCE_WORKERS_ENV, raising=False)
    monkeypatch.setattr(rn, "usable_cpus", lambda: 4)
    assert rn.province_workers(2, 5) == 2
    assert rn.province_workers(4, 5) == 1
    # No more workers than provinces, and a cap on a large host.
    assert rn.province_workers(2, 1) == 1
    monkeypatch.setattr(rn, "usable_cpus", lambda: 64)
    assert rn.province_workers(2, 20) == rn.MAX_PROVINCE_WORKERS
    monkeypatch.setenv(rn.PROVINCE_WORKERS_ENV, "1")
    assert rn.province_workers(2, 5) == 1
    for bad in ("0", "auto"):
        monkeypatch.setenv(rn.PROVINCE_WORKERS_ENV, bad)
        with pytest.raises(ValueError, match=rn.PROVINCE_WORKERS_ENV):
            rn.province_workers(2, 5)


def test_renewalsp_keeps_its_rt_walk_step_sd_off_zero(
    task: ForecastTask,
) -> None:
    # The centred walk funnels where the step SD nears 0: at 2026-11-22,
    # CD-NK alone mixed rt_sd at bulk ESS 29 and E-BFMI 0.27-0.36 under
    # HalfNormal(0.2), and 72 and 0.49-0.70 under LogNormal(log 0.12, 0.5).
    import numpyro

    from nowcastrenewal.model import model_args
    from nowcastrenewal.model.zones import build_model

    priors = {
        name: rn.model_config(v).rt_sd_lognormal
        for name, v in rn.VARIANTS.items()
    }
    assert priors == {
        "renewal": None,
        "renewalsp": (0.12, 0.5),
        "renewalbvd": None,
    }
    variant = rn.VARIANTS["renewalsp"]
    one = replace(task, locations=task.locations[:1], mask=())
    data = rn.vintage_inputs(one, rn.model_config(variant))
    model = build_model(data, rn.fit_config(variant, data))
    tr = numpyro.handlers.trace(
        numpyro.handlers.seed(model.sample, rng_seed=0)
    ).get_trace(**model_args(model, data))
    fn = tr["rt_sd"]["fn"]
    assert type(fn).__name__ == "LogNormal"
    assert float(fn.loc) == pytest.approx(np.log(0.12))
    assert float(fn.scale) == pytest.approx(0.5)
