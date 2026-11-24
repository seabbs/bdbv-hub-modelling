from __future__ import annotations

import datetime as dt
import urllib.error
from collections.abc import Collection
from dataclasses import replace
from email.message import Message
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from bdbvhub.models import get_model
from nowcastbdbv.backtest import make_tasks
from nowcastbdbv.data.zones import load_zone_data
from nowcasteval.backtest import BacktestConfig
from nowcastrenewal import forecaster as rn
from nowcastrenewal import zones as zr
from nowcastutils.base import ForecastTask, ModelUnavailableError
from nowcastutils.hierarchy import is_zone, province_of

pytest.importorskip("pyrenew")

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "tests" / "fixtures" / "data"


@pytest.fixture
def task(target: pd.DataFrame, delays: pd.DataFrame) -> ForecastTask:
    rows = target.loc[target["vintage"] == "2026-10-17"]
    config = BacktestConfig(n_samples=20, seed=3)
    return make_tasks(rows, delays, config, zones=load_zone_data(DATA))[0]


@pytest.fixture
def quiet_zones(
    task: ForecastTask, monkeypatch: pytest.MonkeyPatch
) -> set[str]:
    """Make every zone silent but the four with most cases."""
    zones = zr.task_zones(task)
    counts = zr.zone_target(task).groupby("location")["total"].sum()
    busy = set(counts.sort_values().index[-4:])
    silent = {z for z in zones if z not in busy}
    monkeypatch.setattr(zr, "silent_zones", lambda _: silent)
    return silent


def _fake_run(
    task: ForecastTask, variant: rn.Variant, quiet: Collection[str] = ()
) -> rn.RenewalRun:
    """A run as `ZoneRenewalModel.run` returns it.

    Its trajectories cover every zone of the task. The zones in `quiet`
    are silent: they are not in its fitted inputs, and so have no Rt.
    """
    places = zr.zone_vintage_inputs(task, rn.model_config(variant))
    fitted = tuple(z for z in task.locations if z not in set(quiet))
    data = zr.zone_vintage_inputs(
        replace(task, locations=fitted), rn.model_config(variant)
    )
    rng = np.random.default_rng(4)
    weeks = places.n_weeks + places.horizon_weeks
    zone = rng.poisson(5.0, size=(task.n_samples, weeks, len(places.names)))
    draws: dict[str, Any] = {
        "province": zone.astype(float),
        "insample": {"known": zone * 0.7, "imputed": zone * 0.3},
        "as_reported": {
            cut: {"province": np.ones_like(zone, dtype=float)}
            for cut in rn.report_cutoffs(task, places)
        },
        "rt": {
            "rt": rng.lognormal(
                0.0, 0.3, size=(40, 6 + weeks, len(data.names))
            ),
            "weight": rng.gamma(
                2.0, 5.0, size=(40, 6 + weeks, len(data.names))
            ),
            "start": data.start_date - dt.timedelta(days=42),
        },
    }
    n = 50
    extra = {
        "diverging": np.zeros((2, n), dtype=bool),
        "num_steps": np.full((2, n), 15),
        "accept_prob": np.full((2, n), 0.9),
        "energy": rng.normal(size=(2, n)),
        "adapt_state.step_size": np.full((2, n), 0.1),
    }
    delay = pd.concat(
        rn.delay_summary(
            rng.normal(-1, 0.1, size=(n, 20)),
            rng.normal(0, 0.1, size=(n, 14)),
            data.start_date,
            13,
        ).assign(place=place)
        for place in data.parent_names
    )
    return rn.RenewalRun(
        variant=variant,
        data=data,
        draws=draws,
        trace={"sd": rng.normal(size=(2, n))},
        extra=extra,
        seconds=1.5,
        cpu_seconds=2.9,
        delay=delay,
        forecast_data=places if quiet else None,
    )


def test_renewal_and_renewalbvd_fit_zones() -> None:
    model = get_model("renewalbvd")
    assert isinstance(model, zr.ZoneRenewalModel)
    assert model.variant.base == "BVD"
    assert isinstance(get_model("renewal"), zr.ZoneRenewalModel)
    # renewalsp fits each zone on its own.
    assert type(get_model("renewalsp")) is zr.SingleZoneRenewalModel
    # The zone fit's representation differs, so it keys a stored fit.
    assert repr(model) != repr(rn.RenewalModel(rn.VARIANTS["renewalbvd"]))


def test_a_zone_fit_forecasts_the_zones_its_task_holds(
    task: ForecastTask,
) -> None:
    # The task's zone codes are what the forecast is checked against
    # (`nowcastutils.base._check_block`), so the fit takes its zones
    # from them, not from the data directory.
    mine = [z for z in task.zone_codes if province_of(z) in task.locations]
    fewer = replace(task, zone_codes=tuple(mine[::2]))
    assert zr.task_zones(fewer) == tuple(sorted(mine[::2]))


def test_a_real_task_with_no_zone_codes_has_no_zone_fit(
    task: ForecastTask,
) -> None:
    bare = replace(task, zone_codes=())
    with pytest.raises(ModelUnavailableError, match="zone codes"):
        zr.task_zones(bare)
    with pytest.raises(ModelUnavailableError, match="zone codes"):
        get_model("renewal").forecast(bare)


def test_zone_fit_sums_fitted_and_silent_zones_to_provinces(
    task: ForecastTask,
    monkeypatch: pytest.MonkeyPatch,
    quiet_zones: set[str],
    tmp_path: Path,
) -> None:
    seen: list[tuple[str, ...]] = []

    def fake(self: zr.ZoneRenewalModel, t: ForecastTask) -> rn.RenewalRun:
        seen.append(t.locations)
        return _fake_run(t, self.variant)

    monkeypatch.setattr(zr.ZoneRenewalModel, "run", fake)

    def no_walk(_: ForecastTask) -> None:
        raise AssertionError("renewal used the random-walk layer")

    monkeypatch.setattr(zr, "_silent_walk", no_walk)
    model = get_model("renewal")
    assert isinstance(model, zr.ZoneRenewalModel)
    samples = model.forecast(task)
    # #768 item 8: no random-walk layer. The run takes every zone of the
    # task: it fits the zones with a recent case and draws the silent
    # ones from the fitted importation, in one forecast pass.
    (given,) = seen
    assert set(given) == set(zr.task_zones(task))
    zones = model.zone_samples
    assert zones is not None
    assert set(zones["location"]) == set(given)
    assert quiet_zones < set(zones["location"])
    # The forecast carries the zone fit's own zones, so the store and the
    # scores read them rather than a split of the provinces.
    own = samples["location"].map(is_zone)
    assert set(samples.loc[~own, "location"]) == set(task.locations)
    pd.testing.assert_frame_equal(
        samples.loc[own].reset_index(drop=True),
        zones[list(samples.columns)].reset_index(drop=True),
    )
    samples = samples.loc[~own]
    # Provinces are the sums of their zones, sample by sample, and
    # national is their sum.
    assert model.last_fit is not None
    keys = ["location", "target_end_date", "sample"]
    summed = (
        zones.assign(location=zones["location"].map(province_of))
        .groupby(keys)["value"]
        .sum()
    )
    pd.testing.assert_series_equal(
        samples.set_index(keys)["value"].sort_index(),
        summed.sort_index(),
        check_names=False,
    )
    # The delay is by province.
    assert model.delay_summary is not None
    assert set(model.delay_summary["location"]) <= {
        *task.locations,
        "shared",
    }
    # The fit directory carries the zone samples for the submission.
    model.write_fit_dir(task, tmp_path, "main")
    written = pd.read_csv(tmp_path / zr.ZONES_FILE)
    assert set(written["location"]) == set(zones["location"])


def test_zone_fit_refuses_national_cases_in_no_province(
    task: ForecastTask,
) -> None:
    target = task.target.copy()
    nat = target["location"] == "CD"
    week = max(target.loc[nat, "target_end_date"])
    target.loc[nat & (target["target_end_date"] == week), "total"] += 2
    model = get_model("renewal")
    assert isinstance(model, zr.ZoneRenewalModel)
    with pytest.raises(ValueError, match="no province"):
        model.forecast_fit(replace(task, target=target))


def test_fit_renewal_fits_zones_and_writes_their_samples(
    task: ForecastTask,
    monkeypatch: pytest.MonkeyPatch,
    quiet_zones: set[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        zr.ZoneRenewalModel,
        "run",
        lambda self, t: _fake_run(t, self.variant, quiet_zones),
    )
    record = rn.fit_renewal(
        task, {"num_chains": 2}, tmp_path, variant="renewal", config_name="main"
    )
    assert record["model"] == "renewal"
    zones = pd.read_csv(tmp_path / zr.ZONES_FILE)
    assert quiet_zones < set(zones["location"])
    samples = pd.read_csv(tmp_path / rn.SAMPLES_FILE)
    keys = ["location", "target_end_date", "sample"]
    final = zones.loc[zones["as_of"].isna()] if "as_of" in zones else zones
    summed = (
        final.assign(location=final["location"].map(province_of))
        .groupby(keys)["value"]
        .sum()
    )
    rows = samples.loc[samples["location"].isin(task.locations)]
    if "as_of" in rows:
        rows = rows.loc[rows["as_of"].isna()]
    pd.testing.assert_series_equal(
        rows.set_index(keys)["value"].sort_index(),
        summed.sort_index(),
        check_names=False,
    )
    # Rt for each fitted zone, its province and the nation; none for the
    # zones outside the fit.
    rt = pd.read_csv(tmp_path / rn.RT_FILE)
    fitted = set(zones["location"]) - quiet_zones
    assert set(rt["location"]) == {
        *fitted,
        *map(province_of, fitted),
        "CD",
    }


def test_zone_inputs_take_the_zone_counts_and_province_delays(
    task: ForecastTask,
) -> None:
    rows = zr.zone_target(task)
    assert {"observed", "imputed", "total", "cutoff"} <= set(rows.columns)
    zones = tuple(rows.groupby("location")["total"].sum().nlargest(6).index)
    zt = replace(task, target=rows, locations=zones)
    data = zr.zone_vintage_inputs(zt, rn.model_config(rn.VARIANTS["renewal"]))
    assert data.names == list(zones)
    assert [data.parent_names[i] for i in data.parents] == [
        province_of(z) for z in zones
    ]
    assert data.delay_counts.shape[1] == len(data.parent_names)
    # Zone weekly counts sum to the province counts of the release.
    prov = task.target.loc[task.target["location"].isin(data.parent_names)]
    wide = prov.pivot_table(
        index="target_end_date", columns="location", values="total"
    )
    zone_totals = rows.assign(p=rows["location"].map(province_of)).pivot_table(
        index="target_end_date", columns="p", values="total", aggfunc="sum"
    )[data.parent_names]
    pd.testing.assert_frame_equal(
        wide.reindex_like(zone_totals).fillna(0),
        zone_totals.fillna(0),
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.parametrize(
    ("rel", "moved"),
    [
        # The zone forecaster keys every renewal model, renewalsp's
        # per-zone fits included.
        ("zones.py", ("renewal", "renewalbvd", "renewalsp")),
        # The model builder is every renewal model's, renewalsp one place
        # a patch.
        ("model/zones.py", ("renewal", "renewalbvd", "renewalsp")),
    ],
)
def test_zone_code_keys_the_models_that_run_it(
    tmp_path: Path, rel: str, moved: tuple[str, ...]
) -> None:
    import shutil

    from bdbvhub.store import store

    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy(ROOT / name, tmp_path / name)
    for folder in ("packages",):
        shutil.copytree(
            ROOT / folder,
            tmp_path / folder,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    models = ("renewal", "renewalbvd", "renewalsp", "randomwalknowcast")
    before = {m: store.code_digest(m, tmp_path) for m in models}
    path = tmp_path / "packages/nowcastrenewal/src/nowcastrenewal" / rel
    path.write_text(path.read_text() + "\n_EDITED = 1\n")
    after = {m: store.code_digest(m, tmp_path) for m in models}
    assert {m for m in models if after[m] != before[m]} == set(moved)


def test_bvd_zone_rt_becomes_one_term_per_fitted_zone(tmp_path: Path) -> None:
    from nowcastrenewal import bvd
    from nowcastrenewal.model.zones import zone_bvd_inputs

    fixtures = (
        Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "bvd"
    )
    release = bvd.load_release(fixtures / "results-3086")
    folder = tmp_path / release.tag
    folder.mkdir()
    pd.DataFrame(
        {
            "zone": ["mambasa", "gety"],
            "label": ["Mambasa", "Gety"],
            "province": ["ituri", "ituri"],
            "R_T_median": [1.2, 0.9],
            "R_T_lower": [0.8, 0.5],
            "R_T_upper": [1.8, 1.6],
        }
    ).to_csv(folder / zr.ZONE_ESTIMATES, index=False)
    names = pd.DataFrame(
        {
            "location": ["CD-IT-25", "CD-IT-10"],
            "location_name": ["Mambasa", "Bunia"],
        }
    )
    rows = zr.zone_bvd_rows(release, names, tmp_path)
    assert rows is not None
    # Gety has no hub case, so it has no term.
    assert rows["group"].tolist() == ["zone:CD-IT-25"]
    assert rows["members"].tolist() == ["CD-IT-25"]
    assert rows["log_mean"].iloc[0] == pytest.approx(np.log(1.2))
    both = pd.concat([bvd.model_rows(release), rows], ignore_index=True)
    zones = ["CD-IT-10", "CD-IT-25", "CD-NK-03"]
    out = zone_bvd_inputs(
        both,
        zones,
        {z: z[:5] for z in zones},
        release.as_of - pd.Timedelta(days=60),
    )
    assert out is not None
    groups = out["groups"].tolist()
    # National holds every zone; the zone term holds Mambasa alone.
    assert groups[0] == [True, True, True]
    assert groups[-1] == [False, True, False]
    assert out["kinds"][-1] == 1

    # A release with no zone file (404): no zone terms.
    def missing(url: str) -> bytes:
        raise urllib.error.HTTPError(url, 404, "Not Found", Message(), None)

    assert zr.zone_bvd_rows(release, names, tmp_path / "none", missing) is None

    # Any other failure is an error, not a release without zones.
    def offline(_: str) -> bytes:
        raise OSError("offline")

    with pytest.raises(OSError, match="offline"):
        zr.zone_bvd_rows(release, names, tmp_path / "none", offline)


@pytest.mark.usefixtures("quiet_zones")
def test_bvd_zone_fit_records_the_release_and_its_rhat(
    task: ForecastTask, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    import shutil

    from nowcastrenewal import bvd

    data = tmp_path / "data"
    shutil.copytree(ROOT / "tests" / "fixtures" / "data", data)
    fixtures = ROOT / "tests" / "fixtures" / "bvd"
    releases = bvd.parse_releases(
        json.loads((fixtures / "releases.json").read_text())
    )
    bvd.save_releases(
        releases.loc[releases["tag"].isin(["results-3151", "results-3086"])],
        data / "bvd",
    )
    for tag in ("results-3151", "results-3086"):
        shutil.copytree(fixtures / tag, data / "bvd" / tag)
    monkeypatch.setenv("BDBV_DATA_DIR", str(data))
    release = rn.bvd_release(task)
    assert release is not None
    assert release.tag == "results-3086"

    def fake(self: zr.ZoneRenewalModel, t: ForecastTask) -> rn.RenewalRun:
        run = _fake_run(t, self.variant)
        return replace(run, bvd=rn.bvd_record(rn.bvd_release(t)))

    monkeypatch.setattr(zr.ZoneRenewalModel, "run", fake)
    record = rn.fit_renewal(
        task,
        {"num_chains": 2},
        tmp_path / "bvd",
        variant="renewalbvd",
        config_name="bvd",
    )
    assert record["bvd"] == {
        "tag": "results-3086",
        "as_of": "2026-10-15",
        "joint_max_rhat": 1.044,
        "max_rhat": bvd.MAX_RHAT,
        "skipped": [],
    }
    # No release qualifies: the record says so.
    assert rn.bvd_record(None) == {"tag": None}
    plain = rn.fit_renewal(
        task,
        {"num_chains": 2},
        tmp_path / "main",
        variant="renewal",
        config_name="main",
    )
    assert "bvd" not in plain


def test_silent_zones_follow_the_fit_through_its_importation(
    task: ForecastTask,
) -> None:
    import jax
    import numpyro
    from numpyro.infer import Predictive

    from nowcastrenewal.model import model_args
    from nowcastrenewal.model.zones import build_zone_model, extend_vintage

    config = rn.model_config(rn.VARIANTS["renewalbvd"])
    zt = replace(task, target=zr.zone_target(task))
    # The four zones with most cases are fitted, the rest are silent.
    counts = zt.target.groupby("location")["total"].sum()
    busy = set(counts.sort_values().index[-4:])
    silent = {z for z in zr.task_zones(task) if z not in busy}
    fitted = tuple(z for z in zr.task_zones(task) if z not in silent)
    quiet = tuple(z for z in zr.task_zones(task) if z in silent)
    # A quiet zone with old cases, and a province with no fitted zone.
    old = set(zt.target["location"]) & set(quiet)
    assert old
    assert {province_of(z) for z in quiet} - {province_of(z) for z in fitted}
    data = zr.zone_vintage_inputs(
        replace(zt, locations=fitted), config, receivers=quiet
    )
    places = zr._zone_places(replace(zt, locations=(*fitted, *quiet)))
    ext = extend_vintage(data, quiet, places)
    p = len(fitted)
    assert ext.names == [*data.names, *quiet]
    assert ext.start_date == data.start_date
    assert ext.known.shape == (data.n_weeks, len(fitted) + len(quiet))
    # The fit's kernel is the fitted block; silent zones only receive.
    np.testing.assert_allclose(ext.kernel[:p, :p], data.kernel)
    np.testing.assert_allclose(ext.kernel[:, p:], 0.0)
    assert np.all(ext.kernel[p:, :p].sum(axis=0) > 0)
    assert np.all(np.isinf(ext.seed_days[p:]))
    # The same sampled sites, so the fit's posterior drives the extension.
    fit_model = build_zone_model(data, config)
    ext_model = build_zone_model(ext, config)

    def sites(model, d):
        tr = numpyro.handlers.trace(
            numpyro.handlers.seed(model.sample, rng_seed=0)
        ).get_trace(**model_args(model, d))
        return {
            k: np.shape(v["value"])
            for k, v in tr.items()
            if v["type"] == "sample" and not v["is_observed"]
        }

    fit_sites = sites(fit_model, data)
    assert sites(ext_model, ext) == fit_sites
    prior = Predictive(
        fit_model.sample, num_samples=3, return_sites=list(fit_sites)
    )(jax.random.key(1), **model_args(fit_model, data))

    def counts(model, d):
        out = Predictive(
            model.sample,
            posterior_samples=prior,
            return_sites=["latent_infections_by_subpop"],
        )(jax.random.key(2), **model_args(model, d))
        x = np.asarray(out["latent_infections_by_subpop"])
        return x * np.asarray(d.populations)

    # On the fit axis the fitted zones are unchanged, and the silent
    # zones take infections only by importation.
    full = counts(ext_model, ext)
    np.testing.assert_allclose(
        full[..., :p], counts(fit_model, data), rtol=1e-5
    )
    assert np.all(full[..., p:] >= 0)
    assert full[..., p:].sum() > 0


def test_silent_zones_have_no_onset_in_the_walk_window(
    task: ForecastTask,
) -> None:
    from nowcastbaselines.chain import Settings
    from nowcastutils.dates import epiweek_end

    quiet = zr.silent_zones(task)
    rows = zr.zone_target(task)
    cut = epiweek_end(task.cutoff)
    start = cut - dt.timedelta(weeks=Settings().walk_weeks - 1)
    recent = rows.loc[
        (rows["target_end_date"] >= str(start))
        & (rows["target_end_date"] <= str(cut))
    ]
    busy = set(recent["location"])
    assert quiet == set(zr.task_zones(task)) - busy
    # The zone task gives the same zones.
    zones = replace(task, locations=zr.task_zones(task))
    assert zr.silent_zones(zones) == quiet


def test_renewalsp_fits_each_zone_on_its_own(
    task: ForecastTask, monkeypatch: pytest.MonkeyPatch
) -> None:
    # #768 item 9, staged: the single-patch model fitted to each zone with
    # a recent case on its own, beside renewalsp. Silent zones keep the
    # walk's introductions, and provinces are zone sums, so no province
    # sample is split across zones.
    model = get_model("renewalsp")
    assert isinstance(model, zr.SingleZoneRenewalModel)
    assert model.variant.per_province
    assert rn.model_config(model.variant).national
    assert rn.FIT_VARIANTS["renewalsp"] == "renewalsp"
    zones = zr.task_zones(task)
    counts = zr.zone_target(task).groupby("location")["total"].sum()
    busy = tuple(sorted(counts.sort_values().index[-3:]))
    silent = [z for z in zones if z not in busy]
    rows = pd.DataFrame(
        [
            (z, date, k, 0.0)
            for z in silent
            for date in task.target_end_dates
            for k in range(1, task.n_samples + 1)
        ],
        columns=["location", "target_end_date", "sample", "value"],
    )
    monkeypatch.setattr(zr, "_silent_walk", lambda _: (rows, set(silent)))
    seen: list[tuple[str, ...]] = []

    def fake(self: rn.RenewalModel, t: ForecastTask) -> rn.RenewalRun:
        seen.append(t.locations)
        return _fake_run(t, self.variant)

    monkeypatch.setattr(rn.RenewalModel, "run", fake)
    samples = model.forecast(task)
    # One fit per busy zone, each alone, on the zone counts.
    assert sorted(seen) == [(z,) for z in busy]
    out = model.zone_samples
    assert out is not None
    assert set(out["location"]) == set(zones)
    # Provinces are the sums of their zones, sample by sample.
    keys = ["location", "target_end_date", "sample"]
    own = samples["location"].map(is_zone)
    final = out.loc[out["as_of"].isna()] if "as_of" in out else out
    summed = (
        final.assign(location=final["location"].map(province_of))
        .groupby(keys)["value"]
        .sum()
    )
    rows_p = samples.loc[~own & samples["location"].isin(task.locations)]
    if "as_of" in rows_p:
        rows_p = rows_p.loc[rows_p["as_of"].isna()]
    pd.testing.assert_series_equal(
        rows_p.set_index(keys)["value"].sort_index(),
        summed.sort_index(),
        check_names=False,
    )


@pytest.fixture
def sp_quiet(task: ForecastTask, monkeypatch: pytest.MonkeyPatch) -> set[str]:
    """renewalsp's silent zones: all but the three with most cases."""
    zones = zr.task_zones(task)
    counts = zr.zone_target(task).groupby("location")["total"].sum()
    busy = set(counts.sort_values().index[-3:])
    silent = {z for z in zones if z not in busy}
    rows = pd.DataFrame(
        [
            (z, date, k, 0.0)
            for z in sorted(silent)
            for date in task.target_end_dates
            for k in range(1, task.n_samples + 1)
        ],
        columns=["location", "target_end_date", "sample", "value"],
    )
    monkeypatch.setattr(zr, "_silent_walk", lambda _: (rows, silent))
    return silent


def test_fit_renewal_writes_renewalsps_ci_contract(
    task: ForecastTask,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sp_quiet: set[str],
) -> None:
    calls: list[dict[str, Any]] = []

    def fake(self: rn.RenewalModel, t: ForecastTask) -> rn.RenewalRun:
        calls.append(dict(self.sampler))
        run = _fake_run(t, self.variant)
        draws = {**run.draws, "draw": np.arange(t.n_samples)[::-1]}
        return replace(run, draws=draws)

    monkeypatch.setattr(rn.RenewalModel, "run", fake)
    sampler = {"num_chains": 2, "num_samples": 50, "seed": 7}
    record = rn.fit_renewal(
        task, sampler, tmp_path, variant="renewalsp", config_name="renewalsp"
    )
    fitted = sorted(set(zr.task_zones(task)) - sp_quiet)
    assert calls[0]["num_chains"] == 2
    assert record["config"] == "renewalsp"
    # Two chains for each zone fitted on its own, labelled by zone.
    assert record["num_chains"] == 2 * len(fitted)
    assert {c["location"] for c in record["chains"]} == set(fitted)
    # The posterior draw behind each trajectory, by zone fitted.
    pairs = pd.read_csv(tmp_path / rn.TRAJECTORY_DRAW_FILE)
    assert set(pairs["location"]) == set(fitted)
    zones = pd.read_csv(tmp_path / zr.ZONES_FILE)
    assert set(zones["location"]) == set(zr.task_zones(task))
    result = rn.forecast_renewal(task, tmp_path)
    assert set(result.samples["target_end_date"]) == set(task.target_end_dates)


def test_renewalsp_gives_counts_as_reported_later_and_a_summary_per_zone(
    task: ForecastTask,
    monkeypatch: pytest.MonkeyPatch,
    sp_quiet: set[str],
) -> None:
    monkeypatch.setattr(
        rn.RenewalModel, "run", lambda self, t: _fake_run(t, self.variant)
    )
    model = get_model("renewalsp")
    task = replace(task, report_by={"2026-10-24": "2026-10-24"})
    samples = model.forecast(task)
    later = samples.loc[samples["as_of"] == "2026-10-24"]
    assert not later.empty
    assert later["target_end_date"].max() <= "2026-10-24"
    final = samples.loc[samples["as_of"].isna()]
    assert set(final["target_end_date"]) == set(task.target_end_dates)
    # One summary row per zone fitted, largest first.
    fitted = set(zr.task_zones(task)) - sp_quiet
    assert {r["fit"] for r in model.fit_diagnostics} == fitted


@pytest.mark.slow
@pytest.mark.usefixtures("sp_quiet")
def test_renewalsp_zones_in_parallel_match_one_after_another(
    task: ForecastTask, monkeypatch: pytest.MonkeyPatch
) -> None:
    tiny = {
        "num_chains": 1,
        "num_warmup": 4,
        "num_samples": 4,
        "target_accept": 0.8,
        "max_tree_depth": 2,
        "seed": 1,
    }
    model = zr.SingleZoneRenewalModel(rn.VARIANTS["renewalsp"], sampler=tiny)
    serial = model.forecast_fit(task)
    monkeypatch.setenv(rn.PROVINCE_WORKERS_ENV, "2")
    parallel = model.forecast_fit(task)
    pd.testing.assert_frame_equal(parallel.samples, serial.samples)
    assert list(parallel.runs) == list(serial.runs)
