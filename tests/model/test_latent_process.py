"""The simulator's latent process: R paths, noise, the start and batches.

No external tool computes these grid constructs, so the paths are checked
against their closed forms on inline values; the renewal and reference
events are checked live against epinowcast in `test_latent_epinowcast.py`.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro import handlers

from nowcastfoundation.simulator import batch as sb
from nowcastfoundation.simulator import draws
from nowcastrenewal.components.pmfs import Delay, generation_time_pmf
from nowcastrenewal.latent.rt import (
    KINDS,
    PlaceRt,
    RtPath,
    WeeklyNoise,
    daily_walk,
)
from nowcastrenewal.latent.simulate import (
    COLUMNS,
    simulate_latent_batch,
)
from nowcastrenewal.latent.start import InitializeStrata

from .r_live import ROOT


def gt_pmf(mean: float, family: str, cv: float) -> np.ndarray:
    """The grid's generation time, from the shared pmfs."""
    return generation_time_pmf(Delay.from_mean_cv(family, mean, cv))


def matrix(x: Any) -> np.ndarray:
    return np.array(x, dtype=float)


def replay(
    model: Any, normal: Any, choice: Any, *args: Any, **kwargs: Any
) -> Any:
    """Run `model` with the given draws at its sample sites, in order."""
    z = np.atleast_1d(np.asarray(normal, dtype=float))
    choices = [choice] if isinstance(choice, str) else list(choice or [])
    with jax.enable_x64(True):
        trace = handlers.trace(handlers.seed(model, 0)).get_trace(
            *args, **kwargs
        )
        data: dict[str, Any] = {}
        i = j = 0
        for name, site in trace.items():
            if site["type"] != "sample":
                continue
            value = jnp.asarray(site["value"])
            if jnp.issubdtype(value.dtype, jnp.integer):
                data[name] = jnp.asarray(KINDS.index(choices[j]))
                j += 1
                continue
            size = int(value.size)
            data[name] = jnp.asarray(z[i : i + size]).reshape(value.shape)
            i += size
        assert i == len(z), "the normal draws were not all used"
        assert j == len(choices), "the path draws were not all used"
        return handlers.substitute(handlers.seed(model, 0), data=data)(
            *args, **kwargs
        )


PATH_VALUES: dict[str, Any] = {
    "amplitude": 0.3,
    "peak": 0.4,
    "flip_days": 10.0,
    "period": 40.0,
    "phase": 1.0,
    "waves": 3,
    "wave_scale": [1.0, 0.8, 0.6, 0.5],
    "walk_r": [0.0, 0.1, -0.05, 0.2],
    "control_days": 20.0,
    "flat": 0.05,
}


def test_paths_take_their_closed_forms() -> None:
    # Each kind on 150 days, u = t / 150, against its formula, to 1e-12.
    n, v = 150, PATH_VALUES
    t = np.arange(n, dtype=float)
    u, a, peak = t / n, v["amplitude"], v["peak"]
    walk = daily_walk(t, v["walk_r"])
    want = {
        "endemic": walk,
        "controlled": a * np.exp(-t / v["control_days"]) + walk,
        "growth": np.full(n, a),
        "decline": np.full(n, -a),
        "flat": np.full(n, v["flat"]),
        "rise_fall_gradual": a * (peak - u) / 0.6,
        "fall_rise": -a * (peak - u) / 0.6,
        "rise_fall_sharp": a
        * (1 - 2 / (1 + np.exp(-(t - peak * n) / (v["flip_days"] / 4.4)))),
        "oscillating": a * np.sin(2 * np.pi * t / v["period"] + v["phase"]),
    }
    for kind, path in want.items():
        np.testing.assert_allclose(
            RtPath.from_values(kind, v).log_r(n),
            path,
            rtol=1e-12,
            atol=1e-15,
            err_msg=kind,
        )
    # Multi-wave: wave k (from 1) of `waves` has amplitude 2 a scale[k].
    wave = np.minimum(4, np.floor(3 * u + 1 / (2 * np.pi)) + 1).astype(int)
    np.testing.assert_allclose(
        RtPath.from_values("multi_wave", v).log_r(n),
        2
        * a
        * np.asarray(v["wave_scale"])[wave - 1]
        * np.sin(2 * np.pi * 3 * u + 1),
        rtol=1e-12,
        atol=1e-15,
    )


def test_the_daily_walk_is_linear_between_weeks() -> None:
    walk = [0.0, 0.7, -0.7]
    days = np.array([0, 7, 10.5, 14, 30])
    np.testing.assert_allclose(
        daily_walk(days, walk), [0.0, 0.7, 0.0, -0.7, -0.7], atol=1e-12
    )


def test_weekly_noise_is_an_ar1_held_for_a_week_from_zero() -> None:
    z = np.array([0.5, -1.0, 2.0, 0.3])
    out = np.asarray(
        replay(
            lambda: WeeklyNoise("w")(n=30, sd=0.2, noise_name="e"),
            z,
            [],
        )
    )
    e = np.zeros(5)
    for w in range(1, 5):
        e[w] = 0.8 * e[w - 1] + 0.2 * z[w - 1]
    np.testing.assert_allclose(out, np.repeat(e, 7)[:30], rtol=1e-15)


def test_weekly_noise_is_pyrenews_weekly_process_over_the_shared_walk() -> None:
    from pyrenew.latent.temporal_processes import WeeklyTemporalProcess

    from nowcastrenewal.components.walks import NonCentredRandomWalk

    noise = WeeklyNoise("w")
    process = noise.process(0.2)
    assert isinstance(process, WeeklyTemporalProcess)
    assert isinstance(process.inner, NonCentredRandomWalk)


def test_a_shared_relation_draws_only_the_strata_noise() -> None:
    path = RtPath("growth", amplitude=0.2)
    rt = PlaceRt("rt", path, "shared", noise_scale=5.0, coupling=1.0)
    with jax.enable_x64(True):
        trace = handlers.trace(handlers.seed(rt, 0)).get_trace(
            n=21, place=np.array([0, 0, 1])
        )
    sites = [k for k, s in trace.items() if s["type"] == "sample"]
    assert sites == [f"rt_stratum_{j}_z" for j in range(3)]


def test_nested_places_share_their_regions_deviation() -> None:
    # All the places' noise from their region (rho 1): places in one
    # region have one path, regions differ.
    path = RtPath("flat", amplitude=0.2)
    rt = PlaceRt(
        "rt",
        path,
        "nested",
        noise_scale=5.0,
        place_sd=0.2,
        place_rho=1.0,
        place_region=[1, 2, 1, 2],
        coupling=1.0,
    )
    with jax.enable_x64(True):
        trace = handlers.trace(handlers.seed(rt, 0)).get_trace(
            n=28, place=np.array([0, 1, 2, 3])
        )
    sites = [k for k, s in trace.items() if s["type"] == "sample"]
    assert sites[:2] == ["rt_region_1_z", "rt_region_2_z"]
    assert sites[2:6] == [f"rt_place_{i}_z" for i in range(4)]
    log_r = np.asarray(trace["rt::log_r"]["value"])
    np.testing.assert_allclose(log_r[:, 0], log_r[:, 2])
    np.testing.assert_allclose(log_r[:, 1], log_r[:, 3])
    assert not np.allclose(log_r[:, 0], log_r[:, 1])


def test_nested_places_need_a_region_each() -> None:
    rt = PlaceRt("rt", RtPath("flat"), "nested", place_region=[1])
    with pytest.raises(ValueError, match="region for each of 2"):
        handlers.seed(rt, 0)(n=7, place=np.array([0, 1]))


@pytest.mark.parametrize("early", [True, False])
def test_the_start_is_a_seed_grown_into_the_last_day_or_a_level(
    early: bool,
) -> None:
    # A seed of at least 1 on the last day, halving each day before at a
    # growth of 2, over the 3 active days; the padding day is 0.
    with jax.enable_x64(True):
        start = InitializeStrata(4, early, active=3, growth=2.0)
        out = np.asarray(start(np.array([0.4, 3.0, 0.0])))
    if early:
        want = np.outer([0.0, 0.25, 0.5, 1.0], [1.0, 3.0, 0.0])
    else:
        want = np.outer([0.0, 1.0, 1.0, 1.0], [0.4, 3.0, 0.0])
    np.testing.assert_array_equal(out, want)


def _simulations() -> list[dict[str, Any]]:
    """Two runs of one established multi-stratum grid draw, and a short one."""
    grid = draws.read_grid(ROOT / "config" / "finetune" / "grid.yaml")
    for i in range(500):
        design = draws.draw_design(grid, draws.generator(77, i, 1), weeks=8)
        if (
            design["window"] == "established"
            and 2 <= int(design["n_places"]) * int(design["n_ages"]) <= 6
        ):
            break
    values = draws.draw_values(grid, design, draws.generator(77, i, 2))
    sims = [
        {
            "id": f"s{k}",
            "key": k,
            "n_days": sb._n_days(values),
            "design": design,
            "values": values,
            "strata": sb._strata(design),
        }
        for k in range(2)
    ]
    return [*sims, {**sims[0], "id": "short", "n_days": 3}]


def _one_day_spikes(x: np.ndarray) -> np.ndarray:
    """Days more than 3 times both neighbours and more than 1 above them."""
    beside = np.maximum(x[:-2], x[2:])
    return np.flatnonzero((x[1:-1] > 3 * beside) & (x[1:-1] - beside > 1)) + 1


@pytest.mark.parametrize(
    ("window", "imports"), [("early", "none"), ("established", "pulses")]
)
def test_seeds_and_import_pulses_make_no_one_day_spike(
    window: str, imports: str
) -> None:
    # A seed put on one day, or a pulse of imports on one day, entered the
    # renewal as a spike that fell straight back (the next day is the
    # seed or pulse times R times the generation time's first lag).
    grid = draws.read_grid(ROOT / "config" / "finetune" / "grid.yaml")
    sims = []
    for i in range(2000):
        design = draws.draw_design(grid, draws.generator(81, i, 1), weeks=12)
        if (design["window"], design["imports"]) != (window, imports):
            continue
        if design["strata_link"] != "linked":
            continue
        values = draws.draw_values(grid, design, draws.generator(81, i, 2))
        if imports == "pulses" and not np.size(values["import_at"]):
            continue
        sims.append(
            {
                "id": f"s{i}",
                "key": i,
                "n_days": sb._n_days(values),
                "design": design,
                "values": values,
                "strata": sb._strata(design),
            }
        )
        if len(sims) == 6:
            break
    long, values = simulate_latent_batch(sims, KINDS)
    assert {v["error"] for v in values.values()} == {None}
    spikes = {
        (k, int(g)): _one_day_spikes(run["infections"].to_numpy())
        for (k, g), run in long[long["group"] > 0].groupby(["id", "group"])
    }
    assert not any(len(d) for d in spikes.values()), spikes


def test_a_batch_runs_each_simulation_and_records_its_errors() -> None:
    sims = _simulations()
    long, values = simulate_latent_batch(sims, KINDS)
    assert list(long.columns) == list(COLUMNS)
    assert set(long["id"]) == {"s0", "s1"}
    k = len(sims[0]["strata"])
    s0 = long[long["id"] == "s0"]
    assert set(s0["group"]) == set(range(k + 1))
    assert s0.groupby("group").size().eq(sims[0]["n_days"]).all()
    # Group 0 is the sum of the strata.
    whole = s0[s0["group"] == 0].set_index("day")["onsets"]
    parts = s0[s0["group"] > 0].groupby("day")["onsets"].sum()
    np.testing.assert_allclose(whole, parts, rtol=1e-12)
    assert (
        values["short"]["error"]
        == "the generation time is longer than the series"
    )
    assert values["s0"]["error"] is None
    assert len(values["s0"]["stratum_population"]) == k
    gt = gt_pmf(
        sims[0]["values"]["gt_mean"],
        sims[0]["design"]["gt_family"],
        sims[0]["values"]["gt_cv"],
    )
    np.testing.assert_allclose(values["s0"]["gt"], gt)
    assert values["s0"]["n_gt"] == len(gt)


def test_a_key_gives_the_same_run_and_another_key_another() -> None:
    sims = _simulations()[:2]
    sims[1]["key"] = 0
    long, _ = simulate_latent_batch(sims, KINDS)
    a = long[long["id"] == "s0"]["onsets"].to_numpy()
    b = long[long["id"] == "s1"]["onsets"].to_numpy()
    np.testing.assert_array_equal(a, b)
    sims[1]["key"] = 5
    long, _ = simulate_latent_batch(sims, KINDS)
    b = long[long["id"] == "s1"]["onsets"].to_numpy()
    assert not np.allclose(a, b)


def test_any_error_in_one_simulation_is_recorded_and_the_batch_goes_on() -> (
    None
):
    # The batch redraws a simulation on any error.
    sims = _simulations()[:2]
    broken = sims[1]
    broken["values"] = {
        k: v for k, v in broken["values"].items() if k != "r_drift"
    }
    long, values = simulate_latent_batch(sims, KINDS)
    assert values["s1"]["error"].startswith("KeyError")
    assert values["s0"]["error"] is None
    assert set(long["id"]) == {"s0"}


def test_a_single_draw_of_a_vector_value_may_be_a_scalar() -> None:
    sim = _simulations()[0]
    sim["values"] = {
        k: (v[0] if isinstance(v, list) and len(v) == 1 else v)
        for k, v in sim["values"].items()
    }
    long, out = simulate_latent_batch([sim], KINDS)
    assert out["s0"]["error"] is None
    assert np.isfinite(long["onsets"]).all()


def test_exponential_depletion_runs_and_stays_within_the_population() -> None:
    sims = _simulations()[:1]
    long, out = simulate_latent_batch(sims, KINDS, depletion="exponential")
    assert out["s0"]["error"] is None
    s = long[(long["id"] == "s0") & (long["group"] > 0)]
    total = s.groupby("group")["infections"].sum().to_numpy()
    assert (total <= np.array(out["s0"]["stratum_population"]) * 2).all()


def test_an_unknown_depletion_is_refused() -> None:
    _, out = simulate_latent_batch(
        _simulations()[:1], KINDS, depletion="quadratic"
    )
    assert "depletion" in out["s0"]["error"]


def test_the_fitted_models_do_not_import_the_latent_process() -> None:
    code = (
        "import sys, nowcastrenewal.forecaster, nowcastrenewal.zones;"
        "print(sorted(m for m in sys.modules"
        " if m.startswith('nowcastrenewal.latent')))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "[]"


def test_ci_runs_the_live_latent_checks() -> None:
    import yaml

    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/heavy.yml").read_text()
    )
    job = workflow["jobs"]["r-live"]
    runs = " ".join(s.get("run", "") for s in job["steps"])
    packages = " ".join(
        s.get("with", {}).get("packages", "") for s in job["steps"]
    )
    assert "packages/nowcastrenewal/tests/test_latent_epinowcast.py" in runs
    for pkg in ("epinowcast", "cmdstanr"):
        assert pkg in packages, pkg
    # The real tools only: no package of our own R code.
    assert "pkgload" not in packages
    assert "engine" not in workflow["jobs"]
