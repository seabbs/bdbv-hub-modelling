"""The simulator's design and value draws (`nowcastfoundation.simulator.draws`).

Checked against the grid's own shares and sampling distributions, and
against the R engine's `draw_design`/`draw_values`, called through Rscript.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from nowcastfoundation.simulator import draws

ROOT = Path(__file__).parents[2]
GRID_FILE = ROOT / "config" / "finetune" / "grid.yaml"
ENGINE = ROOT / "packages" / "nowcastfoundation" / "r" / "R"
OPTIONS = tuple(draws.read_grid(GRID_FILE)["max_delay_days"])


@pytest.fixture(scope="module")
def grid():
    return draws.read_grid(GRID_FILE)


def _scipy(d):
    match d["dist"]:
        case "lognormal":
            return stats.lognorm(d["sdlog"], scale=d["median"])
        case "normal":
            return stats.norm(d["mean"], d["sd"])
        case "halfnormal":
            return stats.halfnorm(scale=d["scale"])
        case "gamma":
            shape = (d["mean"] / d["sd"]) ** 2
            return stats.gamma(shape, scale=d["sd"] ** 2 / d["mean"])
        case "beta":
            return stats.beta(
                d["a"], d["b"], loc=d["lower"], scale=d["upper"] - d["lower"]
            )
        case "uniform":
            return stats.uniform(d["lower"], d["upper"] - d["lower"])
    raise AssertionError(d)


def test_factor_levels_follow_their_weights(grid):
    rng = draws.generator(1)
    n = 20_000
    cells = [draws.draw_cell(grid, rng) for _ in range(n)]
    for name, levels in grid["factors"].items():
        w = np.asarray(list(levels.values()), dtype=float)
        p = w / w.sum()
        seen = Counter(c[name] for c in cells)
        assert set(seen) <= set(levels)
        for level, share in zip(levels, p, strict=True):
            se = np.sqrt(share * (1 - share) / n)
            assert abs(seen[level] / n - share) < 5 * se, (name, level)


def test_every_setting_follows_its_sampling_distribution(grid):
    rng = draws.generator(2)
    for name, d in grid["distributions"].items():
        x = draws.draw_setting(grid, name, rng, 5_000)
        assert stats.kstest(x, _scipy(d).cdf).pvalue > 1e-4, name
    assert isinstance(draws.draw_setting(grid, "gt_mean", rng), float)


def test_counts_are_their_base_plus_a_poisson(grid):
    rng = draws.generator(3)
    for name, k in grid["counts"].items():
        x = draws.draw_count(grid, name, rng, 5_000)
        assert x.min() >= k["base"], name
        se = np.sqrt(max(k["mean"], 1e-9) / 5_000)
        assert abs(x.mean() - k["base"] - k["mean"]) < 5 * se + 1e-9, name


def test_unknown_names_are_refused(grid):
    rng = draws.generator(4)
    with pytest.raises(KeyError):
        draws.draw_setting(grid, "nope", rng)
    with pytest.raises(KeyError):
        draws.draw_count(grid, "nope", rng)
    with pytest.raises(KeyError):
        draws.draw_share(grid, "nope", rng)


def test_keyed_generators_repeat_and_differ(grid):
    a = draws.draw_design(grid, draws.generator(7, 1, 2))
    b = draws.draw_design(grid, draws.generator(7, 1, 2))
    assert a == b
    others = [
        draws.draw_design(grid, draws.generator(7, 1, k)) for k in range(20)
    ]
    assert len({json.dumps(o, sort_keys=True) for o in others}) > 1


def test_designs_fit_the_budget_and_report_on_their_days(grid):
    rng = draws.generator(5)
    for _ in range(3_000):
        d = draws.draw_design(grid, rng)
        assert d["weeks"] >= grid["floors"]["weeks"]
        n_p, n_a = d["n_places"], d["n_ages"]
        if d["strata_link"] != "independent":
            shortest = min(grid["max_delay_days"])
            cells = draws.triangle_cells(n_p * n_a, d["weeks"], shortest)
            assert cells <= grid["max_triangle_cells"]
        else:
            m = grid["max_independent"]
            if d["strata"] == "places_ages":
                assert n_p == 2
                assert n_a <= m // 2
            if d["strata"] == "ages":
                assert n_a <= m
        days = d["report_weekdays"]
        match d["report_days"]:
            case "daily":
                assert days == "all"
            case "weekdays":
                assert days == "Monday;Tuesday;Wednesday;Thursday;Friday"
            case "fixed_days":
                got = days.split(";")
                assert len(got) in (2, 3)
                assert got == sorted(got, key=draws.WEEK.index)
                # Fixed report days fall on weekdays, as the R engine's.
                assert set(got) <= set(draws.WEEK[:5])
            case "weekly":
                assert days in draws.WEEK
    assert draws.draw_design(grid, rng, weeks=9)["weeks"] == 9


def test_values_have_the_engine_s_shapes(grid):
    rng = draws.generator(6)
    for _ in range(1_000):
        d = draws.draw_design(grid, rng)
        v = draws.draw_values(grid, d, rng)
        n_p, n_s = d["n_places"], d["n_places"] * d["n_ages"]
        for key in (
            "ascertainment",
            "incubation_mean",
            "obs_sqrt_phi",
            "obs_family",
            "revise_over",
            "revise_move",
            "revise_keep",
        ):
            assert len(v[key]) == n_s, key
        for key in ("population", "place_lags", "mixing_x", "mixing_y"):
            assert len(v[key]) == n_p, key
        assert sorted(v["mixing_order"]) == list(range(1, n_p + 1))
        # The series, the longest delay possible and the horizon.
        span = d["weeks"] + max(OPTIONS) // 7 + 6
        assert len(v["walk_r"]) == span
        assert len(v["delay_walk"]) == span
        assert len(v["wave_scale"]) == v["waves"] + 1
        assert len(v["quiet_places"]) < n_p
        cases = set(range(1, n_p + 1)) - set(v["quiet_places"])
        assert set(v["late_places"]) <= cases - {min(cases)}
        assert set(v["shock_places"]) <= cases
        assert (d["shocks"] == "none") <= (v["shock_places"] == [])
        n = len(v["shock_places"])
        assert len(v["shock_at"]) == len(v["shock_mult"]) == n
        assert all(0 < a < 1 for a in v["ascertainment"])
        assert v["max_delay"] in OPTIONS
        if d["strata_link"] != "independent":
            cells = draws.triangle_cells(n_s, d["weeks"], v["max_delay"])
            assert cells <= grid["max_triangle_cells"]
        assert 0 <= v["offset"] <= 6
        if d["window"] == "early":
            assert v["variant_at"] == []
    one = draws.draw_values(grid, {**d, "offset": 3}, rng)
    assert one["offset"] == 3


def test_the_longest_delay_is_the_first_option_past_the_99th_centile():
    options = (21, 28, 42, 56, 84, 126)
    assert draws.max_delay_for(5, options) == 21
    assert draws.max_delay_for(21, options) == 21
    assert draws.max_delay_for(21.5, options) == 28
    assert draws.max_delay_for(57, options) == 84
    assert draws.max_delay_for(100, options) == 126
    assert draws.max_delay_for(500, options) == 126
    # R's qgeom counts failures: qgeom(0.99, 0.5) is 6.
    assert draws.delay_q99("none", {"delay_mean": 1.0}) == 6


def test_long_delay_tails_are_a_real_share(grid):
    """A fifth of draws have a 99% delay of 60 days or more (Germany's is
    61 to 81), some with a fast head as well, and the longest delay
    reaches the grid's last option."""
    rng = draws.generator(8)
    q99, head, longest = [], [], []
    for _ in range(3_000):
        d = draws.draw_design(grid, rng)
        v = draws.draw_values(grid, d, rng)
        parametric = d["delay_form"] != "non_parametric"
        family = d["delay_family"] if parametric else "none"
        q = draws.delay_q99(family, v)
        q99.append(q)
        if family == "lognormal":
            p0 = stats.lognorm.cdf(1, v["delay_sdlog"], scale=v["delay_median"])
            head.append(p0 >= 0.2 and q >= 60)
        longest.append(v["max_delay"])
    q99 = np.asarray(q99)
    assert np.mean(q99 >= 60) > 0.15
    assert np.mean(q99 >= 120) > 0.04
    assert np.mean(head) > 0.08
    assert max(OPTIONS) in longest


R_DRAWS = """
for (f in c("grid.R", "design.R", "infections.R", "delays.R")) {
  source(file.path(Sys.getenv("ENGINE"), f))
}
grid <- read_grid(Sys.getenv("GRID_FILE"))
set.seed(11)
n <- as.integer(Sys.getenv("N"))
for (i in seq_len(n)) {
  d <- draw_design(grid)
  v <- draw_values(grid, d)
  cat(jsonlite::toJSON(list(design = d, values = v), digits = NA), "\\n")
}
"""


def _r_draws(n):
    out = subprocess.run(
        ["Rscript", "-e", R_DRAWS],
        env={
            "ENGINE": str(ENGINE),
            "GRID_FILE": str(GRID_FILE),
            "N": str(n),
            "PATH": os.environ["PATH"],
            "HOME": os.environ.get("HOME", ""),
        },
        capture_output=True,
        text=True,
        timeout=900,
        check=True,
    )
    lines = out.stdout.splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _listed(x):
    return x if isinstance(x, list) else [x]


def _needs_r():
    if shutil.which("Rscript") is None:
        return "needs Rscript"
    probe = subprocess.run(
        [
            "Rscript",
            "-e",
            "quit(status = !all(sapply(c('yaml', 'jsonlite'), "
            "requireNamespace, quietly = TRUE)))",
        ],
        capture_output=True,
    )
    return None if probe.returncode == 0 else "needs R packages yaml, jsonlite"


@pytest.mark.skipif(_needs_r() is not None, reason=_needs_r() or "")
def test_draws_match_the_r_engine_s(grid):
    n = 3_000
    r = _r_draws(n)
    rng = draws.generator(12)
    py = []
    for _ in range(n):
        d = draws.draw_design(grid, rng)
        py.append({"design": d, "values": draws.draw_values(grid, d, rng)})
    for part in ("design", "values"):
        assert set(r[0][part]) == set(py[0][part]), part
        for key in py[0][part]:
            a = [_listed(x[part][key]) for x in r]
            b = [_listed(x[part][key]) for x in py]
            # The same number of elements per draw, on average.
            la = np.array([len(x) for x in a], dtype=float)
            lb = np.array([len(x) for x in b], dtype=float)
            se = np.sqrt((la.var() + lb.var()) / n) + 1e-9
            assert abs(la.mean() - lb.mean()) < 5 * se, (part, key)
            # One number per draw (its first element, and its mean), as a
            # draw's elements are not independent.
            ea = [x[0] for x in a if x]
            eb = [x[0] for x in b if x]
            if not ea or not eb:
                assert not ea, (part, key)
                assert not eb, (part, key)
                continue
            if isinstance(ea[0], str | bool):
                ca, cb = Counter(ea), Counter(eb)
                for level in set(ca) | set(cb):
                    pa, pb = ca[level] / len(ea), cb[level] / len(eb)
                    p = (ca[level] + cb[level]) / (len(ea) + len(eb))
                    se = np.sqrt(p * (1 - p) * (1 / len(ea) + 1 / len(eb)))
                    assert abs(pa - pb) < 5 * se + 1e-9, (part, key, level)
                continue
            ma = [np.mean(x) for x in a if x]
            mb = [np.mean(x) for x in b if x]
            for xa, xb in ((ea, eb), (ma, mb)):
                ks = stats.ks_2samp(
                    np.asarray(xa, float), np.asarray(xb, float)
                )
                assert ks.pvalue > 1e-4, (part, key, ks)


def test_incubation_is_wide_with_a_drawn_cv_and_family(grid):
    # Infection to reference: cholera-like 1-2 d to Ebola-like 9-12 d and
    # infection to admission 10-15 d (lead 13:55).
    d = _scipy(grid["distributions"]["incubation_mean"])
    lo, hi = d.ppf([0.025, 0.975])
    assert lo < 2
    assert hi > 20
    assert "incubation_cv" not in grid["constants"]
    assert set(grid["factors"]["incubation_family"]) == {"gamma", "lognormal"}
    cv = _scipy(grid["distributions"]["incubation_cv"]).ppf([0.025, 0.975])
    assert cv[0] < 0.3
    assert cv[1] > 0.9


def test_incubation_varies_by_place_and_age(grid):
    rng = draws.generator(21)
    seen = 0
    for _ in range(400):
        d = draws.draw_design(grid, rng)
        v = draws.draw_values(grid, d, rng)
        assert d["incubation_family"] in ("gamma", "lognormal")
        assert v["incubation_cv"] > 0
        n_p, n_a = d["n_places"], d["n_ages"]
        m = np.log(np.asarray(v["incubation_mean"])).reshape(n_p, n_a)
        # A place effect plus an age effect (strata are places, then ages).
        fit = m.mean(1, keepdims=True) + m.mean(0, keepdims=True) - m.mean()
        np.testing.assert_allclose(m, fit, atol=1e-9)
        if n_p > 1 and n_a > 1:
            seen += 1
            assert np.ptp(m.mean(1)) > 0
            assert np.ptp(m.mean(0)) > 0
    assert seen > 10


def test_settling_and_oscillating_paths_are_common(grid):
    # Lead 14:05: slow declines to R = 1 and oscillation around it are
    # common shapes, in every window; each trajectory has a rising
    # counterpart for early outbreaks not brought under control.
    levels = grid["factors"]["trajectory"]
    assert set(draws.RISING_PATH) == set(levels)
    total = sum(levels.values())
    assert levels["controlled"] / total >= 0.2
    assert levels["oscillating"] / total >= 0.15
    rng = draws.generator(31)
    paths = []
    for _ in range(2_000):
        d = draws.draw_design(grid, rng)
        if d["window"] == "established":
            paths.append(draws.draw_values(grid, d, rng)["path"])
    assert paths.count("controlled") / len(paths) > 0.15


def test_spike_pieces_are_rare_and_small(grid):
    # Shifts, reactive shifts, variants and independent place R are what
    # give R spikes; they stay, rarer and smaller (lead 14:05).
    assert grid["rates"]["shifts"] < 1
    assert grid["shares"]["reactive"] <= 0.5
    assert grid["rates"]["variants_per_year"] <= 1.5
    place = grid["factors"]["place_rt"]
    assert place["independent"] == min(place.values())
    assert place["correlated"] == max(place.values())
    size = _scipy(grid["distributions"]["shift_size"])
    assert size.ppf(0.975) < np.log(1.7)


def test_controlled_paths_start_near_r_1_8(grid):
    # Lead 01:20: slow declines from about R = 1.8 to 1.
    rng = draws.generator(32)
    starts = []
    for _ in range(3_000):
        d = draws.draw_design(grid, rng)
        v = draws.draw_values(grid, d, rng)
        if v["path"] == "controlled":
            starts.append(np.exp(v["amplitude"]))
    assert len(starts) > 200
    assert 1.65 < np.median(starts) < 1.95
    assert np.quantile(starts, 0.025) > 1.2
    assert np.quantile(starts, 0.975) < 2.8


def test_reporting_delay_shapes_are_not_capped_at_2(grid):
    # The joint model reports any gamma shape (#614), so the shape below
    # 2 that epinowcast's simulate branch needed is gone.
    assert grid["distributions"]["delay_shape"]["dist"] == "lognormal"
    shape = _scipy(grid["distributions"]["delay_shape"])
    assert shape.ppf(0.995) > 2
    assert shape.ppf(0.025) < 0.3


def test_strata_mostly_deviate_moderately_from_a_shared_path(grid):
    # Lead 01:20: moderate strata deviations. Strata are mostly linked
    # (one outbreak), not each an outbreak of its own.
    link = grid["factors"]["strata_link"]
    assert link["independent"] / sum(link.values()) <= 0.25
    coupling = _scipy(grid["distributions"]["coupling"])
    assert coupling.ppf(0.025) >= 0.5


def test_oscillating_paths_come_in_every_window(grid):
    rng = draws.generator(41)
    by_window = Counter()
    oscillating = Counter()
    for _ in range(4_000):
        d = draws.draw_design(grid, rng)
        v = draws.draw_values(grid, d, rng)
        by_window[d["window"]] += 1
        oscillating[d["window"]] += v["path"] == "oscillating"
    for window in ("early", "established", "endemic"):
        assert oscillating[window] / by_window[window] > 0.05, window


def test_reporting_delays_put_real_mass_on_99th_centiles_of_40_to_90_days(
    grid,
):
    # Germany's national 99th centile is about 69 days.
    rng = draws.generator(42)
    q99 = []
    for _ in range(3_000):
        d = draws.draw_design(grid, rng)
        v = draws.draw_values(grid, d, rng)
        parametric = d["delay_form"] != "non_parametric"
        family = d["delay_family"] if parametric else "none"
        q99.append(draws.delay_q99(family, v))
    q99 = np.asarray(q99)
    assert np.median(q99) >= 40
    assert np.mean((q99 >= 40) & (q99 <= 90)) >= 0.28


def test_places_nest_in_two_to_six_regions(grid):
    rng = draws.generator(43)
    seen = set()
    for _ in range(2_000):
        d = draws.draw_design(grid, rng)
        v = draws.draw_values(grid, d, rng)
        n_p, k = int(d["n_places"]), v["n_regions"]
        region = v["place_region"]
        assert len(region) == n_p
        if n_p < 4:
            assert k == 1
        else:
            assert 2 <= k <= min(6, n_p // 2)
            seen.add(k)
        # Every region has a place, and regions are as even as they can be.
        sizes = np.bincount(region, minlength=k + 1)[1:]
        assert sizes.min() >= 1 and sizes.max() - sizes.min() <= 1
    assert seen == {2, 3, 4, 5, 6}
