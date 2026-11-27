"""The simulation report (workflow/scripts/sim_report.py)."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "workflow" / "scripts"))
import grid_summaries as gs  # noqa: E402
import sim_report as sr  # noqa: E402
from test_grid_summaries import _batch  # noqa: E402

SETTINGS = {
    "outbreak": "1-0001",
    "window": "early",
    "path": "growth",
    "trajectory": "growth",
    "delay_change": "static",
    "gt_family": "gamma",
    "report_days": "daily",
    "report_weekdays": "all",
    "reference_dow": "no",
    "report_dow": "no",
    "delay_dow": "no",
    "revisions": "no",
    "shocks": "none",
    "strata": "single",
    "strata_link": "linked",
    "observation_strata": "pooled",
    "reporting_strata": "pooled",
    "n_places": 1,
    "n_ages": 1,
    "weeks": 17,
    "gt_mean": 7.0,
    "gt_cv": 0.5,
    "incubation_mean": 5.0,
    "ascertainment": 0.3,
    "delay_mean": 2.0,
    "max_delay": 21,
    "attack": "0.2",
    "population": "100000",
    "variant_at": np.nan,
    "tries": 2,
    "designs": 1,
}


def _round(path: Path) -> Path:
    _batch(path)
    pd.DataFrame([SETTINGS]).to_csv(path / "settings-1.csv", index=False)
    pd.DataFrame(
        [{"outbreak": "1-0001", "try": 1, "reason": "no cases"}]
    ).to_csv(path / "failures-1.csv", index=False)
    out = path / "outbreaks.csv"
    gs.outbreaks(path).to_csv(out, index=False)
    return out


def test_germany_reads_the_development_half_of_the_national_series(tmp_path):
    dates = pd.date_range("2021-03-01", "2022-03-31").strftime("%Y-%m-%d")
    rows = []
    for loc, age in (("DE", "00+"), ("DE-BY", "00+"), ("DE", "05-14")):
        for d in dates:
            row = {"date": d, "location": loc, "age_group": age}
            row |= {f"value_{k}d": 0 for k in range(81)}
            row |= {"value_0d": 3, "value_2d": 1, "value_>80d": 0}
            rows.append(row)
    wide = pd.DataFrame(rows)
    # After the split: never read.
    wide.loc[wide["date"] >= "2022-02-10", "value_0d"] = 1000
    path = tmp_path / "tri.csv"
    wide.to_csv(path, index=False)
    out = sr.germany(path, "2022-02-10").set_index(["kind", "statistic"])
    assert out.loc[("national", "day0_share"), "value"] == 0.75
    assert out.loc[("national", "delay_q99"), "value"] == 2
    assert out.loc[("national", "mean_daily"), "value"] == 4
    assert {"national", "states", "ages"} == set(out.index.get_level_values(0))


def test_a_round_renders_one_self_contained_page(tmp_path):
    sims = tmp_path / "sims"
    sims.mkdir()
    summary = _round(sims)
    de = pd.DataFrame(
        {"kind": "national", "statistic": ["day0_share", "delay_q99"],
         "value": [0.22, 69]}
    )  # fmt: skip
    de_path = tmp_path / "de.csv"
    de.to_csv(de_path, index=False)
    delays = tmp_path / "delays.json"
    delays.write_text(json.dumps(_reference()))
    out = tmp_path / "report" / "sim-report.html"
    sr.render(
        out,
        "Simulated outbreaks, round 1",
        ROOT / "paper" / "data" / "reference-values.csv",
        de_path,
        delays,
        [f"r1={sims}:{summary}"],
    )
    page = out.read_text()
    assert page.startswith("<!doctype html>")
    assert "<title>Simulated outbreaks, round 1</title>" in page
    # The gallery: the early outbreak, with its figure inline.
    assert "early window: 1-0001 (r1)" in page
    # The random gallery and the round's shares of R path pieces.
    assert "random 1: 1-0001 (r1)" in page
    assert "<h3>Shares over the round</h3>" in page
    assert "<td>R path</td><td>growth</td><td>1</td><td>100%</td>" in page
    assert page.count("data:image/png;base64,") >= 2
    assert 'src="http' not in page
    # Flags: the redrawn draw and the audit's combinations.
    assert "1 draws were redrawn, of 2 drawn" in page
    assert "no cases" in page
    assert "<h3>Odd combinations</h3>" in page
    assert "Fewer than 100 cases" in page
    # R(t) levels each run, so implausible R shows.
    assert "<h3>R(t)</h3>" in page
    # The delays' structure against Germany's and BDBV's.
    assert "<h2>3. Structure of the delays</h2>" in page
    assert "Germany" in page
    assert "BDBV" in page
    assert "<h3>Each stratum's delays, week by week</h3>" in page


def test_shares_count_shifts_and_reactive_ones():
    d = pd.DataFrame(
        {"shift_at": [np.nan, "0.2", "0.1;0.5;0.7;0.9"],
         "shift_reactive": [np.nan, "FALSE", "FALSE;TRUE;TRUE;FALSE"],
         "variants": [False, True, False]}
    )  # fmt: skip
    page = sr.shares(d)
    assert "<td>R shifts</td><td>0</td><td>1</td>" in page
    assert "<td>3+</td><td>1</td>" in page
    assert "<td>a reactive shift</td><td>no</td><td>2</td>" in page
    assert "<td>yes</td><td>1</td>" in page


def test_the_random_gallery_is_seeded_and_ignores_regimes():
    d = pd.DataFrame(
        {"round": "r1", "outbreak": [f"1-{i:04d}" for i in range(1, 101)]}
    )
    a = [o["outbreak"] for _, o in sr.pick_random(d)]
    b = [
        o["outbreak"]
        for _, o in sr.pick_random(d.sample(frac=1, random_state=3))
    ]
    assert a == b
    assert len(set(a)) == 24
    assert a != [f"1-{i:04d}" for i in range(1, 25)]


def test_reference_bands_are_one_per_disease_in_the_summaries_units():
    refs = pd.read_csv(ROOT / "paper" / "data" / "reference-values.csv")
    gt = {d: (lo, hi) for d, lo, hi in sr.bands_of(refs, "generation_time")}
    assert gt["ebola*"] == (10, 16.1)
    growth = {d: (lo, hi) for d, lo, hi in sr.bands_of(refs, "growth_rate")}
    # Dengue's 0.195 a week is 0.028 a day; Ebola's doubling in 12.84 to
    # 30.2 days is ln 2 over those.
    assert np.isclose(growth["dengue"][1], 0.213 / 7)
    assert np.isclose(growth["ebola"][1], np.log(2) / 12.84)


def test_diseases_give_each_summary_s_range_against_published_values():
    refs = pd.read_csv(ROOT / "paper" / "data" / "reference-values.csv")
    d = pd.DataFrame({"gt_mean": np.linspace(0, 100, 101), "other": 1.0})
    out = sr.diseases(d, refs)
    assert list(out["summary"]) == ["Generation time mean (days)"]
    row = out.iloc[0]
    assert (row["sim_q5"], row["sim_q50"], row["sim_q95"]) == (5, 50, 95)
    assert "ebola* 10 to 16.1" in row["published"]
    assert "10.1093/aje/kwu209" in row["sources"].split("; ")


def test_r_levels_count_days_with_infections_by_threshold(tmp_path):
    # Days above 2, 3 and 5 among days with an infection, per round; an
    # empty stratum's R does not count.
    sims = tmp_path / "sims"
    sims.mkdir()
    r = [1.0] * 6 + [2.5, 3.5, 6.0, 9.0]
    pd.DataFrame(
        {
            "day": range(10),
            "infections": [5.0] * 9 + [0.0],
            "onsets": 1.0,
            "r": r,
            "reff": r,
            "date": pd.date_range("2026-01-05", periods=10).strftime(
                "%Y-%m-%d"
            ),
            "location": "all",
            "outbreak": "1-0001",
        }
    ).to_csv(sims / "latent-1.csv.gz", index=False)
    table = sr.r_levels({"r1": sims})
    row = table.set_index("round").loc["r1"]
    assert row["days"] == 9
    assert np.isclose(row["above_2"], 3 / 9)
    assert np.isclose(row["above_3"], 2 / 9)
    assert np.isclose(row["above_5"], 1 / 9)
    assert row["max"] == 6.0
    assert np.isclose(row["outbreaks_above_4"], 1.0)
    page = sr.r_section({"r1": sims})
    assert "<h3>R(t)</h3>" in page
    assert "data:image/png;base64," in page


def _wide(dates, locations):
    rows = []
    for loc in locations:
        for d in dates:
            row = {"date": d, "location": loc, "age_group": "00+"}
            row |= {f"value_{k}d": 0 for k in range(81)}
            row |= {"value_0d": 3, "value_2d": 1, "value_>80d": 0}
            rows.append(row)
    return pd.DataFrame(rows)


def test_german_delay_rows_are_the_states_in_the_development_half():
    dates = pd.date_range("2021-03-01", "2022-03-31").strftime("%Y-%m-%d")
    wide = _wide(dates, ["DE", "DE-BY", "DE-BE"])
    rows = sr.germany_delay_rows(wide, "2022-02-10")
    assert set(rows["stratum"]) == {"DE-BY", "DE-BE"}
    assert rows["ref"].min() == pd.Timestamp(sr.DEV_FIRST)
    assert rows["ref"].max() < pd.Timestamp("2022-02-10")
    delays = (rows["rep"] - rows["ref"]).dt.days
    assert set(delays[rows["count"] > 0]) == {0, 2}


def test_bdbv_delay_rows_are_province_changes_between_releases():
    daily = pd.DataFrame(
        {"vintage": ["2026-10-01", "2026-10-01", "2026-10-08",
                     "2026-10-08", "2026-10-08", "2026-10-08"],
         "location": ["CD", "CD-IT", "CD", "CD-IT", "CD-IT", "CD-IT"],
         "onset_date": ["2026-10-01", "2026-10-01", "2026-10-01",
                        "2026-10-01", "2026-10-03", "2026-09-01"],
         "observed": [3, 3, 9, 5, 4, 7]}
    )  # fmt: skip
    daily["cutoff"] = daily["vintage"]
    rows = sr.bdbv_delay_rows(daily, 42).sort_values(["ref", "rep"])
    # The country is the provinces' total, not a stratum; onsets before
    # the first release are left out.
    assert set(rows["stratum"]) == {"CD-IT"}
    got = [
        (r.ref.strftime("%m-%d"), r.rep.strftime("%m-%d"), r.count)
        for r in rows.itertuples()
    ]
    assert got == [
        ("10-01", "10-01", 3),
        ("10-01", "10-08", 2),
        ("10-03", "10-08", 4),
    ]


def test_bdbv_delay_rows_leave_out_a_cumulative_first_release():
    # A dump on 2026-08-01, more than 42 days before the next release's
    # cut-off: left out as the backtests leave it out, so the first
    # usable release is the first one read.
    daily = pd.DataFrame(
        {"vintage": ["2026-08-01", "2026-10-01", "2026-10-08"],
         "location": "CD-IT",
         "onset_date": ["2026-08-01", "2026-10-01", "2026-10-01"],
         "observed": [40, 3, 5]}
    )  # fmt: skip
    daily["cutoff"] = daily["vintage"]
    rows = sr.bdbv_delay_rows(daily, 42).sort_values(["ref", "rep"])
    assert list(rows["rep"].dt.strftime("%m-%d")) == ["10-01", "10-08"]
    assert list(rows["count"]) == [3, 2]
    # Within the delay window the first release is kept.
    kept = sr.bdbv_delay_rows(daily, 70)
    assert "08-01" in set(kept["rep"].dt.strftime("%m-%d"))


def _reference():
    from flowlib import delay_structure as ds

    rows = pd.DataFrame(
        {"stratum": "A",
         "ref": pd.date_range("2026-01-05", periods=60),
         "count": 10}
    )  # fmt: skip
    rows["rep"] = rows["ref"] + pd.Timedelta(days=2)
    s = ds.structure(rows)
    entry = {"structure": s, "scalars": ds.scalars(s), "curves": ds.curves(s)}
    return {
        "germany": {"label": "Germany", **entry},
        "bdbv": {"label": "BDBV", **entry},
    }


def test_the_delay_reference_has_germany_and_bdbv(tmp_path):
    dates = pd.date_range("2021-03-01", "2021-06-30").strftime("%Y-%m-%d")
    tri = tmp_path / "tri.csv"
    _wide(dates, ["DE", "DE-BY"]).to_csv(tri, index=False)
    daily = tmp_path / "daily.csv"
    pd.DataFrame(
        {"vintage": ["2026-10-01"] * 20 + ["2026-10-08"] * 20,
         "location": "CD-IT",
         "onset_date": [str(d.date()) for d in
                        pd.date_range("2026-09-20", periods=20)] * 2,
         "observed": [2] * 20 + [3] * 20}
    ).assign(cutoff=lambda d: d["vintage"]).to_csv(
        daily, index=False
    )  # fmt: skip
    got = sr.delay_reference(tri, "2022-02-10", daily, 42)
    assert set(got) == {"germany", "bdbv"}
    for entry in got.values():
        assert {"short", "label", "structure", "scalars", "curves"} <= set(
            entry
        )
    assert got["germany"]["structure"]["overall"]["mean"] == 0.5
    json.dumps(got)


def test_the_delay_structure_factors_are_in_the_layers():
    factors = {f for _, _, fs in sr.LAYERS for f in fs}
    for name in (
        "delay_change_strata",
        "delay_change_switch",
        "delay_change_combine",
        "delay_counts",
        "delay_strata_noise",
        "delay_jitter",
        "bursts",
    ):
        assert name in factors


def test_the_report_compares_each_new_delay_number():
    from flowlib import delay_structure as ds

    s = ds.structure(
        pd.DataFrame(
            {"stratum": ["A", "B"] * 30,
             "ref": pd.date_range("2026-01-05", periods=60),
             "rep": pd.date_range("2026-01-07", periods=60),
             "count": 10}
        )
    )  # fmt: skip
    keys = {k for k, _ in sr.SCALARS}
    assert set(ds.scalars(s)) <= keys


def test_the_strata_figure_draws_each_data_set_s_strata():
    fig = sr.strata_figure(_reference())
    assert fig is not None
    assert len(fig.axes) == 2 * len(sr.STRATA_PANELS)
    sr.plt.close(fig)
    assert sr.strata_figure({}) is None
