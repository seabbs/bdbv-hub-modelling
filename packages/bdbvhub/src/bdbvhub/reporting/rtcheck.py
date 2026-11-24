# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""A sense check of a renewal fit's `wk R_t` quantiles before submission.

For the country, each province and each fitted health zone, the page
shows the fit's weekly effective Rt (`nowcastrenewal.forecaster.RT_FILE`),
by week of infection, with BVDOutbreakSize's Rt overlaid where the
release has one, and under it the cases by week of onset: reported,
imputed and the fit's nowcast of the final count. The weeks the round
submits are marked. It lists the zones with cases that the fit has no Rt
for. `bdbvhub rt-check` writes it from a fit directory.
"""

from __future__ import annotations

import datetime as dt
import html
import io
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from nowcasteval.plots import HUB_GREYS, THEMES, model_colour, theme
from nowcastutils.hierarchy import is_zone, parent

__all__ = [
    "RT_LEVELS",
    "bvd_rt",
    "count_bands",
    "plot_rt_check",
    "rt_bands",
    "rt_check_from_fit",
    "rt_check_page",
    "submitted_weeks",
]

RT_LEVELS: Final = (0.05, 0.25, 0.5, 0.75, 0.95)
"""The `wk R_t` levels drawn: the median and the 50% and 90% intervals."""
BAND_COLUMNS: Final = ("q05", "q25", "q50", "q75", "q95")
NATIONAL: Final = "CD"


def rt_bands(rt: pd.DataFrame) -> pd.DataFrame:
    """Return the median and 50% and 90% intervals of `wk R_t` rows.

    Parameters
    ----------
    rt : pandas.DataFrame
        `location`, `target_end_date`, `output_type_id` and `value`, as a
        fit's `rt.csv.gz`.

    Returns
    -------
    pandas.DataFrame
        `location`, `target_end_date` and `BAND_COLUMNS`.

    Raises
    ------
    ValueError
        If a level in `RT_LEVELS` is missing.
    """
    levels = rt["output_type_id"].astype(float).round(3)
    missing = set(RT_LEVELS) - set(levels)
    if missing:
        raise ValueError(f"wk R_t rows lack the levels {sorted(missing)}.")
    rows = rt.assign(output_type_id=levels).loc[levels.isin(RT_LEVELS)]
    wide = rows.pivot_table(
        index=["location", "target_end_date"],
        columns="output_type_id",
        values="value",
    )
    wide.columns = list(BAND_COLUMNS)
    return wide.reset_index()


def count_bands(samples: pd.DataFrame) -> pd.DataFrame:
    """Return the median and 90% interval of final-count samples.

    Parameters
    ----------
    samples : pandas.DataFrame
        `location`, `target_end_date`, `sample` and `value`.

    Returns
    -------
    pandas.DataFrame
        `location`, `target_end_date`, `q05`, `q50` and `q95`.
    """
    grp = samples.groupby(["location", "target_end_date"])["value"]
    out = grp.quantile(np.array([0.05, 0.5, 0.95])).unstack()
    out.columns = ["q05", "q50", "q95"]
    return out.reset_index()


def bvd_rt(
    timeseries: pd.DataFrame,
    zones: pd.DataFrame | None,
    provinces: Mapping[str, str],
    z90: float,
) -> pd.DataFrame:
    """Return BVDOutbreakSize's Rt by hub location and date.

    Provinces and the country take the release's daily Rt series
    (`province_timeseries.csv`); zones take their one estimate at the
    release's cut-off, from the BVD term's zone rows.

    Parameters
    ----------
    timeseries : pandas.DataFrame
        The release's `province_timeseries.csv`.
    zones : pandas.DataFrame or None
        `nowcastrenewal.zones.zone_bvd_rows`: `members` (the zone code),
        `as_of`, `log_mean` and `log_sd`.
    provinces : mapping
        BVDOutbreakSize patch key to hub province code
        (`nowcastrenewal.bvd.BVD_PROVINCES`).
    z90 : float
        The standard normal 95% quantile, to turn an SD into a 90%
        interval.

    Returns
    -------
    pandas.DataFrame
        `location`, `date`, `median`, `lower_90` and `upper_90`.
    """
    rt = timeseries.loc[timeseries["series"] == "rt"]
    key = rt["area"].str.lower().str.replace("-", "_")
    codes = key.map({**provinces, "national": NATIONAL})
    out = [
        rt.assign(location=codes)
        .dropna(subset=["location"])
        .loc[:, ["location", "date", "median", "lower_90", "upper_90"]]
    ]
    if zones is not None and len(zones):
        mean = zones["log_mean"].to_numpy(float)
        sd = zones["log_sd"].to_numpy(float)
        out.append(
            pd.DataFrame(
                {
                    "location": zones["members"].astype(str),
                    "date": zones["as_of"].astype(str),
                    "median": np.exp(mean),
                    "lower_90": np.exp(mean - z90 * sd),
                    "upper_90": np.exp(mean + z90 * sd),
                }
            )
        )
    return pd.concat(out, ignore_index=True)


def submitted_weeks(reference_date: str) -> tuple[str, str]:
    """Return the weeks a round submits `wk R_t` for.

    The target takes horizons 0 and -1 only: the reference week and the
    week before.

    Parameters
    ----------
    reference_date : str
        The round's reference date (a Saturday).

    Returns
    -------
    tuple of str
        The week before and the reference week.
    """
    ref = dt.date.fromisoformat(reference_date)
    return ((ref - dt.timedelta(days=7)).isoformat(), ref.isoformat())


def _dates(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values.astype(str))


def plot_rt_check(
    location: str,
    name: str,
    rt: pd.DataFrame,
    bvd: pd.DataFrame,
    cases: pd.DataFrame,
    nowcast: pd.DataFrame,
    submitted: Sequence[str],
    mode: str = "light",
) -> Figure:
    """Draw one location's Rt over its cases.

    Parameters
    ----------
    location : str
        The location code.
    name : str
        Its name, for the title.
    rt : pandas.DataFrame
        `rt_bands` rows of the location.
    bvd : pandas.DataFrame
        `bvd_rt` rows of the location (may be empty).
    cases : pandas.DataFrame
        The release's counts of the location by onset week:
        `target_end_date` and `observed` and `imputed`, or `total` alone
        for a zone.
    nowcast : pandas.DataFrame
        `count_bands` rows of the location.
    submitted : sequence of str
        The weeks the round submits.
    mode : str
        `"light"` or `"dark"`.

    Returns
    -------
    matplotlib.figure.Figure
        Rt above, cases below, on one date axis.
    """
    ink = THEMES[mode]
    ours = model_colour("renewalbvd", ["renewalbvd"], mode)
    grey = HUB_GREYS[mode][0]
    fig, (top, low) = plt.subplots(
        2,
        1,
        figsize=(8, 5.6),
        sharex=True,
        layout="constrained",
        height_ratios=(3, 2),
    )
    x = _dates(rt["target_end_date"]) - pd.Timedelta(days=3)
    top.fill_between(x, rt["q05"], rt["q95"], color=ours, alpha=0.2, lw=0)
    top.fill_between(x, rt["q25"], rt["q75"], color=ours, alpha=0.35, lw=0)
    top.plot(x, rt["q50"], color=ours, label="Our Rt (by infection week)")
    if len(bvd):
        bx = _dates(bvd["date"])
        if len(bvd) > 1:
            top.fill_between(
                bx, bvd["lower_90"], bvd["upper_90"], color=grey, alpha=0.15
            )
            top.plot(bx, bvd["median"], color=grey, lw=1.2, ls="--")
        else:
            top.errorbar(
                bx,
                bvd["median"],
                yerr=[
                    bvd["median"] - bvd["lower_90"],
                    bvd["upper_90"] - bvd["median"],
                ],
                fmt="D",
                color=grey,
                ms=4,
            )
        top.plot([], [], color=grey, ls="--", label="BVDOutbreakSize Rt")
    top.axhline(1.0, color=ink["muted"], lw=0.8)
    for week in submitted:
        end = pd.Timestamp(week)
        top.axvspan(
            end - pd.Timedelta(days=6.5),
            end + pd.Timedelta(days=0.5),
            color=ink["muted"],
            alpha=0.15,
            lw=0,
        )
    top.plot([], [], lw=6, alpha=0.3, color=ink["muted"], label="Submitted")
    top.set_ylabel("Rt (median, 50%, 90%)")
    top.set_title(f"{name} ({location})", loc="left")
    top.legend(loc="upper right", ncols=3)
    cx = _dates(cases["target_end_date"]) - pd.Timedelta(days=3)
    if "observed" in cases:
        low.bar(
            cx,
            cases["observed"],
            width=6,
            color=ink["ink"],
            alpha=0.6,
            label="Reported (onset known)",
        )
        low.bar(
            cx,
            cases["imputed"],
            width=6,
            bottom=cases["observed"],
            color=grey,
            alpha=0.6,
            label="Imputed onset",
        )
    else:
        low.bar(
            cx,
            cases["total"],
            width=6,
            color=ink["ink"],
            alpha=0.6,
            label="Reported",
        )
    if len(nowcast):
        nx = _dates(nowcast["target_end_date"]) - pd.Timedelta(days=3)
        low.fill_between(
            nx, nowcast["q05"], nowcast["q95"], color=ours, alpha=0.2, lw=0
        )
        low.plot(nx, nowcast["q50"], color=ours, label="Nowcast (final)")
    low.set_ylabel("Cases by onset week")
    low.legend(loc="upper left", ncols=3)
    low.xaxis.set_major_locator(mdates.MonthLocator())  # type: ignore[no-untyped-call]
    low.xaxis.set_major_formatter(mdates.DateFormatter("%b"))  # type: ignore[no-untyped-call]
    return fig


def _svg(fig: Figure) -> str:
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    text = buf.getvalue()
    return text[text.index("<svg") :]


def _order(codes: Sequence[str]) -> list[str]:
    """Return the country, then each province followed by its zones."""
    provinces = sorted(c for c in codes if c != NATIONAL and not is_zone(c))
    out = [NATIONAL] if NATIONAL in codes else []
    for p in provinces:
        out.append(p)
        out.extend(sorted(c for c in codes if is_zone(c) and parent(c) == p))
    return out


def rt_check_page(
    rt: pd.DataFrame,
    bvd: pd.DataFrame,
    cases: pd.DataFrame,
    nowcast: pd.DataFrame,
    names: Mapping[str, str],
    silent: Sequence[str],
    meta: Mapping[str, Any],
) -> str:
    """Return the self-contained sense-check page.

    Parameters
    ----------
    rt : pandas.DataFrame
        `rt_bands` of every location.
    bvd : pandas.DataFrame
        `bvd_rt`.
    cases : pandas.DataFrame
        The release's counts by `location` and `target_end_date`:
        `observed` and `imputed` for the country and provinces, `total`
        for every location.
    nowcast : pandas.DataFrame
        `count_bands` of every location.
    names : mapping
        Location code to name.
    silent : sequence of str
        Zones with cases in the release that the fit has no Rt for.
    meta : mapping
        `model`, `vintage`, `reference_date`, `bvd` (the release tag)
        and `commit`, for the header.

    Returns
    -------
    str
        The HTML page, with each location's figure inline in a light
        and a dark version and a location picker.
    """
    submitted = submitted_weeks(str(meta["reference_date"]))
    codes = _order(sorted(set(rt["location"])))
    options, panels = [], []
    for i, code in enumerate(codes):
        label = f"{names.get(code, code)} ({code})"
        depth = "zone" if is_zone(code) else "province"
        if code == NATIONAL:
            depth = "country"
        options.append(
            f'<option value="{i}">'
            f"{'&nbsp;&nbsp;' * (depth == 'zone')}{html.escape(label)}"
            "</option>"
        )
        mine = cases.loc[cases["location"] == code]
        if is_zone(code) or mine[["observed", "imputed"]].isna().all().all():
            mine = mine.loc[:, ["target_end_date", "total"]]
        figs = []
        for mode in ("light", "dark"):
            with theme(mode):
                fig = plot_rt_check(
                    code,
                    names.get(code, code),
                    rt.loc[rt["location"] == code],
                    bvd.loc[bvd["location"] == code],
                    mine.sort_values("target_end_date"),
                    nowcast.loc[nowcast["location"] == code],
                    submitted,
                    mode,
                )
                figs.append(f'<div class="{mode}">{_svg(fig)}</div>')
        hidden = "" if i == 0 else " hidden"
        panels.append(
            f'<div class="panel" id="p{i}"{hidden}>{"".join(figs)}</div>'
        )
    quiet = ", ".join(f"{names.get(z, z)} ({z})" for z in sorted(silent))
    return _PAGE.format(
        model=html.escape(str(meta["model"])),
        vintage=html.escape(str(meta["vintage"])),
        reference=html.escape(str(meta["reference_date"])),
        weeks=" and ".join(submitted),
        bvd=html.escape(str(meta.get("bvd") or "none")),
        commit=html.escape(str(meta.get("commit") or "")[:9]),
        options="\n".join(options),
        panels="\n".join(panels),
        n_silent=len(silent),
        silent=html.escape(quiet) or "none",
    )


def rt_check_from_fit(fit_dir: Path | str, data_dir: Path | str) -> str:
    """Return the sense-check page of a renewal fit directory.

    Parameters
    ----------
    fit_dir : pathlib.Path or str
        A fit directory with `rt.csv.gz` (`fit-store unpack`).
    data_dir : pathlib.Path or str
        The tidy hub data (`target.csv`, `health_zones.csv.gz`) and the
        BVDOutbreakSize release cache (`bvd/`).

    Returns
    -------
    str
        `rt_check_page`.

    Raises
    ------
    FileNotFoundError
        If the fit directory has no `rt.csv.gz`, as a fit made before
        the `wk R_t` output.
    """
    import json

    from nowcastbdbv.data.tidy import load_health_zones, load_target
    from nowcasteval.plots import location_names
    from nowcastrenewal import bvd as bvd_mod
    from nowcastrenewal.forecaster import RT_FILE, SAMPLES_FILE
    from nowcastrenewal.zones import zone_bvd_rows

    fit, data = Path(fit_dir), Path(data_dir)
    if not (fit / RT_FILE).exists():
        raise FileNotFoundError(f"{fit} has no {RT_FILE}: refit after #835.")
    record = json.loads((fit / "fit.json").read_text())
    diagnostics = json.loads((fit / "diagnostics.json").read_text())
    vintage = str(record["vintage"])
    rt = rt_bands(pd.read_csv(fit / RT_FILE, dtype={"target_end_date": str}))
    samples = pd.read_csv(fit / SAMPLES_FILE, dtype={"target_end_date": str})
    provinces = samples.loc[~samples["location"].map(is_zone)]
    country = (
        provinces.loc[provinces["location"] != NATIONAL]
        .groupby(["target_end_date", "sample"], as_index=False)["value"]
        .sum()
        .assign(location=NATIONAL)
    )
    nowcast = count_bands(pd.concat([samples, country], ignore_index=True))
    target = load_target(data)
    target = target.loc[
        (target["kind"] == "real") & (target["vintage"] == vintage)
    ]
    zones = load_health_zones(data)
    zones = zones.loc[zones["vintage"] == vintage]
    cases = pd.concat(
        [
            target.loc[
                :,
                ["location", "target_end_date", "observed", "imputed", "total"],
            ],
            zones.loc[:, ["location", "target_end_date", "total"]],
        ],
        ignore_index=True,
    )
    fitted = {p for p in set(rt["location"]) if not is_zone(p)}
    with_cases = zones.groupby("location")["total"].sum()
    silent = sorted(
        z
        for z in with_cases.index[with_cases > 0]
        if parent(z) in fitted and z not in set(rt["location"])
    )
    tag = (diagnostics.get("bvd") or {}).get("tag")
    overlay = pd.DataFrame(
        columns=["location", "date", "median", "lower_90", "upper_90"]
    )
    if tag:
        cache = data / "bvd"
        release = bvd_mod.load_release(cache / tag)
        overlay = bvd_rt(
            pd.read_csv(cache / tag / "province_timeseries.csv"),
            zone_bvd_rows(release, load_health_zones(data), cache_dir=cache),
            bvd_mod.BVD_PROVINCES,
            bvd_mod.Z90,
        )
    names = {**location_names(target), **location_names(zones)}
    return rt_check_page(
        rt,
        overlay,
        cases,
        nowcast,
        names,
        silent,
        {
            "model": record.get("model", ""),
            "vintage": vintage,
            "reference_date": record["reference_date"],
            "bvd": tag,
            "commit": record.get("commit", ""),
        },
    )


_PAGE: Final = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rt sense check</title>
<style>
:root {{ --bg: #fcfcfa; --ink: #1f1e1c; --muted: #5f5e5a; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg: #1b1b19; --ink: #ecebe6; --muted: #b4b2a9; }}
}}
body {{ background: var(--bg); color: var(--ink); margin: 0 auto;
  max-width: 60rem; padding: 1rem; font: 15px/1.5 system-ui, sans-serif; }}
p, li {{ color: var(--ink); }} .note {{ color: var(--muted); }}
select {{ font: inherit; padding: 0.3rem; max-width: 100%; }}
svg {{ max-width: 100%; height: auto; }}
.dark {{ display: none; }}
@media (prefers-color-scheme: dark) {{
  .light {{ display: none; }} .dark {{ display: block; }}
}}
</style></head><body>
<h1>Rt sense check</h1>
<p>{model} fit to the release of {vintage}, round {reference}
(BVDOutbreakSize release {bvd}, commit {commit}).
The shaded weeks, {weeks}, are the weeks submitted to <code>wk R_t</code>.</p>
<p class="note">Our Rt is indexed to the week of infection (each week is
drawn at its middle). Cases are by week of onset, which lags infection by
the incubation period, so a change in Rt shows in the cases about one to
two weeks later. BVDOutbreakSize's Rt is drawn as published: a daily series
for provinces and the country, and one estimate at its cut-off for each
zone.</p>
<label for="loc">Location</label>
<select id="loc">
{options}
</select>
{panels}
<h2>Zones with cases but no Rt ({n_silent})</h2>
<p>{silent}</p>
<script>
const s = document.getElementById("loc");
s.addEventListener("change", () => {{
  document.querySelectorAll(".panel").forEach(
    (p) => (p.hidden = p.id !== "p" + s.value));
}});
</script>
</body></html>
"""
