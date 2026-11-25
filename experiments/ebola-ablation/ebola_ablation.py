"""Ablate the BDBV hub data towards the simulations, and simulations towards it.

Each variant is a tidy hub data folder, as `bdbvhub data` writes, so the
BDBV check (`bdbvhub ft-check forecast` and `report`, `bdbvhub.ftcheck`)
forecasts and scores it unchanged.

- `bdbv VARIANT --src DATA --out DIR`: the hub data with one change
  towards the simulations (`BDBV_VARIANTS`).
- `sim VARIANT --sims FILE... --out DIR`: held-out simulated outbreaks as
  hub data, each outbreak one province and its leaves that province's
  zones, with one BDBV feature imposed (`SIM_VARIANTS`).
- `summary DIR --out FILE`: the check's tables of one variant's folders
  (`ftcheck_report.skill_tables`, `calibration`) as JSON.
- `config --base FILE --samples N --max-delay D --out FILE`: the backtest
  config with the ablation's draws and longest delay.

`ebola_ablation_report.py` writes the page; README.md gives every step.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

D = 42
"""The longest reporting delay, days."""

PROVINCES = ("CD-IT", "CD-NK", "CD-HU", "CD-TO", "CD-BU", "CD-SK", "CD-TA")
"""Hub province codes the simulated outbreaks take, in order."""


def saturday(days: pd.Series) -> pd.Series:
    """Return the Saturday ending each day's epiweek, as ISO strings."""
    d = pd.to_datetime(days)
    return (
        d + pd.to_timedelta((5 - d.dt.day_of_week) % 7, unit="D")
    ).dt.strftime("%Y-%m-%d")


# ---------------------------------------------------------------- writing


def write_hub(
    out: Path,
    src: Path,
    daily: pd.DataFrame,
    zones: pd.DataFrame,
    origins: list[str],
    places: list[str],
    zone_province: dict[str, str],
    delays: pd.DataFrame | None = None,
) -> None:
    """Write a tidy hub data folder from daily releases.

    `daily` and `zones` hold `vintage`, `cutoff`, `location`, `onset_date`
    and `total`; the weekly target and zone tables are their epiweek sums
    at each of `origins` (the target's vintages), every week from the
    first to the cut-off's, zeros included, for `places`.
    """
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(src / "locations.csv", out / "locations.csv")
    names = pd.read_csv(src / "locations.csv").set_index("location")[
        "location_name"
    ]
    cut = daily.drop_duplicates("vintage").set_index("vintage")["cutoff"]
    first = saturday(pd.Series([daily["onset_date"].min()])).iloc[0]
    rows = []
    for v in origins:
        weeks = pd.date_range(
            first, saturday(pd.Series([cut[v]])).iloc[0], freq="7D"
        ).strftime("%Y-%m-%d")
        sub = daily.loc[daily["vintage"] == v]
        wk = sub.groupby(["location", saturday(sub["onset_date"])])[
            "total"
        ].sum()
        grid = pd.MultiIndex.from_product(
            [places, weeks], names=["location", "target_end_date"]
        )
        tot = wk.reindex(grid, fill_value=0).rename("total").reset_index()
        rows.append(tot.assign(vintage=v, cutoff=cut[v]))
    target = pd.concat(rows, ignore_index=True)
    target = target.assign(
        kind="real",
        location_name=target["location"].map(names),
        observed=target["total"],
        imputed=0,
    )[
        [
            "kind",
            "vintage",
            "location",
            "location_name",
            "target_end_date",
            "observed",
            "imputed",
            "total",
            "cutoff",
        ]
    ]
    target.to_csv(out / "target.csv", index=False)

    def dailies(frame: pd.DataFrame) -> pd.DataFrame:
        f = frame.loc[frame["total"] != 0]
        return f.assign(observed=f["total"], imputed=0)[
            [
                "vintage",
                "location",
                "onset_date",
                "observed",
                "imputed",
                "cutoff",
            ]
        ]

    dailies(daily).to_csv(out / "daily.csv", index=False)
    dailies(zones).to_csv(out / "daily_zones.csv.gz", index=False)
    zw = zones.loc[zones["vintage"].isin(origins)]
    hz = (
        zw.groupby(
            ["vintage", "cutoff", "location", saturday(zw["onset_date"])]
        )["total"]
        .sum()
        .rename_axis(["vintage", "cutoff", "location", "target_end_date"])
        .reset_index()
    )
    hz = hz.loc[hz["total"] != 0]
    hz = hz.assign(
        location_name=hz["location"],
        province=hz["location"].map(zone_province),
    )[
        [
            "vintage",
            "location",
            "location_name",
            "province",
            "target_end_date",
            "total",
            "cutoff",
        ]
    ]
    hz.to_csv(out / "health_zones.csv.gz", index=False)
    if delays is None:
        delays = pd.read_csv(src / "delays.csv.gz")
    delays.to_csv(out / "delays.csv.gz", index=False)
    (out / "hub-config").mkdir(exist_ok=True)
    tasks = json.loads((src / "hub-config" / "tasks.json").read_text())
    codes = sorted(set(places) | set(zone_province))
    for rnd in tasks["rounds"]:
        for task in rnd["model_tasks"]:
            loc = task["task_ids"].get("location")
            if loc is not None:
                loc["required"] = None
                loc["optional"] = codes
    (out / "hub-config" / "tasks.json").write_text(json.dumps(tasks))


# ------------------------------------------------------------- BDBV side


def read_bdbv(src: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the hub data's daily releases and zone daily releases."""
    out = []
    for name in ("daily.csv", "daily_zones.csv.gz"):
        f = pd.read_csv(src / name)
        out.append(
            f.assign(total=f["observed"] + f["imputed"])[
                ["vintage", "cutoff", "location", "onset_date", "total"]
            ]
        )
    return out[0], out[1]


def late_provinces(daily: pd.DataFrame, origin: str) -> list[str]:
    """Provinces with no case in the releases before `origin`."""
    prov = daily.loc[daily["location"].str.count("-") == 1]
    early = prov.loc[(prov["vintage"] < origin) & (prov["total"] > 0)]
    return sorted(set(prov["location"]) - set(early["location"]))


def flatten_weekdays(frame: pd.DataFrame) -> pd.DataFrame:
    """Spread each release's epiweek counts evenly over the week's days.

    The days are those of the week on or before the cut-off, so each
    epiweek total, and the hub target, is unchanged.
    """
    f = frame.assign(week=saturday(frame["onset_date"]))
    wk = f.groupby(["vintage", "cutoff", "location", "week"], as_index=False)[
        "total"
    ].sum()
    parts = []
    for off in range(7):
        day = (
            pd.to_datetime(wk["week"]) - pd.Timedelta(days=6 - off)
        ).dt.strftime("%Y-%m-%d")
        parts.append(wk.assign(onset_date=day))
    long = pd.concat(parts, ignore_index=True)
    long = long.loc[long["onset_date"] <= long["cutoff"]]
    n = long.groupby(["vintage", "location", "week"])["onset_date"].transform(
        "size"
    )
    long["total"] = long["total"] / n
    return long[["vintage", "cutoff", "location", "onset_date", "total"]]


def cap_delay(frame: pd.DataFrame, d: int = D) -> pd.DataFrame:
    """Drop reports more than `d` days after onset.

    Each count at a release whose cut-off is more than `d` days after the
    onset is the count at the last release within `d` days, or zero.
    """
    f = frame.copy()
    cut = f.drop_duplicates("vintage").sort_values("cutoff")[
        ["vintage", "cutoff"]
    ]
    wide = f.pivot_table(
        index=["location", "onset_date"],
        columns="vintage",
        values="total",
        fill_value=0,
    ).reindex(columns=cut["vintage"], fill_value=0)
    onset = pd.to_datetime(wide.index.get_level_values("onset_date"))
    cuts = pd.to_datetime(cut["cutoff"]).to_numpy()
    ok = (cuts[None, :] - onset.to_numpy()[:, None]) <= np.timedelta64(d, "D")
    vals = wide.to_numpy().astype(float)
    # Carry the last within-d count forward over later releases.
    last = np.zeros(len(vals))
    for j in range(vals.shape[1]):
        last = np.where(ok[:, j], vals[:, j], last)
        vals[:, j] = last
    wide[:] = vals
    long = wide.reset_index().melt(
        id_vars=["location", "onset_date"], value_name="total"
    )
    long = long.merge(cut, on="vintage")
    long = long.loc[long["onset_date"] <= long["cutoff"]]
    return long[["vintage", "cutoff", "location", "onset_date", "total"]]


def interpolate_releases(frame: pd.DataFrame) -> pd.DataFrame:
    """Add a release each day between releases, linear in each count.

    Each is labelled by its cut-off, after the label of the release
    before it.
    """
    cut = frame.drop_duplicates("vintage").sort_values("cutoff")
    vint, cuts = cut["vintage"].tolist(), pd.to_datetime(cut["cutoff"]).tolist()
    wide = frame.pivot_table(
        index=["location", "onset_date"],
        columns="vintage",
        values="total",
        fill_value=0,
    ).reindex(columns=vint, fill_value=0)
    onset = pd.to_datetime(wide.index.get_level_values("onset_date"))
    out = [frame]
    for j in range(1, len(vint)):
        a, b = wide[vint[j - 1]].to_numpy(), wide[vint[j]].to_numpy()
        span = (cuts[j] - cuts[j - 1]).days
        for k in range(1, span):
            c = cuts[j - 1] + pd.Timedelta(days=k)
            val = a + (b - a) * k / span
            keep = (onset <= c) & (val != 0)
            iso = c.strftime("%Y-%m-%d")
            # A task takes the releases labelled up to its own, so none
            # may sort before the release it follows.
            if iso <= vint[j - 1]:
                continue
            out.append(
                pd.DataFrame(
                    {
                        "vintage": iso,
                        "cutoff": iso,
                        "location": wide.index.get_level_values("location")[
                            keep
                        ],
                        "onset_date": wide.index.get_level_values("onset_date")[
                            keep
                        ],
                        "total": val[keep],
                    }
                )
            )
    return pd.concat(out, ignore_index=True)


BDBV_VARIANTS = (
    "as_is",
    "rebuilt",
    "no_zones",
    "no_late_provinces",
    "flat_weekdays",
    "cap_d",
    "daily_releases",
    "weekly_source",
    "all",
    "all_but_releases",
    "no_first_release",
    "releases_cap",
    "releases_cap_late",
)


def bdbv_variant(name: str, src: Path, out: Path) -> None:
    """Write the hub data `src` with the change `name` to `out`."""
    if name == "as_is":
        shutil.copytree(src, out, dirs_exist_ok=True)
        return
    daily, zones = read_bdbv(src)
    target = pd.read_csv(src / "target.csv")
    origins = sorted(target["vintage"].unique())
    places = sorted(target["location"].unique())
    hz = pd.read_csv(src / "health_zones.csv.gz")
    zp = dict(zip(hz["location"], hz["province"], strict=False))
    zp.update({z: z.rsplit("-", 1)[0] for z in zones["location"].unique()})
    if name == "no_first_release":
        # The first release (2026-08-18), 50 days before the next, left
        # out, so the releases have no long gap.
        shutil.copytree(src, out, dirs_exist_ok=True)
        for f in ("target.csv", "daily.csv", "daily_zones.csv.gz"):
            t = pd.read_csv(src / f)
            t.loc[t["vintage"] != origins[0]].to_csv(out / f, index=False)
        hz.loc[hz["vintage"] != origins[0]].to_csv(
            out / "health_zones.csv.gz", index=False
        )
        return
    if name == "weekly_source":
        shutil.copytree(src, out, dirs_exist_ok=True)
        for f in ("daily.csv", "daily_zones.csv.gz"):
            pd.read_csv(src / f).head(0).to_csv(out / f, index=False)
        return
    steps = {
        "rebuilt": [],
        "no_zones": ["no_zones"],
        "no_late_provinces": ["no_late"],
        "flat_weekdays": ["flat"],
        "cap_d": ["cap"],
        "daily_releases": ["interp"],
        "all": ["no_late", "no_zones", "cap", "flat", "interp"],
        "all_but_releases": ["no_late", "no_zones", "cap", "flat"],
        "releases_cap": ["no_zones", "cap", "interp"],
        "releases_cap_late": ["no_late", "no_zones", "cap", "interp"],
    }[name]
    for step in steps:
        if step == "no_zones":
            zones = zones.head(0)
        elif step == "no_late":
            # Zones first reported after the first release leave their
            # province's and the country's counts, then provinces so.
            late_z = sorted(
                set(zones["location"])
                - set(
                    zones.loc[
                        (zones["vintage"] < origins[1]) & (zones["total"] > 0),
                        "location",
                    ]
                )
            )
            print(f"late zones: {len(late_z)} of {zones['location'].nunique()}")
            lz = zones.loc[zones["location"].isin(late_z)]
            lz = lz.assign(province=lz["location"].str.rsplit("-", n=1).str[0])
            for col, frame in (
                ("province", lz),
                ("country", lz.assign(country="CD")),
            ):
                sub = frame.groupby(
                    ["vintage", "cutoff", col, "onset_date"], as_index=False
                )["total"].sum()
                sub = sub.rename(columns={col: "location", "total": "minus"})
                daily = daily.merge(
                    sub,
                    on=["vintage", "cutoff", "location", "onset_date"],
                    how="left",
                )
                daily["total"] = (
                    daily["total"] - daily["minus"].fillna(0)
                ).clip(lower=0)
                daily = daily.drop(columns="minus")
            zones = zones.loc[~zones["location"].isin(late_z)]
            late = late_provinces(daily, origins[1])
            print(f"late provinces: {late}")
            drop = daily.loc[daily["location"].isin(late)]
            sub = drop.groupby(
                ["vintage", "cutoff", "onset_date"], as_index=False
            )["total"].sum()
            nat = daily.loc[daily["location"] == "CD"].merge(
                sub, on=["vintage", "cutoff", "onset_date"], how="left"
            )
            nat["total"] = nat["total_x"] - nat["total_y"].fillna(0)
            daily = pd.concat(
                [
                    daily.loc[
                        (daily["location"] != "CD")
                        & ~daily["location"].isin(late)
                    ],
                    nat[
                        ["vintage", "cutoff", "location", "onset_date", "total"]
                    ],
                ],
                ignore_index=True,
            )
            zones = zones.loc[
                ~zones["location"].str.rsplit("-", n=1).str[0].isin(late)
            ]
            places = [p for p in places if p not in late]
        elif step == "flat":
            daily, zones = flatten_weekdays(daily), flatten_weekdays(zones)
        elif step == "cap":
            daily = cap_delay(daily)
            zones = cap_delay(zones) if len(zones) else zones
        elif step == "interp":
            daily = interpolate_releases(daily)
            zones = interpolate_releases(zones) if len(zones) else zones
    write_hub(out, src, daily, zones, origins, places, zp)


# -------------------------------------------------------------- sim side


def pick_outbreaks(
    files: list[Path], k: int, seed: int, lead: int
) -> pd.DataFrame:
    """Return `k` outbreaks' reports, each with two or more leaves.

    Each is active (a peak of at least 20 cases a week) and starts at least
    `lead` days before its peak week; drawn with `seed`.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for f in files:
        d = pd.read_csv(f)
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["ref"] = pd.to_datetime(d["reference_date"])
    d["rep"] = pd.to_datetime(d["report_date"])
    ok = []
    for ob, g in d.groupby("outbreak"):
        if g["location"].nunique() < 2:
            continue
        wk = g.groupby(g["ref"].dt.to_period("W"))["count"].sum()
        peak = wk.idxmax().start_time
        if wk.max() < 20 or (peak - g["ref"].min()).days < lead:
            continue
        ok.append((ob, peak))
    chosen = rng.choice(len(ok), size=min(k, len(ok)), replace=False)
    out = []
    for i, idx in enumerate(sorted(chosen)):
        ob, peak = ok[idx]
        g = d.loc[d["outbreak"] == ob].copy()
        g["province"] = PROVINCES[i]
        leaves = sorted(g["location"].unique())
        g["zone"] = g["location"].map(
            {x: f"{PROVINCES[i]}-{j + 1:02d}" for j, x in enumerate(leaves)}
        )
        g["peak"] = peak
        out.append(g)
    print("outbreaks:", [g["outbreak"].iloc[0] for g in out])
    return pd.concat(out, ignore_index=True)


def thin(
    count: np.ndarray, p: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Binomially thin signed counts, additions and removals alike."""
    pos = rng.binomial(np.clip(count, 0, None), p)
    neg = rng.binomial(np.clip(-count, 0, None), p)
    return pos - neg


DELAY_CAP = 200
"""Most cases of one province and onset week in the simulated delay line."""


def delay_line(
    r: pd.DataFrame, vintage: str, cap: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Return the delay line at `vintage`, as the hub's `delays.csv.gz`.

    One row per case reported by `vintage`, with its province, onset week
    and onset-to-report delay in days, as the hub publishes it. A
    province and onset week with more than `cap` cases gives `cap` of
    them, drawn in proportion to the counts, so the line stays small at
    simulated counts. `r` holds `province`, `ref`, `rep` and `count`;
    removals (negative counts) are not cases.
    """
    rv = r.loc[(r["rep"] <= pd.Timestamp(vintage)) & (r["count"] > 0)]
    rv = rv.assign(
        week=saturday(rv["ref"]).to_numpy(),
        delay=(rv["rep"] - rv["ref"]).dt.days,
    )
    parts = []
    for (prov, week), g in rv.groupby(["province", "week"]):
        count = g["count"].to_numpy(dtype=int)
        total = int(count.sum())
        if total <= cap:
            delay = np.repeat(g["delay"].to_numpy(), count)
        else:
            delay = rng.choice(g["delay"].to_numpy(), size=cap, p=count / total)
        parts.append(
            pd.DataFrame({"location": prov, "onset_week": week, "delay": delay})
        )
    line = pd.concat(parts, ignore_index=True)
    return line.assign(
        vintage=vintage, location_name=line["location"], cutoff=vintage
    )[["vintage", "location", "location_name", "onset_week", "delay", "cutoff"]]


SIM_VARIANTS = (
    "daily",
    "weekly",
    "few",
    "late_start",
    "small",
    "weekday",
    "long_delays",
    "stops",
    "bdbv_like",
    "short",
    "gap",
    "few_daily",
    "fresh",
)


def sim_variant(
    name: str, files: list[Path], out: Path, src: Path, k: int, seed: int
) -> None:
    """Write held-out outbreaks with the BDBV feature `name` as hub data.

    Each outbreak's dates are shifted so its peak week ends on the anchor
    Saturday; the scored origins are the eight Saturdays from three weeks
    before it, as BDBV's seven scored releases span its recent weeks.
    Releases are daily (as most training blocks), but every seventh day
    in `weekly`, and in `few` once 50 days before the first origin and
    weekly from it (BDBV's releases); `late_start` has a third of the zones
    report nothing before a day near the first origin, then everything
    at once; `small` thins each outbreak to a peak of 60 cases a week;
    `weekday` thins weekend onsets by half; `long_delays` delays 30% of
    reports by 14 to 35 more days; `stops` has a third of the zones stop
    reporting from a day near the first origin; `bdbv_like` is `few`,
    `late_start`, `small` and `long_delays` together. Three split
    `few`: `short` releases weekly from 49 days before the first origin
    (as many releases, no gap); `gap` weekly throughout but for the 50
    days before the first origin (the gap, a long release history);
    `few_daily` as `few` but daily from the first origin; `fresh`
    weekly from the first origin alone, so its first release backfills.
    """
    rng = np.random.default_rng(seed)
    r = pick_outbreaks(files, k, seed, lead=70)
    anchor = pd.Timestamp("2026-10-03")
    shift = anchor - (r["peak"] + pd.Timedelta(days=5))
    r["ref"] = r["ref"] + shift
    r["rep"] = r["rep"] + shift
    origins = [anchor + pd.Timedelta(days=7 * i) for i in range(-3, 5)]
    first, last = origins[0], origins[-1]
    feats = {
        "bdbv_like": {"few", "late_start", "small", "long_delays"},
    }.get(name, {name})
    count = r["count"].to_numpy()
    if "small" in feats:
        wk = r.groupby(["outbreak", r["ref"].dt.to_period("W")])["count"].sum()
        peak = wk.groupby(level=0).max()
        p = (60 / peak).clip(upper=1)
        count = thin(count, r["outbreak"].map(p).to_numpy(), rng)
    if "weekday" in feats:
        p = np.where(r["ref"].dt.day_of_week.to_numpy() >= 5, 0.5, 1.0)
        count = thin(count, p, rng)
    if "long_delays" in feats:
        moved = thin(count, np.full(len(count), 0.3), rng)
        extra = rng.integers(14, 36, size=len(count))
        later = r.assign(
            count=moved, rep=r["rep"] + pd.to_timedelta(extra, "D")
        )
        r = pd.concat([r.assign(count=count - moved), later], ignore_index=True)
        count = r["count"].to_numpy()
    r["count"] = count
    zones_all = sorted(r["zone"].unique())
    if "late_start" in feats or "stops" in feats:
        chosen = rng.choice(
            len(zones_all), size=max(1, len(zones_all) // 3), replace=False
        )
        day = {
            zones_all[i]: first + pd.Timedelta(days=int(rng.integers(-21, 22)))
            for i in chosen
        }
        when = r["zone"].map(day)
        has = when.notna()
        if "late_start" in feats:
            r.loc[has, "rep"] = r.loc[has, "rep"].where(
                r.loc[has, "rep"] >= when[has], when[has]
            )
        if "stops" in feats:
            r = r.loc[~(has & (r["rep"] >= when))]
    r = r.loc[r["count"] != 0]
    start = r["rep"].min()
    weekly = list(
        pd.date_range(last - pd.Timedelta(days=7 * 40), last, freq="7D")
    )
    if "few" in feats:
        rel = [first - pd.Timedelta(days=50), *origins]
    elif "few_daily" in feats:
        rel = [
            first - pd.Timedelta(days=50),
            *pd.date_range(first, last, freq="D"),
        ]
    elif "fresh" in feats:
        rel = list(origins)
    elif "short" in feats:
        rel = [c for c in weekly if c >= first - pd.Timedelta(days=49)]
    elif "gap" in feats:
        rel = [
            c
            for c in weekly
            if c <= first - pd.Timedelta(days=50) or c >= first
        ]
        rel = [c for c in rel if c >= start]
    elif "weekly" in feats:
        rel = list(
            pd.date_range(last - pd.Timedelta(days=7 * 40), last, freq="7D")
        )
        rel = [c for c in rel if c >= start]
    else:
        rel = list(pd.date_range(start, last, freq="D"))
    truth = last + pd.Timedelta(days=150)
    rel = sorted(set(rel) | set(origins) | {truth})
    r = r.loc[r["ref"] <= last + pd.Timedelta(days=35)]

    def releases(key: str) -> pd.DataFrame:
        parts = []
        g = r.groupby([key, "ref", "rep"])["count"].sum().reset_index()
        for c in rel:
            s = g.loc[g["rep"] <= c].groupby([key, "ref"])["count"].sum()
            s = s.clip(lower=0)
            s = s.loc[(s > 0) & (s.index.get_level_values("ref") <= c)]
            iso = c.strftime("%Y-%m-%d")
            parts.append(
                pd.DataFrame(
                    {
                        "vintage": iso,
                        "cutoff": iso,
                        "location": s.index.get_level_values(key),
                        "onset_date": s.index.get_level_values("ref").strftime(
                            "%Y-%m-%d"
                        ),
                        "total": s.to_numpy(),
                    }
                )
            )
        return pd.concat(parts, ignore_index=True)

    prov = releases("province")
    zones = releases("zone")
    nat = prov.groupby(["vintage", "cutoff", "onset_date"], as_index=False)[
        "total"
    ].sum()
    daily = pd.concat([prov, nat.assign(location="CD")], ignore_index=True)
    targets = [c.strftime("%Y-%m-%d") for c in [*origins, truth]]
    places = ["CD", *sorted(r["province"].unique())]
    zp = dict(zip(r["zone"], r["province"], strict=False))
    delays = pd.concat(
        [delay_line(r, v, DELAY_CAP, rng) for v in targets], ignore_index=True
    )
    write_hub(out, src, daily, zones, targets, places, zp, delays)
    keep = daily.loc[daily["vintage"].isin(targets)]
    info = {
        "variant": name,
        "outbreaks": sorted(r["outbreak"].unique().tolist()),
        "releases": len(rel) - 1,
        "national_weekly_peak": float(
            keep.loc[
                (keep["location"] == "CD") & (keep["vintage"] == targets[-1])
            ]
            .groupby(saturday(keep["onset_date"]))["total"]
            .sum()
            .max()
        ),
    }
    (out / "variant.json").write_text(json.dumps(info, indent=1))


# --------------------------------------------------------------- summary


def summary(folder: Path, out: Path, reference: str) -> None:
    """Write the check's skill and calibration tables of one variant."""
    from bdbvhub.reporting import ftcheck_report as fr

    parts = sorted(
        p
        for p in (folder / "forecasts").iterdir()
        if (p / "release-samples.csv.gz").exists()
    )
    tables = fr.skill_tables(parts, folder / "data", reference)
    cal = fr.calibration(tables["quantiles"])
    res = {
        "overall": tables["overall"].to_dict("records"),
        "by_weeks_back": tables["by_weeks_back"].to_dict("records"),
        "calibration": cal.to_dict("records"),
    }
    out.write_text(json.dumps(res, indent=1, default=float))


def backtest_config(
    base: Path, out: Path, samples: int, max_delay: int
) -> None:
    """Write the backtest config `base` with `samples` draws and `max_delay`.

    Only the backtest's draws (`[samples] backtest`) and the longest delay
    (`[forecast] max_delay`) change.
    """
    text = base.read_text()
    text, n1 = re.subn(
        r"(?m)^max_delay = \d+$", f"max_delay = {max_delay}", text
    )
    text, n2 = re.subn(r"(?m)^backtest = \d+$", f"backtest = {samples}", text)
    if (n1, n2) != (1, 1):
        raise ValueError(f"{base} has no single max_delay and backtest")
    conf = tomllib.loads(text)
    assert conf["forecast"]["max_delay"] == max_delay
    assert conf["samples"]["backtest"] == samples
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)


def main() -> None:
    """Command line."""
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bdbv")
    b.add_argument("variant", choices=BDBV_VARIANTS)
    b.add_argument("--src", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("sim")
    s.add_argument("variant", choices=SIM_VARIANTS)
    s.add_argument("--sims", type=Path, nargs="+", required=True)
    s.add_argument("--src", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--k", type=int, default=6)
    s.add_argument("--seed", type=int, default=1)
    m = sub.add_parser("summary")
    m.add_argument("folder", type=Path)
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--reference", default="randomwalknowcast")
    c = sub.add_parser("config")
    c.add_argument("--base", type=Path, required=True)
    c.add_argument("--samples", type=int, required=True)
    c.add_argument("--max-delay", type=int, required=True)
    c.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "bdbv":
        bdbv_variant(a.variant, a.src, a.out)
    elif a.cmd == "sim":
        sim_variant(a.variant, a.sims, a.out, a.src, a.k, a.seed)
    elif a.cmd == "config":
        backtest_config(a.base, a.out, a.samples, a.max_delay)
    else:
        summary(a.folder, a.out, a.reference)


if __name__ == "__main__":
    main()
