# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""How each coupling of quantile forecasts into trajectories scores.

The evidence for `nowcastutils.trajectories`: every model's quantile
forecasts at each origin are joined into trajectories by each coupling
(`couple`), and the trajectories are scored on what was reported. Each
cell keeps its marginal, so the cell scores (WIS) are the same for every
coupling; only scores of several cells together differ:

- the energy score of each origin's cells, on counts and on log counts;
- the CRPS and 90% coverage of each place's sum over the nowcast steps
  (horizon <= 0), the sums a hub's weekly totals are made of.

The gaussian coupling at an origin is fitted (`fit_coupling`) from normal
scores (`pit_scores`) of the model's forecasts at the origins of the
`fit_days` before it, on the cells whose truth was reported by then
(`truth_vintage`), so it uses nothing a forecaster at that origin could
not have known. `summary` gives each coupling's mean score as a ratio to
`independent`, with a paired bootstrap interval over origins.

The Schaake shuffle at an origin orders the draws as windows of the
truth reported by then (`history_windows`), consecutive days of all
places, newest first, up to one per draw; with none it is independence.

`online` is the foundation model's step (`_online`): at each origin
`choose_coupling` picks the gaussian over a look-back window or the
Schaake shuffle by the energy score of each one's draws on the model's
forecasts at the earlier origins (`coupling_releases`), on
the truth reported by the origin, aligned by horizon.

Input rows are a German unit's cache (`nowcastgermany.checkpoint`):
`model`, `origin`, `location`, `target_end_date`, `horizon`, `quantile`,
`value`, `truth` and `truth_vintage`.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from nowcasteval.scoring import coverage_sample, crps_sample, energy_sample
from nowcastutils.trajectories import (
    choose_coupling,
    couple,
    coupling_releases,
    fit_coupling,
    history_windows,
    pit_scores,
)

COUPLINGS: Final = (
    "independent",
    "comonotone",
    "gaussian",
    "schaake",
    "online",
)
"""The couplings compared, `independent` first as the reference. `online`
is what the foundation model submits: the coupling `choose_coupling`
picks and fits at each origin."""

FIT_DAYS: Final = 56
"""Days of earlier origins the gaussian coupling is fitted from: 8 weeks,
as the foundation model's fit (`nowcastfoundation.forecaster`)."""

CELL: Final = ["location", "target_end_date"]


def panel_grid(
    rows: pd.DataFrame,
) -> tuple[list[str], list[str], np.ndarray]:
    """Return a model's places, end dates and quantile levels."""
    places = sorted(rows["location"].unique())
    dates = sorted(rows["target_end_date"].unique())
    levels = np.sort(rows["quantile"].unique()).astype(float)
    return places, dates, levels


def _origin(
    rows: pd.DataFrame,
    places: list[str],
    dates: list[str],
    levels: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return one origin's arrays on the model's grid of cells.

    `quantiles` `[places, dates, Q]`, `truth`, `vintage` (the truth's
    report date, as days) and `horizon`, NaN where the origin has no cell.
    """
    p = pd.Index(places).get_indexer(pd.Index(rows["location"]))
    d = pd.Index(dates).get_indexer(pd.Index(rows["target_end_date"]))
    q = np.searchsorted(levels, rows["quantile"].to_numpy(dtype=float))
    shape = (len(places), len(dates))
    quantiles = np.full((*shape, len(levels)), np.nan)
    quantiles[p, d, q] = rows["value"].to_numpy(dtype=float)
    return {"quantiles": quantiles, **_fill(rows, places, dates)}


def _fill(
    rows: pd.DataFrame, places: list[str], dates: list[str]
) -> dict[str, np.ndarray]:
    """Return each cell's truth, horizon and report day on a grid.

    `truth`, `horizon` and `vintage` (the truth's report day) on
    `[places, dates]`, NaN where no row has the cell. A unit has one
    truth per cell, so any row of the cell gives it.
    """
    first = rows.drop_duplicates(CELL)
    p = pd.Index(places).get_indexer(pd.Index(first["location"]))
    d = pd.Index(dates).get_indexer(pd.Index(first["target_end_date"]))
    out = {}
    for name, col in [
        ("truth", first["truth"]),
        ("horizon", first["horizon"]),
        ("vintage", pd.to_datetime(first["truth_vintage"]).map(_days)),
    ]:
        a = np.full((len(places), len(dates)), np.nan)
        a[p, d] = np.asarray(col, dtype=float)
        out[name] = a
    return out


def _days(t: pd.Timestamp) -> float:
    return float(t.toordinal())


def _cells(a: dict[str, np.ndarray]) -> np.ndarray:
    """Cells the origin forecast, with all their quantiles."""
    return np.isfinite(a["quantiles"]).all(axis=-1)


def _fit(
    method: str,
    origin: int,
    origins: list[tuple[int, dict[str, np.ndarray]]],
    levels: np.ndarray,
    fit_days: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Return the coupling's fit at `origin` (a day) from earlier origins."""
    if method != "gaussian":
        return fit_coupling(np.empty((0, 0, 0)), method)
    scores = []
    for day, a in origins:
        if not origin - fit_days <= day < origin:
            continue
        seen = np.where(a["vintage"] <= origin, a["truth"], np.nan)
        ok = _cells(a)
        q = np.where(ok[..., None], a["quantiles"], 0.0)
        z = pit_scores(q, levels, np.where(ok, seen, np.nan), rng)
        scores.append(z)
    if not scores:
        return fit_coupling(np.full((0, 1, 2), np.nan), "gaussian")
    return fit_coupling(np.stack(scores), "gaussian")


def _matured(
    rows: pd.DataFrame, places: list[str], dates: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the truth and its report day on every calendar day.

    `[places, days]` from the first end date to the last, so windows are
    consecutive days, and each of `dates`' position among the days.
    """
    days = pd.date_range(dates[0], dates[-1], freq="D").strftime("%Y-%m-%d")
    cells = _fill(rows, places, list(days))
    at = pd.Index(days).get_indexer(pd.Index(dates))
    return cells["truth"], cells["vintage"], at


def _schaake(
    q: np.ndarray,
    ok: np.ndarray,
    origin: int,
    levels: np.ndarray,
    history: tuple[np.ndarray, np.ndarray, np.ndarray],
    n_samples: int,
    seed: int,
) -> tuple[np.ndarray, int]:
    """Return Schaake draws on the origin's end dates and the windows used.

    `history` is `_matured`'s. The other end dates keep zero draws, as
    `_scores` reads only `ok`.

    Raises
    ------
    ValueError
        If the origin's end dates are not consecutive days.
    """
    truth, day, at = history
    cols = np.flatnonzero(ok.any(axis=0))
    if np.any(np.diff(at[cols]) != 1):
        raise ValueError("an origin's end dates must be consecutive days")
    seen = np.where(day <= origin, truth, np.nan)
    windows = history_windows(seen, len(cols), n_samples)
    method = "schaake" if len(windows) else "independent"
    fit = {"windows": windows} if len(windows) else {}
    draws = np.zeros((n_samples, *ok.shape))
    draws[:, :, cols] = couple(q[:, cols], levels, method, fit, n_samples, seed)
    return draws, len(windows)


def _by_horizon(
    a: np.ndarray, columns: np.ndarray, fill: float = np.nan
) -> np.ndarray:
    """Return a grid's columns at each horizon, `fill` where it has none."""
    out = np.asarray(np.take(a, np.maximum(columns, 0), axis=1), dtype=float)
    out[:, columns < 0] = fill
    return out


def _online(
    day: int,
    a: dict[str, np.ndarray],
    earlier: list[tuple[int, dict[str, np.ndarray]]],
    column: dict[int, int],
    horizons: np.ndarray,
    levels: np.ndarray,
    history: tuple[np.ndarray, np.ndarray, np.ndarray],
    n_samples: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the online coupling's draws at an origin and its record.

    Each origin's cells are aligned by horizon, `[places, horizons]`:
    the earlier origins' quantiles and their truth reported by `day`, and
    the truth reported by then on the calendar days before this origin's
    first horizon as the Schaake shuffle's history. `choose_coupling`
    picks and fits the coupling; the draws are put back on the grid of
    end dates, zero where the origin has no cell.
    """

    def columns(at: int) -> np.ndarray:
        return np.array([column.get(at + int(h), -1) for h in horizons])

    def aligned(b: dict[str, np.ndarray], at: int) -> tuple[np.ndarray, ...]:
        cols = columns(at)
        ok = _by_horizon(_cells(b).astype(float), cols, 0.0) > 0
        q = _by_horizon(b["quantiles"], cols, 0.0)
        q = np.where(ok[..., None], q, 0.0)
        seen = np.where(b["vintage"] <= day, b["truth"], np.nan)
        return q, np.where(ok, _by_horizon(seen, cols), np.nan), ok

    truth, vintage, _ = history
    first_day = min(column)
    seen = np.where(vintage <= day, truth, np.nan)
    past_q, past_y, starts = [], [], []
    days = [at for at, _ in earlier]
    for at, b in earlier:
        q, y, _ = aligned(b, at)
        past_q.append(q)
        past_y.append(y)
        starts.append(max(0, at + int(horizons[0]) - first_day))
    q, _, ok = aligned(a, day)
    shape = (len(earlier), *q.shape)
    fit = choose_coupling(
        np.asarray(past_q).reshape(shape),
        levels,
        np.asarray(past_y).reshape(shape[:-1]),
        np.asarray(days, dtype=int),
        day,
        seen[:, : max(0, day + int(horizons[0]) - first_day)],
        np.asarray(starts, dtype=int),
        n_samples=n_samples,
        seed=seed,
    )
    drawn = couple(q, levels, fit["method"], fit, n_samples, seed)
    cols = columns(day)
    draws = np.zeros((n_samples, *a["truth"].shape))
    draws[:, :, cols[cols >= 0]] = np.where(ok, drawn, 0.0)[:, :, cols >= 0]
    return draws, {k: v for k, v in fit.items() if k != "windows"}


def _scores(draws: np.ndarray, a: dict[str, np.ndarray]) -> dict[str, float]:
    """Return the joint scores of one origin's trajectories."""
    truth, ok = a["truth"], _cells(a) & np.isfinite(a["truth"])
    x, y = draws[:, ok], truth[ok]
    out = {
        "n_cells": int(ok.sum()),
        "energy": energy_sample(y, x),
        "log_energy": energy_sample(np.log1p(y), np.log1p(np.maximum(x, 0))),
    }
    now = ok & (a["horizon"] <= 0)
    crps, cover = [], []
    for p in range(truth.shape[0]):
        if not now[p].any():
            continue
        total = draws[:, p, now[p]].sum(axis=1)
        obs = float(truth[p, now[p]].sum())
        crps.append(crps_sample(obs, total))
        cover.append(coverage_sample(obs, total, 0.9))
    out["sum_crps"] = float(np.mean(crps)) if crps else np.nan
    out["sum_cover90"] = float(np.mean(cover)) if cover else np.nan
    return out


def coupling_scores(
    rows: pd.DataFrame,
    couplings: Sequence[str] = COUPLINGS,
    n_samples: int = 200,
    seed: int = 1,
    fit_days: int = FIT_DAYS,
) -> pd.DataFrame:
    """Score every model's trajectories by each coupling at each origin.

    Parameters
    ----------
    rows : pandas.DataFrame
        Quantile forecasts with their truth (see the module docstring).
    couplings : sequence of str
        Couplings to compare, `independent` among them.
    n_samples : int
        Trajectories per origin.
    seed : int
        Seed; every coupling at an origin draws from the same seed.
    fit_days : int
        Days of earlier origins the gaussian coupling is fitted from.

    Returns
    -------
    pandas.DataFrame
        One row per `model`, `origin` and `coupling`: `n_cells`, `energy`,
        `log_energy`, `sum_crps`, `sum_cover90`, the gaussian fit's `rho`
        and `s`, the Schaake windows used, `n_windows` (NaN for the
        other couplings), and for `online` the coupling it `chose`.
    """
    out = []
    for model, own in rows.groupby("model", sort=True):
        places, dates, levels = panel_grid(own)
        history = _matured(own, places, dates)
        origins = [
            (
                int(_days(pd.Timestamp(str(o)))),
                _origin(g, places, dates, levels),
            )
            for o, g in own.groupby("origin", sort=True)
        ]
        names = sorted(own["origin"].unique())
        column = {int(_days(pd.Timestamp(d))): i for i, d in enumerate(dates)}
        h = own["horizon"].astype(int)
        horizons = np.arange(h.min(), h.max() + 1)
        by_name = dict(zip(names, origins, strict=True))
        for (day, a), name in zip(origins, names, strict=True):
            ok = _cells(a)
            q = np.where(ok[..., None], a["quantiles"], 0.0)
            for method in couplings:
                n_windows, chose = np.nan, ""
                fit: dict[str, Any] = {}
                if method == "online":
                    earlier = coupling_releases([n for n in names if n <= name])
                    draws, fit = _online(
                        day,
                        a,
                        [by_name[n] for n in earlier if n in by_name],
                        column,
                        horizons,
                        levels,
                        history,
                        n_samples,
                        seed + day,
                    )
                    n_windows = fit.get("n_windows", np.nan)
                    chose = str(fit["method"])
                    if fit.get("window") is not None:
                        chose += f"-{fit['window']}"
                elif method == "schaake":
                    draws, n_windows = _schaake(
                        q, ok, day, levels, history, n_samples, seed + day
                    )
                else:
                    rng = np.random.default_rng([seed, day, 1])
                    fit = _fit(method, day, origins, levels, fit_days, rng)
                    draws = couple(
                        q, levels, method, fit, n_samples, seed + day
                    )
                out.append(
                    {
                        "model": model,
                        "origin": name,
                        "coupling": method,
                        **_scores(draws, a),
                        "rho": fit.get("rho", np.nan),
                        "s": fit.get("s", np.nan),
                        "n_windows": n_windows,
                        "chose": chose,
                    }
                )
    return pd.DataFrame(out)


def summary(
    scores: pd.DataFrame,
    reference: str = "independent",
    n_boot: int = 2000,
    seed: int = 1,
) -> pd.DataFrame:
    """Return each coupling's mean scores as ratios to the reference.

    Parameters
    ----------
    scores : pandas.DataFrame
        `coupling_scores` output.
    reference : str
        The coupling every other is divided by.
    n_boot : int
        Paired bootstrap resamples of the origins.
    seed : int
        Seed for the resamples.

    Returns
    -------
    pandas.DataFrame
        One row per `model`, `coupling` and `score` (`energy`,
        `log_energy`, `sum_crps`): `ratio` of means and its 90% interval
        `lower` and `upper` from resampling origins; and the mean
        `sum_cover90` as `cover90`.
    """
    rng = np.random.default_rng(seed)
    out = []
    for model, own in scores.groupby("model", sort=True):
        wide = own.pivot(index="origin", columns="coupling")
        n = len(wide)
        pick = rng.integers(0, n, (n_boot, n))
        for method in own["coupling"].unique():
            cover = float(wide[("sum_cover90", method)].mean())
            for score in ("energy", "log_energy", "sum_crps"):
                num = wide[(score, method)].to_numpy()
                den = wide[(score, reference)].to_numpy()
                boot = np.nanmean(num[pick], axis=1) / np.nanmean(
                    den[pick], axis=1
                )
                lower, upper = np.nanquantile(boot, [0.05, 0.95])
                out.append(
                    {
                        "model": model,
                        "coupling": method,
                        "score": score,
                        "ratio": float(np.nanmean(num) / np.nanmean(den)),
                        "lower": float(lower),
                        "upper": float(upper),
                        "cover90": cover,
                        "n_origins": n,
                    }
                )
    return pd.DataFrame(out)


def main(argv: Sequence[str]) -> None:
    """`python -m nowcasteval.coupling <unit dir> <scores> <summary>`."""
    if len(argv) != 3:
        raise SystemExit(__doc__)
    unit, scores_out, summary_out = (Path(a) for a in argv)
    dtype = {"origin": str, "target_end_date": str, "truth_vintage": str}
    files = sorted(unit.glob("*.csv.gz"))
    if not files:
        raise SystemExit(f"no cached rows in {unit}")
    rows = pd.concat([pd.read_csv(f, dtype=dtype) for f in files])
    # The unit's forecast days (horizon > 0) are for the page: the
    # coupling is scored on the hub's nowcast cells.
    rows = rows.loc[rows["horizon"] <= 0].dropna(subset=["truth"])
    # The daily counts are another view: the coupling is on the 7-day sums.
    if "target" in rows:
        rows = rows.loc[rows["target"].fillna("7day") == "7day"]
    scores = coupling_scores(rows)
    scores_out.parent.mkdir(parents=True, exist_ok=True)
    scores.round(6).to_csv(scores_out, index=False)
    summary(scores).round(4).to_csv(summary_out, index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
