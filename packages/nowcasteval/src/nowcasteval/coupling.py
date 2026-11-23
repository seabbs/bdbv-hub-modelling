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
truth reported by then (`history_windows`): every stretch of the
origin's end dates, of all places, whose truth had matured, newest
first, up to one per draw. With none it is independence. The truth is
the unit's own, so windows exist only from the truth's delay after the
unit's first origin (40 days in Germany).

Input rows are a German unit's cache (`nowcastfoundation.checkpoint`):
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
    couple,
    fit_coupling,
    history_windows,
    pit_scores,
)

COUPLINGS: Final = ("independent", "comonotone", "gaussian", "schaake")
"""The couplings compared, `independent` first as the reference."""

FIT_DAYS: Final = 56
"""Days of earlier origins the gaussian coupling is fitted from: 8 weeks,
as the foundation model's fit (`nowcastfoundation.forecaster`)."""

CELL: Final = ["location", "target_end_date"]


def _grid(
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
    out: dict[str, np.ndarray] = {"quantiles": quantiles}
    first = rows.drop_duplicates(CELL)
    p = pd.Index(places).get_indexer(pd.Index(first["location"]))
    d = pd.Index(dates).get_indexer(pd.Index(first["target_end_date"]))
    for name, col in [
        ("truth", first["truth"]),
        ("horizon", first["horizon"]),
        ("vintage", pd.to_datetime(first["truth_vintage"]).map(_days)),
    ]:
        a = np.full(shape, np.nan)
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
) -> tuple[np.ndarray, np.ndarray]:
    """Return each cell's truth and its report day `[places, dates]`."""
    first = rows.drop_duplicates(CELL)
    p = pd.Index(places).get_indexer(pd.Index(first["location"]))
    d = pd.Index(dates).get_indexer(pd.Index(first["target_end_date"]))
    truth = np.full((len(places), len(dates)), np.nan)
    day = np.full_like(truth, np.inf)
    truth[p, d] = first["truth"].to_numpy(dtype=float)
    day[p, d] = pd.to_datetime(first["truth_vintage"]).map(_days)
    return truth, day


def _schaake(
    q: np.ndarray,
    ok: np.ndarray,
    origin: int,
    levels: np.ndarray,
    history: tuple[np.ndarray, np.ndarray],
    n_samples: int,
    seed: int,
) -> tuple[np.ndarray, int]:
    """Return Schaake draws on the origin's end dates and the windows used.

    The other end dates keep zero draws, as `_scores` reads only `ok`.
    """
    truth, day = history
    cols = np.flatnonzero(ok.any(axis=0))
    seen = np.where(day <= origin, truth, np.nan)
    windows = history_windows(seen, len(cols), n_samples)
    draws = np.zeros((n_samples, *ok.shape))
    if len(windows):
        fit, method = {"windows": windows}, "schaake"
    else:
        fit, method = {}, "independent"
    draws[:, :, cols] = couple(q[:, cols], levels, method, fit, n_samples, seed)
    return draws, len(windows)


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
        and `s`, and the Schaake windows used, `n_windows` (NaN for the
        other couplings).
    """
    out = []
    for model, own in rows.groupby("model", sort=True):
        places, dates, levels = _grid(own)
        history = _matured(own, places, dates)
        origins = [
            (
                int(_days(pd.Timestamp(str(o)))),
                _origin(g, places, dates, levels),
            )
            for o, g in own.groupby("origin", sort=True)
        ]
        names = sorted(own["origin"].unique())
        for (day, a), name in zip(origins, names, strict=True):
            ok = _cells(a)
            q = np.where(ok[..., None], a["quantiles"], 0.0)
            for method in couplings:
                n_windows = np.nan
                if method == "schaake":
                    fit: dict[str, Any] = {}
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
