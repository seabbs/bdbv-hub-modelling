# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""How much pairing the walk models' draws matters on the German benchmark.

The German study of the sensitivity page: `nowcasteval.pairing` gives the
variants and their scores, and this forecasts the German development
half with them (`germany_sensitivity`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nowcasteval.nowcast import rolling_truth
from nowcasteval.pairing import (
    MODELS,
    RHO_COLUMNS,
    SCORE_COLUMNS,
    variant_forecasts,
    variant_scores,
)
from nowcastgermany import evaluation as de
from nowcastgermany.data import (
    AGE_GROUPS,
    hub_forecasts,
    hub_target,
    hub_triangle,
    nowcast_tasks,
    within_delay,
)


def germany_sensitivity(
    data: str,
    config_path: str = "config/germany.toml",
    dates: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    r"""Score both pairings on the German development half.

    Each nowcast date's states tasks (with Germany their sum) and age
    groups tasks, one pair per weekday offset, are forecast as the German
    benchmark does (`data.germany.nowcast_tasks`, `hub_forecasts`). They
    give the 7-day sums ending every day from 0 to 28 days back. The truth
    is the sum reported the config's truth delay after it ends.

    Parameters
    ----------
    data : str
        The German data directory.
    config_path : str
        The German config.
    dates : list of str, optional
        Only these nowcast dates (a block of a chunked run).

    Returns
    -------
    pandas.DataFrame
        `variant_scores` rows of every model, variant, date and level.
    pandas.DataFrame
        The rolling $\rho_p$ of each model and task (`RHO_COLUMNS`): its
        copula record, with `origin` the nowcast date and `level` its
        places (`states` or `ages`).
    """
    cfg = de.load_germany_config(config_path)
    triangle = hub_triangle(Path(data) / Path(cfg.triangle).name)
    run = de.nowcast_dates(cfg) if dates is None else list(dates)
    levels = dict.fromkeys(AGE_GROUPS, "ages")
    frames, fits = [], []
    # The models see the counts the truth holds (`within_delay`).
    seen = within_delay(triangle, cfg.truth_delay)
    for date in run:
        for task in nowcast_tasks(
            seen,
            date,
            cfg.n_samples,
            cfg.seed,
        ):
            level = (
                "ages" if set(task.locations) & set(AGE_GROUPS) else "states"
            )
            for name in MODELS:
                drawn, fit = variant_forecasts(name, task)
                if fit is not None:
                    fits.append({**fit, "origin": date, "level": level})
                if not drawn:
                    continue
                moved = {
                    label: hub_forecasts(
                        s.assign(model=name, as_of=np.nan), task, date
                    )
                    for label, s in drawn.items()
                }
                ends = sorted(
                    set(next(iter(moved.values()))["target_end_date"])
                )
                truth = rolling_truth(
                    hub_target(triangle, ends), cfg.truth_delay
                )
                for label, s in moved.items():
                    frames.append(
                        variant_scores(
                            s,
                            truth,
                            study="germany",
                            model=name,
                            variant=label,
                            origin=date,
                            levels={
                                **levels,
                                **{
                                    p: "states"
                                    for p in set(s["location"])
                                    if p not in levels and p != "DE"
                                },
                            },
                        )
                    )
    rho = pd.DataFrame(fits, columns=list(RHO_COLUMNS))
    if not frames:
        return pd.DataFrame(columns=list(SCORE_COLUMNS)), rho
    out = pd.concat(frames, ignore_index=True)
    # Germany's provinces are its states.
    out["level"] = out["level"].replace({"provinces": "states"})
    return out, rho
