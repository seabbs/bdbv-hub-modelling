"""The BDBV foundation-model check: Chronos-2 on every hub release, scored.

Each model forecasts every real release of the hub data that has a later
one (`nowcasteval.nowcast.forecast_releases`), alone, into its own folder,
so the reference models are made once and a new fine-tuned checkpoint
reruns only itself. The Chronos-2 models, zero-shot and fine-tuned alike,
build the fine-tuned models' layout: one joint block of the country,
every province and the active zones, and one block per province of the
province and all its zones (`nowcastfoundation.forecaster.active_views`,
the rule in `config/finetune.toml`).

The report scores the folders together (`score_releases`) and gives each
model's pairwise relative skill (`nowcasteval.scoring.pairwise_skill`) on
the CRPS and the log CRPS, scaled to the reference, overall and by weeks
back, with each model's forecasts and rounds, the active zones per
release, and the forecasts with their 50% and 90% intervals.

- `release_tasks`: the scored tasks of every release.
- `check_model`: a model as the check runs it.
- `forecast`: one model's forecasts, written to a folder.
- `skill_tables`: the scores and the pairwise tables.
- `write_report`: the page.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import html
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from bdbvhub.models import get_model
from bdbvhub.reporting.explorer import (
    explorer_data,
    explorer_html,
    write_explorer,
)
from bdbvhub.store import store
from nowcastbdbv.data.tidy import load_daily, load_delays, load_target
from nowcastbdbv.data.zones import load_zone_data
from nowcasteval import backtest as bt
from nowcasteval.labels import MODEL_STYLES, NEUTRAL, model_label
from nowcasteval.nowcast import (
    ReleaseForecasts,
    forecast_releases,
    resolve_truth_delay,
    score_releases,
)
from nowcasteval.plots import location_names
from nowcasteval.scoring import pairwise_skill
from nowcastfoundation.forecaster import (
    FINETUNED_ENV,
    FoundationModel,
    active_zone_rule,
)
from nowcastutils.base import ForecastModel, ForecastTask

__all__ = [
    "check_model",
    "explorer_quantiles",
    "forecast",
    "release_tasks",
    "skill_tables",
    "write_report",
]

METRICS = ("crps", "log_crps")
"""The scores compared pairwise: the CRPS on counts and on log counts."""

MATCH = ("origin", "target", "truth_vintage", "location", "target_end_date")
"""What makes two models' scores the same forecast."""


def release_tasks(
    data: Path | str, config: Path | str = "config/backtest.toml"
) -> tuple[list[ForecastTask], pd.DataFrame, pd.DataFrame, Any]:
    """Return the scored task of every release, the target, daily, config.

    Parameters
    ----------
    data : pathlib.Path or str
        The tidy hub data (`bdbvhub data`).
    config : pathlib.Path or str
        The backtest config.

    Returns
    -------
    tuple
        The tasks, the target table, the daily onsets and the config.
    """
    conf = bt.load_config(config)
    target, delays, daily = (
        load_target(data),
        load_delays(data),
        load_daily(data),
    )
    tasks = store.origin_tasks(
        target, delays, conf, daily, load_zone_data(data)
    ).values()
    return bt.tasks_for_scoring(list(tasks), conf), target, daily, conf


def check_model(name: str, checkpoint: str | None = None) -> ForecastModel:
    """Return a registered model as the check runs it.

    Parameters
    ----------
    name : str
        A registered model.
    checkpoint : str, optional
        The fine-tuned checkpoint (a settings hash) an `_ft` model loads,
        in place of `config/finetune.toml`'s.

    Returns
    -------
    ForecastModel
        The model; a Chronos-2 model with the active-zone rule.
    """
    if checkpoint:
        os.environ[FINETUNED_ENV] = checkpoint
    model = get_model(name)
    if isinstance(model, FoundationModel):
        cases, weeks, recent = active_zone_rule()
        model = dataclasses.replace(
            model,
            active_zone_cases=cases,
            active_zone_weeks=weeks,
            active_zone_recent_weeks=recent,
        )
    return model


def forecast(
    name: str,
    data: Path | str,
    out: Path | str,
    checkpoint: str | None = None,
) -> ReleaseForecasts:
    """Forecast every release with one model and write its folder.

    Also writes `blocks.csv`, the block sizes of a Chronos-2 model at
    each release (`FoundationModel.block_sizes`).

    Parameters
    ----------
    name : str
        A registered model.
    data : pathlib.Path or str
        The tidy hub data.
    out : pathlib.Path or str
        The folder.
    checkpoint : str, optional
        See `check_model`.

    Returns
    -------
    ReleaseForecasts
        The forecasts.
    """
    tasks, target, _, _ = release_tasks(data)
    model = check_model(name, checkpoint)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if isinstance(model, FoundationModel):
        sizes = [
            model.block_sizes(t).assign(origin=t.vintage)
            for t in tasks
            if t.kind == "real"
        ]
        pd.concat(sizes, ignore_index=True).to_csv(
            out / "blocks.csv", index=False
        )
    run = forecast_releases(tasks, target, [model])
    run.write(out)
    return run


def skill_tables(
    parts: Sequence[Path | str], data: Path | str, reference: str
) -> dict[str, pd.DataFrame]:
    """Score the folders together and return the report's tables.

    Parameters
    ----------
    parts : sequence of pathlib.Path or str
        The models' folders (`forecast`).
    data : pathlib.Path or str
        The tidy hub data.
    reference : str
        The model the relative skills are scaled to.

    Returns
    -------
    dict of str to pandas.DataFrame
        `scores` (complete weeks), `overall` and `by_weeks_back` (pairwise
        skill), `quantiles` and `blocks` (each Chronos-2 model's block
        sizes by release).
    """
    tasks, target, daily, conf = release_tasks(data)
    run = ReleaseForecasts.read([Path(p) for p in parts])
    result = score_releases(
        tasks, daily, target, run, truth_delay=resolve_truth_delay(conf, target)
    )
    scores = result.scores.loc[result.scores["target"] == "complete"]
    overall = pairwise_skill(
        scores, reference, metrics=METRICS, match_on=MATCH, rounds="origin"
    )
    by = pairwise_skill(
        scores,
        reference,
        by=("weeks_back",),
        metrics=METRICS,
        match_on=MATCH,
        rounds="origin",
    )
    blocks = [
        pd.read_csv(Path(p) / "blocks.csv").assign(model=Path(p).name)
        for p in parts
        if (Path(p) / "blocks.csv").exists()
    ]
    return {
        "scores": scores,
        "overall": overall,
        "by_weeks_back": by,
        "quantiles": result.quantiles,
        "blocks": pd.concat(blocks, ignore_index=True)
        if blocks
        else pd.DataFrame(),
        "target": target,
    }


def explorer_quantiles(
    quantiles: pd.DataFrame, names: dict[str, str]
) -> pd.DataFrame:
    """Return the release quantiles as the forecast explorer reads them.

    Complete weeks only, models by their display names, each origin as
    the round, and the 5% quantile (`q5`) as `q05`.

    Parameters
    ----------
    quantiles : pandas.DataFrame
        `ReleaseBacktest.quantiles`.
    names : dict of str to str
        Display name of each model id.

    Returns
    -------
    pandas.DataFrame
        Columns `model`, `ours`, `reference_date`, `location`,
        `target_end_date` and `QUANTILE_COLUMNS`.
    """
    q = quantiles
    if "target" in q:
        q = q.loc[q["target"] == "complete"]
    q = _named(q.rename(columns={"q5": "q05"}), names)
    return q.assign(ours=True, reference_date=q["origin"])


def _table(frame: pd.DataFrame, cols: Sequence[str]) -> str:
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    body = []
    for row in frame[list(cols)].itertuples(index=False):
        cells = []
        for v in row:
            text = f"{v:.3f}" if isinstance(v, float) else str(v)
            cells.append(f"<td>{html.escape(text)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        f"<table><thead><tr>{head}</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def _named(frame: pd.DataFrame, names: dict[str, str]) -> pd.DataFrame:
    return frame.assign(model=frame["model"].map(lambda m: names.get(m, m)))


def write_report(
    tables: dict[str, pd.DataFrame],
    out: Path | str,
    reference: str,
    checkpoint: str,
    scripts: Sequence[Path] = (),
) -> Path:
    """Write the page and its data folder `<stem>-data/`; return the page.

    Parameters
    ----------
    tables : dict of str to pandas.DataFrame
        `skill_tables`.
    out : pathlib.Path or str
        The page.
    reference : str
        The reference model.
    checkpoint : str
        The fine-tuned checkpoint shown.
    scripts : sequence of pathlib.Path
        The explorer's scripts, inlined.
    """
    out = Path(out)
    data = out.with_name(out.stem + "-data")
    data.mkdir(parents=True, exist_ok=True)
    models = list(dict.fromkeys(tables["overall"]["model"]))
    names = {m: model_label(m) for m in models}
    colours = {
        names[m]: (
            MODEL_STYLES.get(m, (NEUTRAL, "solid"))[0],
            MODEL_STYLES.get(m, (NEUTRAL, "solid"))[0],
            MODEL_STYLES.get(m, (NEUTRAL, "solid"))[1],
        )
        for m in models
    }
    target = tables["target"]
    q = explorer_quantiles(tables["quantiles"], names)
    write_explorer(
        data / "explorer.json",
        explorer_data(
            q, target, location_names(target), 4, colours, samples=None
        ),
    )
    overall = _named(tables["overall"], names).sort_values("relative_crps")
    by = _named(tables["by_weeks_back"], names).sort_values(
        ["weeks_back", "relative_crps"]
    )
    blocks = tables["blocks"]
    active = (
        blocks.loc[blocks["part"] == "joint"]
        .groupby(["model", "origin"])["places"]
        .sum()
        .reset_index()
        if len(blocks)
        else pd.DataFrame()
    )
    cols = ["model", "relative_crps", "relative_log_crps", "n", "rounds"]
    js = "\n".join(p.read_text() for p in scripts)
    parts = [
        "<h1>BDBV foundation-model check</h1>",
        f"<p class=mut>Every hub release with a later one, scored on complete "
        f"weeks. Fine-tuned checkpoint {html.escape(checkpoint)}. Built "
        f"{dt.datetime.now(dt.UTC):%Y-%m-%d %H:%M} UTC.</p>",
        "<p>Relative CRPS is pairwise relative skill (as scoringutils): "
        "each pair's mean CRPS ratio on the forecasts both made, a model's "
        "geometric mean over every model, scaled so "
        f"{html.escape(model_label(reference))} is 1; below 1 is better. "
        "The log CRPS is the CRPS of log counts. Zero-shot and fine-tuned "
        "Chronos-2 build the same layout: one joint block of the country, "
        "every province and the active zones, and one block per province "
        "of the province and all its zones.</p>",
        "<h2>Overall</h2>",
        _table(overall, cols),
        "<h2>By weeks back</h2>",
        _table(by, ["weeks_back", *cols]),
        "<h2>Forecasts</h2>",
        "<p class=mut>Medians with their 50% and 90% intervals.</p>",
        explorer_html(f"{data.name}/explorer.json", "Forecast explorer"),
    ]
    if len(active):
        parts += [
            "<h2>Joint block per release</h2>",
            "<p class=mut>Places in the joint block (country, provinces "
            "and active zones).</p>",
            _table(active, ["model", "origin", "places"]),
        ]
    doc = (
        "<!doctype html><html lang=en><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width,initial-scale=1'>"
        "<title>BDBV FM check</title>"
        "<style>body{font-family:system-ui;margin:16px;max-width:1100px}"
        "table{border-collapse:collapse}td,th{padding:2px 8px;"
        "border-bottom:1px solid #ddd;text-align:right}.mut{color:#666}"
        "</style><body>"
        + "\n".join(parts)
        + f"<script>{js}</script></body></html>"
    )
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(doc)
    tmp.rename(out)
    (data / "skill.json").write_text(
        json.dumps(
            {
                "overall": tables["overall"].to_dict("records"),
                "by_weeks_back": tables["by_weeks_back"].to_dict("records"),
            },
            default=str,
        )
    )
    return out
