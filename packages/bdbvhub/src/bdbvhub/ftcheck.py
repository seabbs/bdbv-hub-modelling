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
back, with each model's forecasts and rounds; each model's calibration by
level and weeks back (`calibration`: medians against truth and against
what was already reported, 90% coverage) and the week-to-week roughness
of its medians against the truth's (`roughness`); the active zones per
release; and the forecasts with their 50% and 90% intervals. Only the
hub's weekly targets at its locations are scored.

- `release_tasks`: the scored tasks of every release.
- `check_model`: a model as the check runs it.
- `forecast`: one model's forecasts, written to a folder.
- `skill_tables`: the scores and the pairwise tables.
- `calibration`, `roughness`: the calibration tables.
- `write_report`: the page.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import html
import json
import os
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bdbvhub.models import get_model
from bdbvhub.reporting.page import html_table, standalone, write_page
from bdbvhub.store import store
from nowcastbdbv.data.tidy import load_daily, load_delays, load_target
from nowcastbdbv.data.zones import load_zone_data
from nowcasteval import backtest as bt
from nowcasteval.explorer import (
    explorer_data,
    explorer_html,
    write_explorer,
)
from nowcasteval.labels import model_label
from nowcasteval.nowcast import (
    COMPLETE_WEEKS,
    ReleaseForecasts,
    forecast_releases,
    resolve_truth_delay,
    score_releases,
)
from nowcasteval.plots import comparison_colours, location_names
from nowcasteval.scorechart import write_score_chart
from nowcasteval.scoresummary import summary_html, summary_rows
from nowcasteval.scoring import pairwise_skill
from nowcastfoundation.forecaster import (
    FINETUNED_ENV,
    FoundationModel,
    active_zone_rule,
)
from nowcastutils.base import ForecastModel, ForecastTask

__all__ = [
    "calibration",
    "check_model",
    "explorer_quantiles",
    "forecast",
    "release_tasks",
    "roughness",
    "skill_tables",
    "summary_lines",
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


LEVELS = {0: "national", 1: "province"}
"""Place levels by the number of dashes in the code; deeper are zones."""


def _complete(quantiles: pd.DataFrame) -> pd.DataFrame:
    q = quantiles
    if "target" in q:
        q = q.loc[q["target"] == "complete"]
    level = q["location"].str.count("-").map(LEVELS).fillna("zone")
    return q.assign(level=level).dropna(subset=["truth"])


def calibration(quantiles: pd.DataFrame) -> pd.DataFrame:
    """Return each model's medians against truth and what was reported.

    On complete weeks, by model, level and weeks back: the summed medians
    over the summed truth (`median_over_truth`) and over the summed counts
    already reported at the origin (`median_over_reported`), and the
    share of truths inside the 90% interval (`coverage_90`). A nowcast
    that only repeats what was reported has `median_over_reported` 1.

    Parameters
    ----------
    quantiles : pandas.DataFrame
        `ReleaseBacktest.quantiles`.

    Returns
    -------
    pandas.DataFrame
        Columns `model`, `level`, `weeks_back`, `n` and the three.
    """
    q = _complete(quantiles)
    inside = (q["q5"] <= q["truth"]) & (q["truth"] <= q["q95"])
    grp = q.assign(_in=inside).groupby(["model", "level", "weeks_back"])
    out = grp.size().rename("n").to_frame()
    out["median_over_truth"] = grp["q50"].sum() / grp["truth"].sum()
    reported = grp["reported"].sum()
    # Weeks after the origin have nothing reported: no ratio.
    out["median_over_reported"] = grp["q50"].sum() / reported.where(
        reported > 0
    )
    out["coverage_90"] = grp["_in"].mean()
    return out.reset_index()


def roughness(quantiles: pd.DataFrame) -> pd.DataFrame:
    """Return how much medians and truth move from one week to the next.

    The mean absolute change of `log1p` from each week to the next within
    a forecast (one origin and place), of the medians (`median`) and of
    the truth (`truth`), by model and level, on complete weeks.

    Parameters
    ----------
    quantiles : pandas.DataFrame
        `ReleaseBacktest.quantiles`.

    Returns
    -------
    pandas.DataFrame
        Columns `model`, `level`, `median` and `truth`.
    """
    q = _complete(quantiles).sort_values(
        ["model", "origin", "location", "target_end_date"]
    )
    q = q.drop_duplicates(["model", "origin", "location", "target_end_date"])
    keys = ["model", "origin", "location"]
    steps = q.assign(
        median=np.log1p(q["q50"]).groupby([q[k] for k in keys]).diff().abs(),
        truth=np.log1p(q["truth"]).groupby([q[k] for k in keys]).diff().abs(),
    )
    out = steps.groupby(["model", "level"])[["median", "truth"]].mean()
    return out.reset_index()


def explorer_quantiles(
    quantiles: pd.DataFrame, names: dict[str, str]
) -> pd.DataFrame:
    """Return the release quantiles as the forecast explorer reads them.

    Every forecast week once (complete or not), models by their display
    names, each origin as the round, and the 5% quantile (`q5`) as
    `q05`.

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
    # Every forecast week, complete or not, once: a forecast is scored
    # against several releases, with the same quantiles each time.
    keys = ["model", "origin", "location", "target_end_date"]
    q = quantiles.drop_duplicates(keys)
    q = _named(q.rename(columns={"q5": "q05"}), names)
    return q.assign(ours=True, reference_date=q["origin"])


def _named(frame: pd.DataFrame, names: dict[str, str]) -> pd.DataFrame:
    return frame.assign(model=frame["model"].map(lambda m: names.get(m, m)))


def _by_horizon(frame: pd.DataFrame) -> pd.DataFrame:
    """Return weeks back as the hub's horizon, negative for nowcasts."""
    return frame.assign(horizon=-frame["weeks_back"]).drop(columns="weeks_back")


def _roles(models: Sequence[str]) -> tuple[str, str]:
    """Return the fine-tuned and the zero-shot Chronos-2 model shown."""
    tuned = next((m for m in models if m.endswith("_ft")), "")
    zero = next(
        (m for m in models if m.startswith("fm_") and not m.endswith("_ft")),
        "",
    )
    return tuned, zero


def summary_lines(
    overall: pd.DataFrame,
    cal: pd.DataFrame,
    reference: str,
    checkpoint: str,
    rounds: Sequence[str],
) -> list[str]:
    """Return the page's three-line summary.

    Which checkpoint and rounds; the fine-tuned model's relative CRPS
    against the reference and against zero-shot Chronos-2; and its worst
    calibrated nowcast (lowest 90% coverage at horizon 0 or earlier, at
    the national or province level).
    """
    rel = overall.set_index("model")
    tuned, zero = _roles(list(rel.index))
    lines = [
        f"Fine-tuned checkpoint {checkpoint}, on {len(rounds)} hub rounds "
        f"({rounds[0]} to {rounds[-1]})"
        if rounds
        else f"Fine-tuned checkpoint {checkpoint}, with no scored round yet."
    ]
    if tuned in rel.index:
        nat = rel["relative_crps"].astype(float).to_dict()
        crps = nat[tuned]
        log = rel["relative_log_crps"].astype(float).to_dict()[tuned]
        text = (
            f"Relative CRPS {crps:.2f} (log scale {log:.2f}) against "
            f"{model_label(reference)}, where below 1 is better"
        )
        base = nat.get(zero, np.nan)
        if base > 0:
            ratio = crps / base
            text += f"; {ratio:.2f} times zero-shot Chronos-2's CRPS"
        lines.append(text + ".")
    worst = cal.loc[
        (cal["model"] == tuned)
        & (cal["weeks_back"] >= 0)
        & cal["level"].isin(["national", "province"])
    ].sort_values("coverage_90")
    if len(worst):
        w = worst.iloc[0]
        lines.append(
            f"Weakest nowcast: {w['level']} at horizon "
            f"{-int(w['weeks_back'])}, "
            f"90% intervals cover {w['coverage_90']:.0%} of the truth, "
            f"medians {w['median_over_truth']:.2f} times it."
        )
    return lines


def write_report(
    tables: dict[str, pd.DataFrame],
    out: Path | str,
    reference: str,
    checkpoint: str,
    docs: Path | str = "docs",
) -> Path:
    """Write the page and its data folder `<stem>-data/`; return the page.

    Built from the same parts as the hub rounds and this-round pages
    (`bdbvhub.reporting.page`): a three-line summary; the score summary
    of relative CRPS by horizon (`nowcasteval.scoresummary`, natural and
    log scale); the forecast explorer (`nowcasteval.explorer`) by
    national, province and zone with each round's data and the later
    truth; calibration and roughness; and the full tables, folded.

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
    docs : pathlib.Path or str
        The repository's `docs` folder (stylesheet and scripts).

    Returns
    -------
    pathlib.Path
        The page.
    """
    out = Path(out)
    data = out.with_name(out.stem + "-data")
    if data.exists():
        shutil.rmtree(data)
    data.mkdir(parents=True)
    models = list(dict.fromkeys(tables["overall"]["model"]))
    names = {m: model_label(m) for m in models}
    colours = comparison_colours({names[m]: m for m in models}, {}, [])
    target = tables["target"]
    real = target.loc[target["kind"] == "real"]
    places = location_names(real)
    q = explorer_quantiles(tables["quantiles"], names)
    rounds = sorted(set(q["reference_date"].astype(str)))
    asof = real.loc[real["vintage"].astype(str).isin(rounds)]
    asof = asof.assign(reference_date=asof["vintage"].astype(str))
    write_explorer(
        data / "explorer.json",
        explorer_data(
            q,
            real,
            places,
            COMPLETE_WEEKS,
            colours,
            samples=None,
            asof=asof,
        ),
    )
    srcs = {}
    scores = tables.get("scores")
    if scores is not None and len(scores):
        s = _named(scores, names).assign(horizon=-scores["weeks_back"])
        write_score_chart(
            data / "scores.json",
            summary_rows(
                s,
                [names[m] for m in models],
                colours,
                places,
                baseline=names.get(reference, reference),
            ),
        )
        srcs["complete"] = f"{data.name}/scores.json"
    cal = calibration(tables["quantiles"])
    lines = summary_lines(tables["overall"], cal, reference, checkpoint, rounds)
    skill = {
        "model": "Model",
        "relative_crps": "Relative CRPS",
        "relative_log_crps": "Relative CRPS (log scale)",
        "n": "Forecasts",
        "rounds": "Rounds",
    }
    overall = _named(tables["overall"], names).sort_values("relative_crps")
    by = _by_horizon(_named(tables["by_weeks_back"], names)).sort_values(
        ["horizon", "relative_crps"]
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
    parts = [
        "<h1>Ebola (BDBV): fine-tuned Chronos-2, "
        f"{html.escape(checkpoint)}</h1>",
        "<p>" + "<br>".join(html.escape(line) for line in lines) + "</p>",
        f"<p class=mut>Built {dt.datetime.now(dt.UTC):%Y-%m-%d %H:%M} UTC. "
        "Each hub round forecast from its release; scored on weeks the "
        "latest release reports in full. Cases per week.</p>",
        "<h2>Skill by horizon</h2>",
        "<p class=mut>Pairwise relative CRPS (each pair of models on the "
        "forecasts both made, as scoringutils), scaled so "
        f"{html.escape(model_label(reference))} is 1; below 1 is better. "
        "Horizon in weeks from the round, negative for nowcasts. Pick "
        "the metric for the log scale, the scale and the models.</p>",
        summary_html(srcs, "Skill by horizon", view="horizon")
        if srcs
        else "<p>No scores yet.</p>",
        "<h2>Forecasts</h2>",
        "<p class=mut>Medians with 50% and 90% intervals, against the data "
        "as of each round (line) and the latest release (points; hollow "
        "where not yet complete). Pick the location by national, "
        "province or zone, the rounds and the models.</p>",
        explorer_html(f"{data.name}/explorer.json", "Forecast explorer"),
        "<h2>Calibration</h2>",
        "<p class=mut>National and province nowcasts: summed medians over "
        "the truth and over what was already reported at the round (near 1 "
        "means the nowcast only repeats the reported counts), and the "
        "share of truths inside the 90% interval.</p>",
        html_table(
            _named(
                _by_horizon(
                    cal.loc[
                        cal["level"].isin(["national", "province"])
                        & (cal["weeks_back"] >= 0)
                    ]
                ),
                names,
            ).sort_values(["level", "horizon", "model"]),
            {
                "level": "Level",
                "horizon": "Horizon",
                "model": "Model",
                "median_over_truth": "Median / truth",
                "median_over_reported": "Median / reported",
                "coverage_90": "90% coverage",
                "n": "Forecasts",
            },
        ),
        "<h2>Week-to-week roughness</h2>",
        "<p class=mut>Mean absolute change of log(1 + cases) from one week "
        "to the next within a forecast: the medians against the "
        "truth.</p>",
        html_table(
            _named(roughness(tables["quantiles"]), names),
            {
                "model": "Model",
                "level": "Level",
                "median": "Medians",
                "truth": "Truth",
            },
        ),
        "<h2>Tables</h2>",
        "<details><summary>Relative CRPS over every horizon</summary>",
        html_table(overall, skill),
        "</details>",
        "<details><summary>Relative CRPS by horizon</summary>",
        html_table(by, {"horizon": "Horizon", **skill}),
        "</details>",
        "<details><summary>Calibration at every level and horizon</summary>",
        html_table(
            _named(_by_horizon(cal), names),
            {
                "model": "Model",
                "level": "Level",
                "horizon": "Horizon",
                "median_over_truth": "Median / truth",
                "median_over_reported": "Median / reported",
                "coverage_90": "90% coverage",
                "n": "Forecasts",
            },
        ),
        "</details>",
    ]
    if len(active):
        parts += [
            "<details><summary>Places in the joint block per round</summary>",
            "<p class=mut>The country, every province and the active "
            "zones.</p>",
            html_table(
                _named(active, names),
                {"model": "Model", "origin": "Round", "places": "Places"},
                digits=0,
            ),
            "</details>",
        ]
    (data / "skill.json").write_text(
        json.dumps(
            {
                "overall": tables["overall"].to_dict("records"),
                "by_weeks_back": tables["by_weeks_back"].to_dict("records"),
            },
            default=str,
        )
    )
    return write_page(out, standalone(f"BDBV check {checkpoint}", parts, docs))
