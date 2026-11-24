# Copyright (c) 2026 The bdbv-hub-modelling authors. MIT licence, see LICENSE.
"""Write hub submissions and model metadata.

A submission is a CSV with columns `reference_date`, `forecast_date`,
`target`, `target_end_date`, `location`, `output_type`, `output_type_id`
and `value`.
Every count row (`wk inc confirmed cases`) is one week of one sample
trajectory. A renewal model also gives the optional `wk R_t` target as
quantiles (`rt_table`) for the weeks the round takes (`rt_end_dates`).
`forecast_date` is the day the PR is opened.
A resubmission in the same week keeps the file on the hub's main branch
unchanged and appends rows with one new `forecast_date`.
The round `BDBV 2026 nowcasting and forecasting` takes exactly 100
samples for each `(reference_date, location, target)`, and each sample
index covers the same target weeks within a location.

`submission.validate` checks the written files as `hubValidations` does.
The authoritative check is still `hubValidations` (`task bdbvhub:validate`).
"""

from __future__ import annotations

import datetime as dt
import json
import re
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import pandas as pd
import yaml

from nowcastutils.dates import as_date
from nowcastutils.files import write_csv

__all__ = [
    "HUB_COLUMNS",
    "METADATA_DEFAULTS",
    "OUTPUT_TYPE",
    "ROUND_NAME",
    "RT_TARGET",
    "SUBMISSION_CONFIG",
    "TEAM_ABBR",
    "model_abbr_for",
    "model_id",
    "own_national",
    "render_model_metadata",
    "rt_end_dates",
    "rt_table",
    "run_date",
    "submission_path",
    "to_hub_table",
    "write_model_metadata",
    "write_submission",
    "zones_need_not_sum",
]

HUB_COLUMNS: Final = (
    "reference_date",
    "forecast_date",
    "target",
    "target_end_date",
    "location",
    "output_type",
    "output_type_id",
    "value",
)
"""Columns of a submission file, in the order we write them."""

ROUND_NAME: Final = "BDBV 2026 nowcasting and forecasting"
"""The round this repository submits to."""

OUTPUT_TYPE: Final = "sample"
"""The output type of the count target."""

RT_TARGET: Final = "wk R_t"
"""The hub's optional weekly effective reproduction number target."""

RT_OUTPUT_TYPE: Final = "quantile"
"""The output type of `RT_TARGET`, the only one the hub takes for it."""

TEAM_ABBR: Final = "seabbs"
"""Team abbreviation, of our hub models."""

# The schema allows 16 characters and the README says "less than 15" or
# "less than 16", so keep to 14 to satisfy all three
# (docs/submission/hub-spec.md).
_ID_PART = re.compile(r"^[A-Za-z0-9_]{1,14}$")

NATIONAL: Final = "CD"
"""Location code of the whole country."""

SUBMISSION_CONFIG: Final = Path("packages/bdbvhub/config/hub.toml")
"""Where the submitted model abbreviation is set."""


def model_abbr_for(model: str, path: Path | str = SUBMISSION_CONFIG) -> str:
    """Return the hub abbreviation a model submits under.

    A model with its own entry under `[models]` in
    `packages/bdbvhub/config/hub.toml` is a
    hub model of its own. Every other model submits under the
    `[submission]` abbreviation, one value not derived from a model key,
    so every renewal variant and every fallback submits under one id.

    Parameters
    ----------
    model : str
        The model key, such as `"randomwalknowcast"`.
    path : pathlib.Path or str
        The configuration file.

    Returns
    -------
    str
        The abbreviation, such as `"baseline"`.
    """
    with Path(path).open("rb") as fh:
        config = tomllib.load(fh)
    own = config.get("models", {}).get(model, {})
    return str(own.get("model_abbr") or config["submission"]["model_abbr"])


def model_id(model_abbr: str, team_abbr: str = TEAM_ABBR) -> str:
    """Join a team and model abbreviation into a hub model id.

    Parameters
    ----------
    model_abbr : str
        Model abbreviation, such as `"randomwalk"`.
    team_abbr : str
        Team abbreviation.

    Returns
    -------
    str
        `"<team_abbr>-<model_abbr>"`.

    Raises
    ------
    ValueError
        If either part is not 1 to 14 letters, digits or underscores.
    """
    for part in (team_abbr, model_abbr):
        if not _ID_PART.match(part):
            raise ValueError(
                f"{part!r} must be 1 to 14 letters, digits or underscores."
            )
    return f"{team_abbr}-{model_abbr}"


def to_hub_table(
    samples: pd.DataFrame,
    reference_date: dt.date | str,
    target: str = "wk inc confirmed cases",
    *,
    unassigned: pd.DataFrame | None = None,
    zones: pd.DataFrame | None = None,
    n_samples: int = 100,
    forecast_date: dt.date | str | None = None,
    zones_sum: bool = True,
    national: str = "sum",
) -> pd.DataFrame:
    """Turn forecast samples into the hub's submission table.

    The table meets every guarantee in `docs/submission/hub-spec.md`:
    sample ids `1..n_samples` reused in every location, one row per id for every
    week, the same weeks for every location, whole non-negative counts
    and, by default, the national value as the sum of the provinces for
    each sample, replacing any national rows in `samples`. A model that
    forecasts the country itself (`own_national`, the Chronos-2 models)
    keeps its own national rows with `national="own"`. The hub does not
    require the sum (`docs/submission/hub-spec.md`).

    Parameters
    ----------
    samples : pandas.DataFrame
        Columns `location`, `target_end_date`, `sample` (an integer
        index, one joint draw across weeks and provinces) and `value`.
    reference_date : datetime.date or str
        The round's reference date.
    target : str
        The target name.
    unassigned : pandas.DataFrame, optional
        A modelled component of the national count not assigned to any
        province, with columns `target_end_date`, `sample` and `value`.
        It is added to the national sum. Missing rows count as zero.
    zones : pandas.DataFrame, optional
        Health-zone samples (columns as `samples`), such as from
        `nowcastbaselines.zones.zones_for_model`. They are not added to the
        national value.
    n_samples : int
        Number of samples each location must have.
    forecast_date : datetime.date or str, optional
        The day the PR is opened. The default is today (UTC), see
        `run_date`. The hub requires it to be on or before the
        reference date and within a day of the validation date.
    zones_sum : bool
        Check that the zones sum to the rounded province samples, sample
        by sample, as a split model's must. False for the random walks'
        own zone draws and the foundation models' block zones
        (`zones_need_not_sum`); the hub
        does not require the sum (its `model-output/README.md` allows
        any subset of locations).
    national : {"sum", "own"}
        `"sum"` writes each sample's national value as the sum of its
        provinces, plus `unassigned`. `"own"` writes the national rows in
        `samples` as they are, checked like any location; `unassigned`
        must then be empty.

    Returns
    -------
    pandas.DataFrame
        Columns `HUB_COLUMNS`, sorted by location, sample and week.
        Values are rounded to whole cases before the national sum, so the
        national value is exactly the sum of the written province values.

    Raises
    ------
    ValueError
        If a province or zone does not have sample ids `1..n_samples`, a
        sample lacks a week, locations cover different weeks, or, with
        `zones_sum`, the zones do not sum to their provinces, or, with
        `national="own"`, `samples` has no complete national rows.

    Examples
    --------
    >>> from bdbvhub.submission.hub import (
    ...     model_id, to_hub_table, write_submission
    ... )
    >>> samples = forecasts.query("model == 'randomwalknowcast'")
    >>> table = to_hub_table(samples, task.reference_date)
    >>> write_submission(table, "output/submission", model_id("rwnowcast"))
    """
    if national not in ("sum", "own"):
        raise ValueError(f"national must be 'sum' or 'own', not {national!r}.")
    ref = as_date(reference_date).isoformat()
    fdate = as_date(forecast_date or run_date()).isoformat()
    own = _tidy(samples.loc[samples["location"] == NATIONAL])
    if national == "own":
        if own.empty:
            raise ValueError("No national rows to keep (national='own').")
        if unassigned is not None and not unassigned.empty:
            raise ValueError("An unassigned component needs national='sum'.")
    prov = _tidy(samples.loc[samples["location"] != NATIONAL])
    if prov.empty:
        raise ValueError("No province samples to write.")
    zone = _zone_rows(prov, zones, zones_sum)
    ids = set(range(1, n_samples + 1))
    weeks = sorted(set(prov["target_end_date"]))
    checked = [prov, zone, own] if national == "own" else [prov, zone]
    for loc, rows in pd.concat(checked).groupby("location"):
        if set(rows["sample"]) != ids:
            raise ValueError(
                f"{loc}: sample ids must be 1 to {n_samples}, the same in "
                "every location."
            )
        if sorted(set(rows["target_end_date"])) != weeks:
            raise ValueError(f"{loc}: every location needs the same weeks.")
        if (
            len(rows) != n_samples * len(weeks)
            or rows.duplicated(["sample", "target_end_date"]).any()
        ):
            raise ValueError(
                f"{loc}: every sample needs exactly one row for each week."
            )
    if national == "own":
        both = pd.concat([prov, own, zone], ignore_index=True)
        return _hub_rows(both, ref, fdate, target)
    national_rows = (
        prov.groupby(["target_end_date", "sample"])["value"].sum().reset_index()
    )
    if unassigned is not None and not unassigned.empty:
        extra = (
            unassigned.assign(
                target_end_date=unassigned["target_end_date"].astype(str),
                sample=unassigned["sample"].astype("int64"),
                value=unassigned["value"].clip(lower=0).round(),
            )
            .groupby(["target_end_date", "sample"])["value"]
            .sum()
        )
        national_rows = national_rows.set_index(["target_end_date", "sample"])
        national_rows["value"] = national_rows["value"].add(extra, fill_value=0)
        national_rows = national_rows.reset_index()
    national_rows["location"] = NATIONAL
    both = pd.concat([prov, national_rows, zone], ignore_index=True)
    return _hub_rows(both, ref, fdate, target)


def _tidy(rows: pd.DataFrame) -> pd.DataFrame:
    """Return sample rows with string keys and whole non-negative values."""
    rows = rows.copy()
    rows["target_end_date"] = rows["target_end_date"].astype(str)
    rows["location"] = rows["location"].astype(str)
    rows["sample"] = rows["sample"].astype("int64")
    rows["value"] = rows["value"].clip(lower=0).round()
    return rows


def _hub_rows(
    both: pd.DataFrame, ref: str, fdate: str, target: str
) -> pd.DataFrame:
    """Lay out tidy sample rows as the hub's columns, sorted."""
    out = pd.DataFrame(
        {
            "reference_date": ref,
            "forecast_date": fdate,
            "target": target,
            "target_end_date": both["target_end_date"],
            "location": both["location"],
            "output_type": OUTPUT_TYPE,
            "output_type_id": both["sample"].astype("int64"),
            "value": both["value"].astype("int64"),
        }
    )
    return out.sort_values(
        ["location", "output_type_id", "target_end_date"]
    ).reset_index(drop=True)[list(HUB_COLUMNS)]


def own_national(name: str | None) -> bool:
    """Return whether model `name` forecasts the country itself.

    The Chronos-2 models forecast the country as one series of their
    joint block, so their national draws are written as they are
    (`to_hub_table(..., national="own")`). Every other model's national
    value is the per-sample sum of its provinces.
    """
    if name is None:
        return False
    import typing

    from bdbvhub.models import list_models
    from nowcastfoundation.forecaster import FoundationModel

    specs = {s.name: s for s in list_models()}
    if name not in specs:
        return False
    # Decided from the registered factory, so no model (and no fine-tuned
    # checkpoint) is built: FoundationModel itself, or a builder of one.
    func = getattr(specs[name].factory, "func", specs[name].factory)
    if isinstance(func, type):
        return issubclass(func, FoundationModel)
    return typing.get_type_hints(func).get("return") is FoundationModel


def zones_need_not_sum(name: str | None) -> bool:
    """Return whether model `name`'s zones need not sum to its provinces.

    The random walks forecast each zone from its own data
    (`nowcastbaselines.zones.walks_own_zones`), and the foundation models take
    their zones straight from their block, as they do the provinces and
    the country (`own_national`). Every other model splits its provinces
    across the zones, so its zones must sum to them.
    """
    from nowcastbaselines.zones import walks_own_zones

    return walks_own_zones(name) or own_national(name)


def _zone_rows(
    prov: pd.DataFrame, zones: pd.DataFrame | None, check: bool
) -> pd.DataFrame:
    """Tidy zone samples, and with `check` check they sum to provinces."""
    if zones is None or zones.empty:
        return prov.iloc[:0]
    from nowcastbaselines.zones import check_zone_coherence

    zone = zones[list(prov.columns)].copy()
    zone["target_end_date"] = zone["target_end_date"].astype(str)
    zone["location"] = zone["location"].astype(str)
    zone["sample"] = zone["sample"].astype("int64")
    zone["value"] = zone["value"].clip(lower=0).round()
    gaps = check_zone_coherence(prov, zone) if check else []
    if gaps:
        raise ValueError(f"{len(gaps)} zone sums differ: {gaps[0]}.")
    return zone


def submission_path(
    root: Path | str, reference_date: dt.date | str, mid: str
) -> Path:
    """Return where a submission file goes in a hub checkout.

    Parameters
    ----------
    root : pathlib.Path or str
        Root of the hub checkout (or any directory laid out like one).
    reference_date : datetime.date or str
        The round's reference date.
    mid : str
        The model id, such as `"seabbs-randomwalk"`.

    Returns
    -------
    pathlib.Path
        `root/model-output/<mid>/<reference_date>-<mid>.csv`.
    """
    ref = as_date(reference_date).isoformat()
    return Path(root) / "model-output" / mid / f"{ref}-{mid}.csv"


def run_date(now: dt.datetime | None = None) -> dt.date:
    """Return the date a run opens its PR on, in UTC.

    Parameters
    ----------
    now : datetime.datetime, optional
        The time of the run. Defaults to now.

    Returns
    -------
    datetime.date
        The UTC date.
    """
    moment = now or dt.datetime.now(dt.UTC)
    return moment.astimezone(dt.UTC).date()


def rt_end_dates(
    tasks_json: Path | str, reference_date: str
) -> tuple[str, ...]:
    """Return the weeks the round takes `RT_TARGET` for.

    Parameters
    ----------
    tasks_json : pathlib.Path or str
        The hub's `hub-config/tasks.json`.
    reference_date : str
        The round.

    Returns
    -------
    tuple of str
        The target end dates of the `wk R_t` task open to this reference
        date, from the week before it to the reference week (the target
        takes no horizon after 0); empty if the round has none.
    """
    config = json.loads(Path(tasks_json).read_text())
    ref = as_date(reference_date)
    out: set[str] = set()
    for rnd in config["rounds"]:
        for task in rnd["model_tasks"]:
            ids = task["task_ids"]
            targets = _options(ids.get("target", {}))
            refs = _options(ids.get("reference_date", {}))
            if RT_TARGET not in targets or reference_date not in refs:
                continue
            for end in _options(ids.get("target_end_date", {})):
                if 0 <= (ref - as_date(end)).days <= 7:
                    out.add(end)
    return tuple(sorted(out))


def _options(spec: Mapping[str, Any]) -> list[str]:
    """Return a task id's required and optional values."""
    return [*(spec.get("required") or []), *(spec.get("optional") or [])]


def rt_table(
    rt: pd.DataFrame,
    reference_date: str,
    end_dates: tuple[str, ...],
    forecast_date: str | dt.date | None = None,
) -> pd.DataFrame:
    """Return a fit's Rt quantiles as `RT_TARGET` rows of the submission.

    Parameters
    ----------
    rt : pandas.DataFrame
        A renewal fit's `nowcastrenewal.forecaster.RT_FILE`: `location`,
        `target_end_date`, `output_type_id` (the quantile level) and
        `value`.
    reference_date : str
        The round.
    end_dates : tuple of str
        The weeks the round takes (`rt_end_dates`); other weeks are left
        out.
    forecast_date : str or datetime.date, optional
        As in `to_hub_table`.

    Returns
    -------
    pandas.DataFrame
        Rows with `HUB_COLUMNS`, `output_type` `quantile`. Its
        `output_type_id` is of object dtype, so joined to the count rows
        their sample ids stay integers.
    """
    fd = as_date(forecast_date) if forecast_date else run_date()
    keep = rt.loc[rt["target_end_date"].astype(str).isin(end_dates)]
    out = pd.DataFrame(
        {
            "reference_date": str(reference_date),
            "forecast_date": str(fd),
            "target": RT_TARGET,
            "target_end_date": keep["target_end_date"].astype(str),
            "location": keep["location"],
            "output_type": RT_OUTPUT_TYPE,
            # Object, so a file with integer sample ids keeps them.
            "output_type_id": keep["output_type_id"]
            .astype(float)
            .astype(object),
            "value": keep["value"].astype(float),
        }
    )
    return out.sort_values(
        ["location", "target_end_date", "output_type_id"]
    ).reset_index(drop=True)[list(HUB_COLUMNS)]


def write_submission(
    table: pd.DataFrame,
    root: Path | str,
    mid: str,
) -> Path:
    """Write a submission table to its place under `root`.

    Parameters
    ----------
    table : pandas.DataFrame
        Output of `to_hub_table`.
    root : pathlib.Path or str
        Root of the hub checkout or an output directory.
    mid : str
        The model id.

    Returns
    -------
    pathlib.Path
        The path written.
    """
    refs = table["reference_date"].unique()
    if len(refs) != 1:
        raise ValueError("A submission must have one reference date.")
    if len(table["forecast_date"].unique()) != 1:
        raise ValueError("A submission must have one forecast date.")
    path = submission_path(root, str(refs[0]), mid)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(table, path)
    return path


# --------------------------------------------------------------------------
# Model metadata
# --------------------------------------------------------------------------

FUNDING: Final = (
    "This study is funded by the National Institute for Health and Care "
    "Research (NIHR) Health Protection Research Unit in Health Analytics & "
    "Modelling, a partnership between the UK Health Security Agency, "
    "Imperial College London and the London School of Hygiene & Tropical "
    "Medicine (grant code NIHR207404). The views expressed are those of the "
    "author(s) and not necessarily those of the NIHR, UK Health Security "
    "Agency or the Department of Health and Social Care."
)
"""Sam Abbott's HPRU funding statement (`team_funding` in the metadata)."""

METADATA_DEFAULTS: Final[dict[str, Any]] = {
    "team_name": "seabbs",
    "team_abbr": TEAM_ABBR,
    "model_contributors": [
        {
            "name": "Sam Abbott",
            "affiliation": "London School of Hygiene & Tropical Medicine",
            "orcid": "0000-0001-8057-8037",
            "email": "sam.abbott@lshtm.ac.uk",
        }
    ],
    "license": "CC-BY-4.0",
    "repo_url": "https://github.com/seabbs/bdbv-hub-modelling",
    "website_url": "https://samabbott.co.uk/bdbv-hub-modelling/",
    "ensemble_of_hub_models": False,
    "team_funding": FUNDING,
}
"""Fields shared by every model's metadata."""


def render_model_metadata(
    model_abbr: str,
    model_name: str,
    methods: str,
    methods_long: str,
    data_inputs: str,
    model_version: str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a model metadata record in the hub's field order.

    Parameters
    ----------
    model_abbr : str
        Model abbreviation (the part after `seabbs-`).
    model_name : str
        Full model name, under 50 characters.
    methods : str
        Short description, at most 200 characters.
    methods_long : str
        Full description.
    data_inputs : str
        The data the model uses.
    model_version : str, optional
        A version identifier, such as a git SHA.
    overrides : mapping, optional
        Any fields to replace in `METADATA_DEFAULTS`.

    Returns
    -------
    dict
        The metadata record.
    """
    base = {**METADATA_DEFAULTS, **(overrides or {})}
    record: dict[str, Any] = {
        "team_name": base["team_name"],
        "team_abbr": base["team_abbr"],
        "model_name": model_name,
        "model_abbr": model_abbr,
        "model_contributors": base["model_contributors"],
        "license": base["license"],
        "model_details": {
            "data_inputs": data_inputs,
            "methods": methods,
            "methods_long": methods_long,
        },
    }
    if model_version is not None:
        record["model_version"] = model_version
    for key in (
        "website_url",
        "repo_url",
        "team_funding",
        "ensemble_of_hub_models",
    ):
        record[key] = base[key]
    return record


def write_model_metadata(
    record: Mapping[str, Any], directory: Path | str
) -> Path:
    """Write a metadata record to `<team_abbr>-<model_abbr>.yml`.

    Parameters
    ----------
    record : mapping
        The metadata record.
    directory : pathlib.Path or str
        Where to write it, such as `packages/bdbvsubmission/model-metadata/`.

    Returns
    -------
    pathlib.Path
        The path written.
    """
    path = Path(directory) / f"{record['team_abbr']}-{record['model_abbr']}.yml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            dict(record), sort_keys=False, allow_unicode=True, width=1000
        )
    )
    return path
