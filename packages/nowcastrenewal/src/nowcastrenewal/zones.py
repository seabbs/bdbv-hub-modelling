# Copyright (c) 2026 The bdbv-hub-modelling authors. MIT licence, see LICENSE.
"""`renewal`, `renewalbvd` and `renewalsp`, fitted on health zones.

`ZoneRenewalModel` fits the zones with a case in the random walks' walk
window in one model, nested in their provinces (`model.zones`), draws
the silent zones with them from its posterior through its fitted
importation (`model.zones.extend_vintage`), and sums the zones to the
provinces, sample by sample. National is the sum of the
provinces. `SingleZoneRenewalModel` fits each zone with a recent case on
its own (`renewalsp`). Importing this module registers all three.
"""

from __future__ import annotations

import datetime as dt
import os
import time
import urllib.error
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from nowcastrenewal.forecaster import (
    VARIANTS,
    RenewalFit,
    RenewalModel,
    RenewalRun,
    Variant,
    _as_reported,
    _silent_walk,
    bvd_record,
    bvd_release,
    check_national,
    engine,
    fit_config,
    horizon_weeks,
    model_config,
    report_cutoffs,
    run_nuts,
    run_records,
    samples_table,
    task_imputation_delay,
    task_places,
)
from nowcastutils.base import (
    SAMPLE_COLUMNS,
    ForecastTask,
    ModelUnavailableError,
    register_model,
)
from nowcastutils.files import write_csv

__all__ = [
    "ZONES_FILE",
    "ZONE_VARIANTS",
    "SingleZoneRenewalModel",
    "ZoneRenewalFit",
    "ZoneRenewalModel",
    "sum_zones",
    "task_zones",
    "zone_target",
    "zone_vintage_inputs",
]

ZONE_VARIANTS: Final = ("renewal", "renewalbvd")
"""The variants fitted on health zones."""

ZONES_FILE: Final = "zones.csv.gz"
"""Zone samples in a zone fit's directory, summing to its provinces."""


def _data_dir() -> Path:
    return Path(os.environ.get("BDBV_DATA_DIR") or "data")


def _zone_daily(task: ForecastTask) -> pd.DataFrame:
    """Return the daily zone onsets the task carries, up to its vintage."""
    if task.zone_daily_history is None:
        msg = "The task carries no daily zone onsets (`zone_daily_history`)."
        raise ValueError(msg)
    return task.zone_daily_history


def _zone_names(task: ForecastTask) -> pd.DataFrame:
    """Return the hub's zone codes and names the task carries."""
    if task.zone_history is None:
        msg = "The task carries no hub zone rows (`zone_history`)."
        raise ValueError(msg)
    return task.zone_history


def _zones_of(task: ForecastTask) -> tuple[str, ...]:
    """Return the task's zone codes in its provinces, sorted."""
    from nowcastutils.hierarchy import province_of

    return tuple(
        sorted(z for z in task.zone_codes if province_of(z) in task.locations)
    )


def task_zones(task: ForecastTask) -> tuple[str, ...]:
    """Return the health zones of a task's provinces.

    The task's zone codes (`ForecastTask.zone_codes`, the hub's codes),
    which its forecast is checked against.

    Parameters
    ----------
    task : ForecastTask
        A province task.

    Returns
    -------
    tuple of str
        Zone codes, sorted.

    Raises
    ------
    ModelUnavailableError
        If the task holds no zone codes (a task built without zone data).
    """
    if not task.zone_codes:
        raise ModelUnavailableError(
            "A zone fit needs the task's zone codes; build the task with "
            "the zone data (`load_zone_data`)."
        )
    return _zones_of(task)


def zone_target(task: ForecastTask) -> pd.DataFrame:
    """Weekly known and imputed onsets of each zone at a task's vintage.

    Summed by epi week from the daily zone table, as the province target
    is from the daily province table.

    Parameters
    ----------
    task : ForecastTask
        A province task.

    Returns
    -------
    pandas.DataFrame
        Columns `vintage`, `location`, `target_end_date`, `observed`,
        `imputed`, `total` and `cutoff`, for zone weeks with an onset.
    """
    daily = _zone_daily(task)
    rows = daily.loc[daily["vintage"].astype(str) == task.vintage]
    onset = pd.to_datetime(rows["onset_date"])
    end = onset + pd.to_timedelta((5 - onset.dt.day_of_week) % 7, unit="D")
    out = (
        rows.assign(target_end_date=end.dt.date.astype(str))
        .groupby(["location", "target_end_date"], as_index=False)[
            ["observed", "imputed"]
        ]
        .sum()
    )
    out["total"] = out["observed"] + out["imputed"]
    out["vintage"] = task.vintage
    out["cutoff"] = str(rows["cutoff"].iloc[0]) if len(rows) else task.cutoff
    return out.loc[out["total"] > 0].reset_index(drop=True)


def silent_zones(task: ForecastTask) -> set[str]:
    """Return the task's zones with no case in the walk window.

    The random walks' rule (`nowcastbaselines.introductions.
    silent_places`): no onset in the `walk_weeks` weeks up to the week
    holding the vintage's cut-off, as reported at the vintage. So the
    renewal and the walks call the same zones silent.

    Parameters
    ----------
    task : ForecastTask
        A task of a real release whose locations are provinces, or the
        zones of its provinces (as `ZoneRenewalModel.run` takes it).

    Returns
    -------
    set of str
        Zone codes.
    """
    from nowcastbaselines.chain import Settings
    from nowcastbaselines.introductions import silent_places
    from nowcastutils.dates import epiweek_end
    from nowcastutils.hierarchy import is_zone, province_of

    if any(is_zone(c) for c in task.locations):
        zones = tuple(task.locations)
        task = replace(
            task, locations=tuple(sorted({province_of(z) for z in zones}))
        )
    else:
        zones = task_zones(task)
    weekly = zone_target(task).pivot_table(
        index="location", columns="target_end_date", values="total"
    )
    cut = epiweek_end(task.cutoff)
    first = min([cut, *(dt.date.fromisoformat(w) for w in weekly.columns)])
    weeks = [str(d.date()) for d in pd.date_range(first, cut, freq="7D")]
    reported = weekly.reindex(index=list(zones), columns=weeks).fillna(0.0)
    quiet = silent_places(
        reported.to_numpy(float), weeks.index(str(cut)), Settings().walk_weeks
    )
    return {z for z, q in zip(zones, quiet, strict=True) if q}


def _zone_walk_task(task: ForecastTask) -> ForecastTask:
    """Return the random walks' zone task, for their silent zones."""
    from nowcastbaselines.zones import zone_task
    from nowcastutils.snapshots import weekly_counts

    daily = _zone_daily(task)
    counts = weekly_counts(daily, task.vintage)
    history = (
        daily
        if task.daily_history is None
        else pd.concat([task.daily_history, daily], ignore_index=True)
    )
    return zone_task(task, counts, task_zones(task), history)


def sum_zones(samples: pd.DataFrame) -> pd.DataFrame:
    """Sum zone samples to their provinces, sample by sample.

    Parameters
    ----------
    samples : pandas.DataFrame
        Zone samples (`SAMPLE_COLUMNS`), optionally with `as_of`.

    Returns
    -------
    pandas.DataFrame
        Province samples with the same columns.
    """
    from nowcastutils.hierarchy import province_of

    keys = [c for c in samples.columns if c not in ("location", "value")]
    out = (
        samples.assign(location=samples["location"].map(province_of))
        .groupby(["location", *keys], as_index=False, dropna=False)["value"]
        .sum()
    )
    return pd.DataFrame(out).loc[:, list(samples.columns)]


ZONE_ESTIMATES: Final = "zone_estimates.csv"
"""BVDOutbreakSize's zone Rt summaries in a release (its zone model)."""

ZONE_PREFIX: Final = "zone:"
"""Prefix of the BVD term groups of single zones (`zone_bvd_rows`)."""


def _norm(name: object) -> str:
    import unicodedata

    text = unicodedata.normalize("NFKD", str(name))
    return "".join(c for c in text if c.isalnum()).lower()


def zone_bvd_rows(
    release: Any,
    names: pd.DataFrame,
    cache_dir: Path | str | None = None,
    download: Any = None,
) -> pd.DataFrame | None:
    """BVDOutbreakSize's zone Rt of a release, as rows of the BVD term.

    Each zone in the release's `zone_estimates.csv` becomes a row like a
    patch row of `nowcastrenewal.bvd.model_rows`: its log median and the
    log-scale SD of its 90% interval, relative to the release's national
    reference. Its group is `ZONE_PREFIX` and its hub code, and its member
    is that code. Zones are matched to hub codes by province and name
    (`health_zones.csv`); zones with no hub case yet are not matched and
    are dropped, as they are not fitted.

    Parameters
    ----------
    release : nowcastrenewal.bvd.BvdRelease
        The release chosen for the vintage.
    names : pandas.DataFrame
        Hub zone codes and names: rows with `location` and
        `location_name` (a task's `zone_history`).
    cache_dir : pathlib.Path or str, optional
        The release cache. Defaults to `bvd/` in the data directory. The
        file is downloaded into the release's folder if missing.
    download : callable, optional
        Returns the bytes at a URL.

    Returns
    -------
    pandas.DataFrame or None
        Columns of `nowcastrenewal.bvd.MODEL_ROW_COLUMNS`, or None if the
        release has no zone estimates.
    """
    from nowcastrenewal import bvd

    cache = Path(cache_dir) if cache_dir is not None else _data_dir() / "bvd"
    path = cache / release.tag / ZONE_ESTIMATES
    if not path.exists():
        url = (
            f"https://github.com/{bvd.BVD_REPO}/releases/download/"
            f"{release.tag}/{ZONE_ESTIMATES}"
        )
        try:
            body = (download or bvd._download)(url)
        except urllib.error.HTTPError as err:
            if err.code != 404:
                raise
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    est = pd.read_csv(path)
    hub = names.drop_duplicates("location")
    key = hub["location"].str[:5] + "|" + hub["location_name"].map(_norm)
    code = dict(zip(key, hub["location"], strict=True))
    est = est.assign(
        location=(
            est["province"].map(bvd.BVD_PROVINCES)
            + "|"
            + est["label"].map(_norm)
        ).map(code)
    ).dropna(subset=["location", "R_T_median", "R_T_lower", "R_T_upper"])
    est = est.loc[(est["R_T_lower"] > 0) & (est["R_T_median"] > 0)]
    nat = bvd.model_rows(release).iloc[0]
    ref = (float(nat["ref_log_mean"]), float(nat["ref_log_sd"]))
    rows = [
        (
            ZONE_PREFIX + str(r["location"]),
            str(r["location"]),
            release.as_of.isoformat(),
            float(np.log(r["R_T_median"])),
            float(np.log(r["R_T_upper"] / r["R_T_lower"]) / (2 * bvd.Z90)),
            *ref,
        )
        for _, r in est.iterrows()
    ]
    if not rows:
        return None
    return pd.DataFrame(rows, columns=list(bvd.MODEL_ROW_COLUMNS))


def zone_vintage_inputs(
    task: ForecastTask,
    config: Any,
    bvd: pd.DataFrame | None = None,
    receivers: tuple[str, ...] = (),
) -> Any:
    """Turn a zone task into the zone model's `ZoneVintage`.

    Parameters
    ----------
    task : ForecastTask
        A task whose locations are zones and whose target is
        `zone_target`. Its mask names provinces.
    config : nowcastrenewal.model.ModelConfig
        The model configuration.
    bvd : pandas.DataFrame, optional
        BVDOutbreakSize rows (`bvd_rows`).
    receivers : tuple of str
        Silent zones the importation kernel is normalised over but that
        are not fitted (`model.zones.extend_vintage`).

    Returns
    -------
    nowcastrenewal.model.zones.ZoneVintage
        The model inputs.
    """
    from nowcastrenewal.model.zones import zone_vintage_from_tables
    from nowcastutils.releases import release_pairs

    pairs = None
    if task.delay_history is not None:
        pairs = release_pairs(task.delay_history, task.vintage)
    return zone_vintage_from_tables(
        task.target,
        task.delays,
        task.vintage,
        tuple(task.locations),
        _zone_places(replace(task, locations=(*task.locations, *receivers))),
        config,
        horizon_weeks(task),
        tuple(task.mask),
        bvd=bvd,
        release_pairs=pairs,
        imputation_delay=task_imputation_delay(task),
        receivers=receivers,
    )


def _zone_places(task: ForecastTask) -> pd.DataFrame:
    """Return the places of a zone task's zones, in its order.

    A zone the task's places lack takes its province's centre and the
    mean population of the province's zones that have places, or the
    median zone's where none do.
    """
    from nowcastutils.hierarchy import is_zone, province_of

    places = task_places(task).set_index("location")
    zones = places.loc[places.index.map(is_zone)]
    missing = [z for z in task.locations if z not in places.index]
    if missing:
        parents = [province_of(z) for z in missing]
        absent = sorted(set(parents) - set(places.index))
        if absent:
            raise ValueError(f"No places for provinces {absent}.")
        by_province = zones.groupby(zones.index.map(province_of))[
            "population"
        ].mean()
        median = zones["population"].median()
        centres = places.loc[parents, ["latitude", "longitude"]]
        extra = pd.DataFrame(
            {
                "population": [by_province.get(p, median) for p in parents],
                "latitude": centres["latitude"].to_numpy(),
                "longitude": centres["longitude"].to_numpy(),
            },
            index=pd.Index(missing, name="location"),
        )
        places = pd.concat([places, extra])
    return places.loc[list(task.locations)].reset_index()


@dataclass(frozen=True)
class ZoneRenewalFit(RenewalFit):
    """A `RenewalFit` with the zone samples it sums to its provinces.

    Attributes
    ----------
    zones : pandas.DataFrame or None
        Final-count samples of every zone of the task's provinces
        (`SAMPLE_COLUMNS`). They sum to `samples`.
    """

    zones: pd.DataFrame | None = None


@dataclass
class ZoneRenewalModel(RenewalModel):
    """A renewal variant fitted on health zones, as a `ForecastModel`.

    Attributes
    ----------
    zone_samples : pandas.DataFrame or None
        Zone samples of the last `forecast`, which sum to its province
        samples.
    """

    zone_samples: pd.DataFrame | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def run(self, task: ForecastTask) -> RenewalRun:
        """Fit the zone model to a zone task and draw trajectories.

        As `RenewalModel.run`, with `zone_vintage_inputs` and
        `build_zone_model`, and the delay summaries by province. The
        zones with a recent case are fitted. The silent ones
        (`silent_zones`) are drawn with them in one forecast pass of the
        extended model (`model.zones.extend_vintage`) from the fit's
        posterior, so their infections come only through its fitted
        importation and sample k of every zone is one joint draw. The
        posterior checks and the weekly Rt are the fitted zones'.

        Raises
        ------
        ModelUnavailableError
            If every zone of the task is silent.
        """
        from nowcastrenewal.model.zones import build_zone_model, extend_vintage

        quiet_set = silent_zones(task)
        quiet = tuple(z for z in task.locations if z in quiet_set)
        fitted = tuple(z for z in task.locations if z not in quiet_set)
        if not fitted:
            raise ModelUnavailableError(
                f"{self.name}: every zone of {task.vintage} is silent."
            )
        places = _zone_places(task)
        task = replace(task, locations=fitted)

        pr = engine()
        config = model_config(self.variant)
        rows, used = None, None
        if config.bvd:
            from nowcastrenewal import bvd

            release = bvd_release(task)
            used = bvd_record(release)
            if release is not None:
                rows = bvd.model_rows(release)
                zone_rows = zone_bvd_rows(release, _zone_names(task))
                if zone_rows is not None:
                    rows = pd.concat([rows, zone_rows], ignore_index=True)
                    used = {**used, "zones": len(zone_rows)}
            self._log(f"variant={self.name} bvd={used['tag'] or 'none'}")
        data = zone_vintage_inputs(task, config, bvd=rows, receivers=quiet)
        # The same centring rule as every variant (`fit_config`).
        config = fit_config(self.variant, data)
        model = build_zone_model(data, config)
        self._log(
            f"variant={self.name} zones={data.n_provinces} sampling start"
        )
        start, cpu_start = time.perf_counter(), time.process_time()
        run_nuts(model, data, {"seed": task.seed, **self.sampler})
        np.asarray(next(iter(model.mcmc.get_samples().values())))
        seconds = time.perf_counter() - start
        cpu_seconds = time.process_time() - cpu_start
        self._log(
            f"variant={self.name} sampling done seconds={seconds:.0f} "
            f"cpu_seconds={cpu_seconds:.0f}"
        )
        ext = extend_vintage(data, quiet, places)
        sampled = {
            k: v
            for k, v in model.mcmc.get_samples().items()
            if k in model.mcmc.last_state.z
        }
        traj = pr.forecast_trajectories(
            build_zone_model(ext, config),
            ext,
            task.n_samples,
            seed=task.seed,
            posterior=sampled,
            as_of=tuple(report_cutoffs(task, ext)),
        )
        # The posterior checks are the fitted zones' (the first columns).
        p = len(data.names)
        draws = {
            **traj,
            "insample": {k: v[..., :p] for k, v in traj["insample"].items()},
            "rt": pr.weekly_rt(model, data, seed=task.seed),
        }
        # Each province's delay, which its zones share, and the shared one.
        trace, extra, delay = run_records(model, data, data.parent_names)
        return RenewalRun(
            self.variant,
            data,
            draws,
            trace,
            extra,
            seconds,
            cpu_seconds,
            delay,
            used,
            forecast_data=ext,
        )

    def fit_inputs(self, task: ForecastTask) -> str:
        """Hash the zone data a fit of `task` reads beside the task.

        The daily zone onsets of every release up to the task's vintage
        in its provinces, the zones it forecasts (`task_zones`) and, for
        a BVD variant, the zone names matched to the release's zone
        estimates. The fit key holds it (`bdbvhub.store.store.fit_key`).

        Parameters
        ----------
        task : ForecastTask
            A province task.

        Returns
        -------
        str
            The first 16 hex digits of a SHA-256.
        """
        import hashlib

        from nowcastutils.hierarchy import province_of

        daily = _zone_daily(task)
        mine = daily.loc[
            (daily["vintage"].astype(str) <= task.vintage)
            & daily["location"].map(province_of).isin(task.locations)
        ]
        digest = hashlib.sha256(mine.to_csv(index=False).encode())
        digest.update(repr(_zones_of(task)).encode())
        if model_config(self.variant).bvd:
            digest.update(_zone_names(task).to_csv(index=False).encode())
        return digest.hexdigest()[:16]

    def forecast_fit(self, task: ForecastTask) -> ZoneRenewalFit:
        """Fit the zones together and sum them to the provinces.

        The zones with a case in the random walks' walk window are
        fitted in one model, and the silent ones are drawn with them from
        its posterior, taking infections only through its fitted
        importation (`run`, #768 item 8). Each province sample is
        the sum of its zones' samples of the same id. National is the
        sum of the provinces, and a task with national cases in no
        province stops (`check_national`).

        Parameters
        ----------
        task : ForecastTask
            A province task of a real release.

        Returns
        -------
        ZoneRenewalFit
            The run, province samples and zone samples.

        Raises
        ------
        ModelUnavailableError
            For a simulated task, which has no zone data.
        """
        if task.kind != "real":
            raise ModelUnavailableError(
                f"{self.name} fits the hub's zone data, which a "
                f"{task.kind} task does not have."
            )
        check_national(task)
        zt = replace(task, target=zone_target(task), locations=task_zones(task))
        runs = self._run_all({"all": zt})
        run = runs["all"]
        zone_samples = samples_table(zt, run.places, run.draws)
        later = _as_reported(zt, run)
        # The samples hold the zones too, so the forecast, the store and
        # the scores read the zone fit's own zones, not a split
        # (`nowcastbaselines.zones.with_zones` leaves them as they are).
        return ZoneRenewalFit(
            runs,
            pd.concat(
                [sum_zones(zone_samples), zone_samples[list(SAMPLE_COLUMNS)]],
                ignore_index=True,
            ),
            as_reported=sum_zones(pd.concat(later, ignore_index=True))
            if later
            else None,
            zones=zone_samples,
        )

    def forecast(self, task: ForecastTask) -> pd.DataFrame:
        """Return province samples, keeping the zone samples.

        As `RenewalModel.forecast`, and sets `zone_samples`.
        """
        out = super().forecast(task)
        fit = self.last_fit
        self.zone_samples = getattr(fit, "zones", None)
        return out

    def write_fit_dir(
        self, task: ForecastTask, fit_dir: Path | str, config_name: str
    ) -> dict[str, Any]:
        """Write the last fit's directory, with `ZONES_FILE`.

        As `RenewalModel.write_fit_dir`, plus the zone samples, which the
        submission takes in place of a split of the provinces.
        """
        record = super().write_fit_dir(task, fit_dir, config_name)
        if self.zone_samples is not None:
            write_csv(
                self.zone_samples.loc[:, list(SAMPLE_COLUMNS)],
                Path(fit_dir) / ZONES_FILE,
            )
        return record


@dataclass
class SingleZoneRenewalModel(ZoneRenewalModel):
    """A single-patch variant fitted to each zone on its own.

    Each zone with a case in the random walks' walk window is fitted with
    the single-patch model on its own series (`RenewalModel.run` on the
    zone). The silent zones take
    `randomwalknowcast`'s introductions (`_silent_walk`), as in
    `renewalsp`. Provinces are the sums of their zones, sample by sample,
    and national is their sum, so no province sample is split across
    zones. Zone fits pair draws by index, as `renewalsp`'s provinces do.
    It is `renewalsp` (Sam, 2026-11-24; #768 item 9).
    """

    def run(self, task: ForecastTask) -> RenewalRun:
        """Fit one zone on its own (`RenewalModel.run`)."""
        return RenewalModel.run(self, task)

    def forecast_fit(self, task: ForecastTask) -> ZoneRenewalFit:
        """Fit each zone with a recent case on its own and sum them.

        Parameters
        ----------
        task : ForecastTask
            A province task of a real release.

        Returns
        -------
        ZoneRenewalFit
            The runs by zone, province samples and zone samples.

        Raises
        ------
        ModelUnavailableError
            For a simulated task, which has no zone data.
        """
        if task.kind != "real":
            raise ModelUnavailableError(
                f"{self.name} fits the hub's zone data, which a "
                f"{task.kind} task does not have."
            )
        check_national(task)
        walk, silent = _silent_walk(_zone_walk_task(task))
        target = zone_target(task)
        fitted = tuple(z for z in task_zones(task) if z not in silent)
        tasks = {
            z: replace(task, target=target, locations=(z,), mask=())
            for z in fitted
        }
        runs = self._run_all(tasks)
        frames = [walk]
        later = [
            walk.loc[walk["target_end_date"] <= d].assign(as_of=d)
            for d in task.report_by
        ]
        for key, run in runs.items():
            frames.append(samples_table(tasks[key], run.data, run.draws))
            later.extend(_as_reported(tasks[key], run))
        zone_samples = pd.concat(frames, ignore_index=True)
        return ZoneRenewalFit(
            runs,
            pd.concat(
                [sum_zones(zone_samples), zone_samples[list(SAMPLE_COLUMNS)]],
                ignore_index=True,
            ),
            as_reported=sum_zones(pd.concat(later, ignore_index=True))
            if later
            else None,
            zones=zone_samples,
        )


def _factory(variant: Variant) -> ZoneRenewalModel:
    return ZoneRenewalModel(variant)


for _name in ZONE_VARIANTS:
    _variant = VARIANTS[_name]
    register_model(
        _name,
        partial(_factory, _variant),
        _variant.description,
        fitted=True,
    )

for _variant in VARIANTS.values():
    if _variant.per_province:
        register_model(
            _variant.name,
            partial(SingleZoneRenewalModel, _variant),
            _variant.description,
            fitted=True,
        )
