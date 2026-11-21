# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Simulate a batch of outbreaks from the grid config.

Each simulation draws its design and values (`draws`), runs its latent
infections and expected onsets (`nowcastrenewal.latent`, PyRenew
components), then its reporting (a `ReportingStep`: the R engine's
epinowcast step by default).
A draw that errors or reports fewer than `floors.min_cases` cases is drawn
again, up to `tries` times, then from a new design, up to `designs`, so the
batch always holds `n` simulations.
Independent strata are single-stratum simulations of their own, with their
own designs, sharing the outbreak's series length, calendar offset and
observation family.

The simulations advance together in passes: each pass runs one draw of
every simulation still without one, through one latent batch and one
reporting call. Every draw's generator is keyed by the batch seed and the
draw's place (simulation, design, stratum, try), so a simulation's draws
do not depend on the others'.
"""

from __future__ import annotations

import datetime as dt
import platform
import socket
import sys
import time
from collections.abc import Generator, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .draws import draw_design, draw_values, generator, read_grid
from .reporting import ReportingStep, ReportJob, RscriptReporting

__all__ = [
    "COUNTS",
    "LATENT",
    "simulate_batch",
]

COUNTS = (
    "outbreak",
    "timestep",
    "location",
    "reference_date",
    "report_date",
    "count",
)
"""The counts file's columns: each outbreak's daily release triangle as
increments, outbreak `<seed>-<index>-<draw>`."""

LATENT = (
    "day",
    "infections",
    "onsets",
    "r",
    "reff",
    "date",
    "location",
    "outbreak",
)
"""The latent file's columns: each outbreak's whole path (`location`
`all`), or each independent stratum's."""

START = dt.date(2026, 1, 3)
"""A series' first day with offset 0 (the R engine's)."""

# Generator tags, so no two kinds of draw share a key.
DESIGN, VALUES, SHARED, STRATUM, LATENT_KEY, REPORT_KEY = range(1, 7)


@dataclass
class _Job:
    id: str
    design: dict[str, Any]
    vals: dict[str, Any]
    key: tuple[int, ...]


@dataclass
class _Outcome:
    error: str | None = None
    vals: dict[str, Any] = field(default_factory=dict)
    counts: pd.DataFrame | None = None
    latent: pd.DataFrame | None = None
    cases: float = 0.0


Draw = Generator[_Job, _Outcome, Any]


@dataclass
class _Batch:
    grid: Mapping[str, Any]
    seed: int
    draws: int
    weeks: int | None

    def redraw_reason(self, cases: float) -> str | None:
        floor = self.grid["floors"]["min_cases"]
        return f"fewer than {floor} cases" if cases < floor else None

    def cell(
        self, design: dict[str, Any], keys: tuple[int, ...], sim_id: str
    ) -> Draw:
        """Up to `tries` draws of a design's values; the first kept."""
        n_s = int(design["n_places"]) * int(design["n_ages"])
        if design["strata_link"] == "independent" and n_s > 1:
            return (yield from self.independent(design, keys, sim_id))
        failures: list[dict[str, Any]] = []
        for t in range(1, int(self.grid["tries"]) + 1):
            key = (*keys, VALUES, t)
            v = draw_values(self.grid, design, generator(self.seed, *key))
            res = yield _Job(sim_id, design, v, key)
            reason = res.error or self.redraw_reason(res.cases)
            if reason is None:
                return res, failures
            failures.append(
                flat(
                    {
                        "outbreak": sim_id,
                        "try": t,
                        "reason": reason,
                        **design,
                        **v,
                    }
                )
            )
        return None, failures

    def independent(
        self, design: dict[str, Any], keys: tuple[int, ...], sim_id: str
    ) -> Draw:
        """Independent strata, each a single-stratum simulation."""
        n_p, n_a = int(design["n_places"]), int(design["n_ages"])
        place_ids = [f"P{i:02d}" for i in range(1, n_p + 1)]
        age_ids = [f"A{i}" for i in range(1, n_a + 1)]
        if n_p > 1 and n_a > 1:
            labels = [f"{p}-{a}" for a in age_ids for p in place_ids]
        elif n_a > 1:
            labels = age_ids
        else:
            labels = place_ids
        shared = draw_values(
            self.grid, design, generator(self.seed, *keys, SHARED)
        )
        quiet = set(shared["quiet_places"])
        # Age strata have no place, so none is quiet.
        heads = [lab.split("-")[0] for lab in labels]
        place_of = [
            place_ids.index(h) + 1 if h in place_ids else None for h in heads
        ]
        active = [s for s, p in enumerate(place_of) if p not in quiet]
        failures: list[dict[str, Any]] = []
        record = []
        parts, latents, values_of = [], [], []
        values: dict[str, Any] = {}
        for s in active:
            run = None
            for attempt in range(1, int(self.grid["tries"]) + 1):
                key = (*keys, STRATUM, s + 1, attempt)
                one = draw_design(
                    self.grid,
                    generator(self.seed, *key),
                    weeks=int(design["weeks"]),
                )
                one.update(
                    offset=shared["offset"],
                    strata="single",
                    n_places=1,
                    n_ages=1,
                    spread="even",
                    seeding="start",
                    mixing="none",
                    strata_link="linked",
                    # One family per outbreak.
                    observation=design["observation"],
                )
                res, f = yield from self.cell(one, key, f"{sim_id}/{labels[s]}")
                failures += f
                if res is not None:
                    run = res
                    break
            if run is None:
                return None, failures
            parts.append(run.counts.assign(location=labels[s]))
            latents.append(run.latent.assign(location=labels[s]))
            record.append(
                {
                    "trajectory": one["trajectory"],
                    "delay_family": one["delay_family"],
                    "observation": _first(run.vals["obs_family"]),
                    "size": run.cases,
                    "max_delay": run.vals["max_delay"],
                }
            )
            values_of.append(run.vals)
            if s == active[0]:
                values = dict(run.vals)
        for k in ("trajectory", "delay_family", "observation", "size"):
            values[f"stratum_{k}"] = ";".join(_text(r[k]) for r in record)
        values["max_delay"] = max(int(r["max_delay"]) for r in record)
        values["stratum_population"] = [
            float(np.sum(x["stratum_population"])) for x in values_of
        ]
        values["stratum_location"] = [labels[s] for s in active]
        values["quiet_places"] = sorted(quiet)
        counts = pd.concat(parts, ignore_index=True)
        out = _Outcome(
            vals=values,
            counts=counts,
            latent=pd.concat(latents, ignore_index=True),
            cases=float(counts["count"].sum()) / self.draws,
        )
        return out, failures

    def simulation(self, i: int) -> Draw:
        """One simulation: designs until one gives an outbreak."""
        sim_id = f"{self.seed}-{i:04d}"
        failures: list[dict[str, Any]] = []
        for d in range(1, int(self.grid["designs"]) + 1):
            key = (i, d)
            design = draw_design(
                self.grid,
                generator(self.seed, *key, DESIGN),
                weeks=self.weeks,
            )
            out, f = yield from self.cell(design, key, sim_id)
            failures += f
            if out is not None:
                return sim_id, design, out, len(f) + 1, d, failures
        raise RuntimeError(
            f"simulation {sim_id}: no design in {self.grid['designs']} gave one"
        )


def _first(x: Any) -> Any:
    return x[0] if isinstance(x, list | tuple) else x


def _text(e: Any) -> str:
    if isinstance(e, np.generic):
        e = e.item()
    if isinstance(e, bool):
        return "TRUE" if e else "FALSE"
    return repr(e) if isinstance(e, float) else str(e)


def flat(x: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten values for a settings row, as the R engine's `flat_values`.

    A one-element vector is its element, others are joined by ";".
    """
    out: dict[str, Any] = {}
    for k, e in x.items():
        if isinstance(e, list | tuple | np.ndarray):
            items = list(e)
            out[k] = (
                _scalar(items[0])
                if len(items) == 1
                else ";".join(_text(i) for i in items)
            )
        else:
            out[k] = _scalar(e)
    return out


def _scalar(e: Any) -> Any:
    if isinstance(e, np.generic):
        e = e.item()
    if isinstance(e, bool):
        return "TRUE" if e else "FALSE"
    return e


def _strata(design: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the strata in epinowcast's `.group` order: places, then ages."""
    n_p, n_a = int(design["n_places"]), int(design["n_ages"])
    return [
        {"place": f"P{p:02d}", "age": f"A{a}", ".group": (p - 1) * n_a + a}
        for p in range(1, n_p + 1)
        for a in range(1, n_a + 1)
    ]


def _n_days(values: Mapping[str, Any]) -> int:
    """Return the reporting frame's days: series, longest delay, horizon."""
    return 7 * (int(values["weeks"]) + int(values["max_delay"]) // 7 + 4)


def _run_pass(
    batch: _Batch,
    jobs: Sequence[_Job],
    reporting: ReportingStep,
    grid_file: str | Path,
) -> list[_Outcome]:
    """Run one draw of each job: the latent batch, then the reporting."""
    from nowcastrenewal.latent.simulate import simulate_latent_batch

    grid = batch.grid
    sims = [
        {
            "id": j.id,
            "key": int(
                generator(batch.seed, *j.key, LATENT_KEY).integers(2**31 - 1)
            ),
            "n_days": _n_days(j.vals),
            "design": j.design,
            "values": j.vals,
            "strata": _strata(j.design),
        }
        for j in jobs
    ]
    long, meta = simulate_latent_batch(
        sims,
        kinds=tuple(grid["factors"]["trajectory"]),
        incubation_cv=float(grid["constants"]["incubation_cv"]),
    )
    by_id = {str(k): g for k, g in long.groupby("id", sort=False)}
    outcomes: dict[str, _Outcome] = {}
    report_jobs = []
    for j in jobs:
        m = meta[j.id]
        if m["error"] is not None:
            outcomes[j.id] = _Outcome(error=str(m["error"]))
            continue
        rows = by_id[j.id]
        onsets = (
            rows[rows["group"] > 0]
            .pivot(index="day", columns="group", values="onsets")
            .sort_index()
            .to_numpy()
        )
        whole = rows[rows["group"] == 0].sort_values("day")
        dates = [
            (
                START + dt.timedelta(days=int(j.vals["offset"]) + int(d))
            ).isoformat()
            for d in whole["day"]
        ]
        latent = pd.DataFrame(
            {
                "day": whole["day"].to_numpy(),
                "infections": whole["infections"].to_numpy(),
                "onsets": whole["onsets"].to_numpy(),
                "r": whole["r"].to_numpy(),
                "reff": whole["reff"].to_numpy(),
                "date": dates,
                "location": "all",
            }
        )
        values = {
            **j.vals,
            "stratum_population": list(m["stratum_population"]),
            "attack": m["attack"],
        }
        outcomes[j.id] = _Outcome(vals=values, latent=latent)
        seed = generator(batch.seed, *j.key, REPORT_KEY).integers(1, 2**31 - 1)
        strata = _strata(j.design)
        report_jobs.append(
            ReportJob(
                j.id,
                int(seed),
                j.design,
                values,
                onsets,
                places=[s["place"] for s in strata],
                ages=[s["age"] for s in strata],
            )
        )
    for r in reporting.report(report_jobs, grid_file, batch.draws):
        o = outcomes[r.id]
        if r.error is not None:
            o.error = r.error
            continue
        o.vals = r.vals
        o.counts = r.counts
        o.cases = float(r.counts["count"].sum()) / batch.draws
    return [outcomes[j.id] for j in jobs]


def simulate_batch(
    grid_file: str | Path,
    seed: int,
    n: int,
    out_dir: str | Path,
    *,
    draws: int = 1,
    reporting: ReportingStep | None = None,
    rscript: str = "Rscript",
    enw_lib: str | None = None,
    weeks: int | None = None,
) -> dict[str, Path]:
    """Simulate a batch of outbreaks and write its files.

    Parameters
    ----------
    grid_file : str or pathlib.Path
        The grid config (`config/finetune/grid.yaml`).
    seed : int
        The batch's seed.
    n : int
        Simulations.
    out_dir : str or pathlib.Path
        Where the files go: `counts-<seed>.csv.gz`, `settings-<seed>.csv`,
        `latent-<seed>.csv.gz` and, if any try failed,
        `failures-<seed>.csv`.
    draws : int
        Observation draws per simulation.
    reporting : ReportingStep, optional
        The reporting step; the R engine's epinowcast step
        (`RscriptReporting` with `rscript` and `enw_lib`) if `None`.
    rscript, enw_lib : str
        The default reporting step's Rscript and pinned epinowcast library.
    weeks : int, optional
        Fix every series' length (pilots).

    Returns
    -------
    dict
        The files written, by name.

    Raises
    ------
    RuntimeError
        If a simulation gets no outbreak from `designs` designs.
    """
    grid = read_grid(grid_file)
    if reporting is None:
        reporting = RscriptReporting(rscript=rscript, enw_lib=enw_lib)
    batch = _Batch(grid, int(seed), int(draws), weeks)
    running: dict[int, Draw] = {}
    pending: dict[int, _Job] = {}
    seconds = dict.fromkeys(range(1, n + 1), 0.0)
    done: dict[int, Any] = {}
    for i in range(1, n + 1):
        running[i] = batch.simulation(i)
        pending[i] = next(running[i])
    while pending:
        order = sorted(pending)
        t0 = time.monotonic()
        outcomes = _run_pass(
            batch, [pending[i] for i in order], reporting, grid_file
        )
        share = (time.monotonic() - t0) / len(order)
        for i, outcome in zip(order, outcomes, strict=True):
            seconds[i] += share
            try:
                pending[i] = running[i].send(outcome)
            except StopIteration as stop:
                done[i] = stop.value
                del pending[i]
    stack = {
        **reporting.software(),
        "python_version": platform.python_version(),
        **{
            f"{name}_version": metadata.version(name)
            for name in ("jax", "numpyro", "pyrenew")
        },
        "host": socket.gethostname(),
    }
    counts, latent, settings, failures = [], [], [], []
    for i in range(1, n + 1):
        sim_id, design, out, tries, designs, f = done[i]
        failures += f
        c = out.counts
        counts.append(
            pd.DataFrame(
                {
                    "outbreak": [f"{sim_id}-{d}" for d in c["draw"]],
                    "timestep": "day",
                    "location": c["location"].to_numpy(),
                    "reference_date": c["reference_date"].to_numpy(),
                    "report_date": c["report_date"].to_numpy(),
                    "count": c["count"].astype(int).to_numpy(),
                }
            )
        )
        latent.append(out.latent.assign(outbreak=sim_id)[list(LATENT)])
        settings.append(
            flat(
                {
                    "outbreak": sim_id,
                    **design,
                    **out.vals,
                    "draws": draws,
                    "tries": tries,
                    "designs": designs,
                    "cases": out.cases,
                    "seconds": seconds[i],
                    **stack,
                }
            )
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    def write(rows: pd.DataFrame, name: str, ext: str) -> None:
        final = out_dir / f"{name}-{seed}.{ext}"
        tmp = out_dir / f".tmp-{final.name}"
        rows.to_csv(tmp, index=False)
        tmp.replace(final)
        written[name] = final

    # Counts before settings, so a reader that sees the settings sees the
    # whole batch.
    if failures:
        write(pd.DataFrame(failures), "failures", "csv")
    write(pd.concat(latent, ignore_index=True), "latent", "csv.gz")
    write(pd.concat(counts, ignore_index=True), "counts", "csv.gz")
    write(pd.DataFrame(settings), "settings", "csv")
    print(
        f"{n} simulations in {sum(seconds.values()):.1f} s (grid {grid_file})",
        file=sys.stderr,
    )
    return written
