"""The fine-tuning runs of workflow/config.yaml.

A run is an entry of `runs`: its name and the `finetune` settings it
overrides. Every run takes the same rules (rules/finetune.smk,
rules/evaluate.smk, rules/publish.smk) through the `{run}` wildcard, and
keeps its files in `<ROOT>/runs/<name>/`. The rounds it trains on are
shared by every run (`<ROOT>/rounds/r<N>/`).
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from flowlib import grids, segments

CONFIG = Path(__file__).parents[1] / "config.yaml"


@dataclass(frozen=True)
class Run:
    """A run's settings and folders."""

    name: str
    ft: dict[str, Any]
    dir: str

    @property
    def train(self) -> str:
        """The trainer's folder: one `n<N>/` per checkpoint."""
        return f"{self.dir}/train"

    @property
    def mirror(self) -> str:
        """The rounds host's manifests and QC summaries, mirrored."""
        return f"{self.dir}/rounds"

    @property
    def elsewhere(self) -> bool:
        """Whether the run's batches come from another machine."""
        return self.ft["sims_from"] in ("release", "remote")

    @property
    def prev(self) -> dict[int, int | None]:
        """Each prefix's previous prefix, None for the first."""
        return {n: p for p, n in segments.segments(self.ft["prefixes"])}

    @property
    def check(self) -> int:
        """The checkpoint checked against zero-shot."""
        return int(self.ft.get("check_at", 1000))


def load_config(path: Path = CONFIG) -> dict[str, Any]:
    """Return the committed workflow config."""
    return yaml.safe_load(path.read_text())


def resolve(
    config: Mapping[str, Any], local: Mapping[str, str]
) -> dict[str, Run]:
    """Return each run of `config["runs"]`, its `finetune` settings resolved.

    A run's settings are `finetune` with its entry's keys on top, and the
    grid's rounds, validation round and sources where unset
    (`grids.resolve`). Where batches come from is this machine's choice
    when it sets SIMS_FROM. A RUN folder in the local config is an error,
    so a machine with one never starts a new run beside it.
    """
    if local.get("RUN"):
        raise ValueError(
            f"RUN={local['RUN']} is set in the local config, but each run's "
            "folder is $ROOT/runs/<name> (`runs` in workflow/config.yaml): "
            "link that folder there and drop RUN"
        )
    out = {}
    for name, overrides in (config.get("runs") or {}).items():
        ft = copy.deepcopy(dict(config["finetune"]))
        ft.update(overrides or {})
        ft = grids.resolve(ft, config["grid_rounds"])
        if local.get("SIMS_FROM"):
            ft["sims_from"] = local["SIMS_FROM"]
        segments.check_chunks(ft["prefixes"], ft["chunk"], ft["outbreaks"])
        segments.check_eval(ft["eval"])
        out[name] = Run(name, ft, f"{local['ROOT']}/runs/{name}")
    return out


def trained(config: Mapping[str, Any], found: Mapping[str, Run]) -> list[str]:
    """Return the runs `finetune` trains: `train_runs`, or every run."""
    names = config.get("train_runs")
    if names is None:
        return list(found)
    unknown = sorted(set(names) - set(found))
    if unknown:
        raise ValueError(f"train_runs names runs not in `runs`: {unknown}")
    return list(names)


def option(flag: str, value: object) -> str:
    """Return `flag value` for a command line, or nothing if `value` is None."""
    return "" if value is None else f"{flag} {value}"


def resume_options(run: Run, n: int) -> str:
    """Return the state segment n resumes from: the previous prefix's."""
    prev = run.prev[n]
    if prev is None:
        return ""
    return f"--prev {run.train}/n{prev}/state --prev-n {prev}"


def trainer_options(run: Run) -> str:
    """Return the run's optional trainer arguments, each only when set.

    The series bias and its rate, the loss horizon, the quiet-block
    floor and a layout other than ragged.
    """
    f = run.ft
    bias = f"--series-bias --bias-lr {f['bias_lr']}" if f["series_bias"] else ""
    return " ".join(
        a
        for a in (
            option("--loss-horizon-days", f.get("loss_horizon_days")),
            option("--quiet-min-cases", f.get("quiet_min_cases")),
            bias,
            "" if f["layout"] == "ragged" else f"--layout {f['layout']}",
        )
        if a
    )
