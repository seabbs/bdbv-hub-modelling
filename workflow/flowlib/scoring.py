"""Checkpoint scoring units' commands and the German report's arguments.

A unit is one `nowcastgermany.checkpoint` study for one checkpoint
label, cached under its study folder (rules/score.smk). `Scoring` holds
what `score.smk`, `evaluate.smk` and the report rules ask of the units;
`common.smk` builds one and binds its methods.
"""

from __future__ import annotations

import shlex
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from flowlib import forecast, machine, notify, releases
from flowlib.training import Training

EVALUATE = "uv run --group chronos python -m nowcastgermany.checkpoint"
"""The scoring CLI."""

REFERENCE_NAMES = {
    "reference": "Zero-shot",
    "randomwalknowcast": "Random walk nowcast",
}
"""The German report's names of the reference unit's models."""

REFERENCE = "reference"
"""The label of the unit of the models no checkpoint changes."""

REFERENCE_CODE = (
    "packages/nowcastbaselines/src",
    "packages/nowcastfoundation/src",
    "packages/nowcastgermany/src/nowcastgermany/evaluation.py",
    "config/germany.toml",
)
"""What the reference's forecasts depend on, past their tasks: its models'
packages, the German forecasting code and config (as
`nowcastgermany.checkpoint.model_key`)."""


def reference_code(label: str, root: Path | str = ".") -> list[str]:
    """Return the files the reference unit depends on; none for a checkpoint."""
    if label != REFERENCE:
        return []
    out = []
    for part in REFERENCE_CODE:
        path = Path(root) / part
        files = sorted(path.rglob("*.py")) if path.is_dir() else [path]
        out += [str(f.relative_to(root)) for f in files]
    return out


def reference_key(label: str, root: Path | str = ".") -> str:
    """Return a digest of `reference_code`'s contents ("" for a checkpoint).

    A rule parameter: Snakemake reruns the unit when it changes (not on
    file times, so a checkout reruns nothing), and the unit redoes only
    the rows of the models whose own key moved.
    """
    import hashlib

    h = hashlib.sha256()
    files = reference_code(label, root)
    for f in files:
        h.update(f.encode())
        h.update((Path(root) / f).read_bytes())
    return h.hexdigest()[:12] if files else ""


def unit_shell(study: str, room: str, segment: str = "") -> str:
    """Return a unit's shell command, as a rule's `shell:` template.

    The unit of `study` caches under {params.cache} as {params.label}
    once `room` (a wait-for-room prefix ending `&& `, or empty) passes;
    `segment`, if given, is the run's own checkpoint folder it loads.
    """
    return (
        f"mkdir -p {{params.copy_back}} && {room}{{params.prefix}} nice -n 19 "
        f"env {{params.env}} {EVALUATE} {study} {{params.cache}} "
        f"{{params.label}} {segment}> {{log}} 2>&1 && touch {{output}}"
    )


def report_args(
    run: str, saved: Iterable[int], references: Iterable[str], unit_dir: str
) -> str:
    """Return the German report's LABEL=NAME arguments and reference folders.

    `saved` are the run's saved prefixes; each reference label's rows are
    read from `unit_dir`/<label>.
    """
    names = REFERENCE_NAMES | {f"n{n}": f"{run} n{n}" for n in saved}
    args = [shlex.quote(f"{k}={v}") for k, v in names.items()]
    args += [f"--from {lb}={unit_dir}/{lb}" for lb in references]
    return " ".join(args)


GERMANY_UNIT = "germany-days"
"""The German units' cache folder, as `evaluation.checkpoint.GERMANY_UNIT`."""


class Scoring:
    """The checkpoint scoring units of a workflow config on this machine.

    Parameters
    ----------
    config : mapping
        The workflow config (`score`).
    local : mapping
        This machine's settings (`localconfig.read`): SCORE_CACHE, the
        room to wait for and FORECAST_DEVICE/FORECAST_THREADS, which
        override the config's `forecast` (`flowlib.forecast`).
    training : Training
        The fine-tuning runs whose checkpoints are scored.
    checkpoints : object
        Snakemake's `checkpoints`, whose `release_list` lists the
        releases to score.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        local: Mapping[str, str],
        training: Training,
        checkpoints: Any,
    ) -> None:
        self.sc: Mapping[str, Any] = config["score"]
        self.local = local
        self.training = training
        self.checkpoints = checkpoints
        self.cache = local["SCORE_CACHE"]
        self.room = (
            f"workflow/scripts/wait_for_room.sh {local['MIN_FREE_GB']} "
            f"{local['MIN_DISK_GB']} {self.cache} {local['MAX_LOAD']} && "
        )
        # Every forecasting job's device and threads (`forecast` in the
        # config, which the machine may override).
        self.forecast = forecast.settings(config, local)
        self.unit_threads = self.forecast.threads
        self.eval_device = self.forecast.device

    def checkpoint_labels(self, wc: Any = None) -> list[str]:  # noqa: ARG002
        """Return the checkpoints to score, less `score.skip`."""
        if self.sc["labels"] is not None:
            return list(self.sc["labels"])
        listing = self.checkpoints.release_list.get().output[0]
        return releases.chosen(listing, self.sc)

    def score_order(self) -> list[str]:
        """Return the labels, oldest published first (none before listed)."""
        return releases.oldest_first(f"{self.cache}/releases.json", self.sc)

    def unit_priority(self, base: float) -> Callable[[Any], float]:
        """Return a unit's priority: `base`, newer checkpoints higher."""
        return lambda wc: releases.priority(self.score_order(), wc.label, base)

    def unit_env(self, wc: Any, threads: int) -> str:  # noqa: ARG002
        """Return the unit's device (torch and JAX), threads and batch."""
        env = machine.torch_env(self.local, threads, self.eval_device)
        env["JAX_PLATFORMS"] = forecast.JAX_PLATFORM[self.eval_device]
        return machine.shell_env(env)

    def unit_rows(self, study: str) -> Callable[[Any], str]:
        """Return the unit's rows folder, which a worker brings back."""
        return lambda wc: f"{self.cache}/{study}/{wc.label}/"

    def unit_shell(self, study: str, segment: str = "") -> str:
        """Return a unit's shell command (`unit_shell`)."""
        return unit_shell(study, self.room, segment)

    def run_scores_dir(self, run: str) -> str:
        """Return the run's score cache: its own checkpoints' units."""
        return f"{self.training.dir}/{run}/scores"

    def saved_prefixes(self, run: str) -> list[int]:
        """Return the run's saved checkpoint prefixes."""
        prefixes = self.training.runs[run].ft["prefixes"]
        return [n for n in prefixes if self.training.saved(run, n)]

    def run_units(self, run: str) -> list[str]:
        """Return the German units of the run's saved checkpoints."""
        unit = f"{self.run_scores_dir(run)}/{GERMANY_UNIT}"
        return [f"{unit}/n{n}.done" for n in self.saved_prefixes(run)]

    def references(self) -> list[str]:
        """Return the reference labels' German units, shared by every run."""
        unit = f"{self.cache}/{GERMANY_UNIT}"
        return [f"{unit}/{label}.done" for label in self.sc["reference"]]

    def report_names(self, wc: Any) -> str:
        """Return the German report's LABEL=NAME arguments and folders."""
        return report_args(
            wc.run,
            self.saved_prefixes(wc.run),
            self.sc["reference"],
            f"{self.cache}/{GERMANY_UNIT}",
        )

    def run_scores(self, wc: Any = None) -> list[str]:  # noqa: ARG002
        """Return each trained run's German report."""
        return [
            f"{self.training.dir}/{run}/reports/german/index.html"
            for run in self.training.trained_runs()
        ]

    def table_inputs(self, wc: Any) -> dict[str, list[str]]:  # noqa: ARG002
        """Return the units the score tables read, by study."""
        labels = self.checkpoint_labels()
        every = [*self.sc["reference"], *labels]
        return {
            "germany": [
                f"{self.cache}/{GERMANY_UNIT}/{label}.done" for label in every
            ],
            "release": [
                f"{self.cache}/heldout/{label}.done" for label in labels
            ],
        }

    def unit_stamps(self, wc: Any = None) -> list[str]:
        """Return every unit the tables read, as one list.

        `rule score` asks for them itself, so a new label's units run
        under tables already built.
        """
        return [f for files in self.table_inputs(wc).values() for f in files]

    def land_tables(
        self,
        input: Any,
        output: Any,  # noqa: ARG002
        params: Any,
        wildcards: Any,  # noqa: ARG002
    ) -> None:
        """Commit the score tables on the evidence branch and push them."""
        head = notify.land(
            input, params.evidence, params.checkpoints, params.branch
        )
        notify.post(
            f"checkpoint scores landed at {head[:9]} on origin/{params.branch}",
            self.local,
        )
