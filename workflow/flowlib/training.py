"""The fine-tuning runs' files, seeds and steps, for the rules.

Each run (`runs` in workflow/config.yaml, `flowlib.runs`) trains in
<ROOT>/runs/<run>/train/ one segment per checkpoint prefix
(`flowlib.segments`), on batches of the simulation rounds
(`flowlib.rounds`). `Training` holds what `finetune.smk`, `evaluate.smk`,
`publish.smk` and the report rules ask of the runs; `common.smk` builds
one and binds its methods.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from flowlib import (
    evaluations,
    hosts,
    manifest,
    notify,
    releases,
    runs,
    segments,
)
from flowlib.rounds import Rounds
from nowcastfoundation.sidecar import HELD_OUT_BLOCKS, SIDECAR

SEGMENT_COPY_BACK: tuple[str, ...] = (
    "outbreaks.txt",
    "consumed.txt",
    "block-combinations.json",
    "finetuned-ckpt/adapter_config.json",
)
"""Files publishing reads beside a segment's outputs, copied back with
them from a worker elsewhere (flowlib.compute)."""


class Training:
    """The fine-tuning runs of a workflow config on this machine.

    Parameters
    ----------
    config : mapping
        The workflow config (`runs`, `finetune`, `next_segment_only`).
    local : mapping
        This machine's settings (`localconfig.read`).
    rounds : Rounds
        The simulation rounds the runs train on.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        local: Mapping[str, str],
        rounds: Rounds,
    ) -> None:
        self.config = config
        self.local = local
        self.rounds = rounds
        self.runs: dict[str, runs.Run] = runs.resolve(config, local)
        self.dir = f"{local['ROOT']}/runs"
        self._listed: dict[str, set[int]] = {}

    # Runs and their settings.

    def trained_runs(self) -> list[str]:
        """Return the runs `finetune` trains."""
        return runs.trained(self.config, self.runs)

    def run_of(self, wc: Any) -> runs.Run:
        """Return a job's run."""
        return self.runs[wc.run]

    def train_dir(self, run: str) -> str:
        """Return a run's training folder."""
        return self.runs[run].train

    def ft(self, key: str) -> Callable[[Any], Any]:
        """Return a param: the job's run's `finetune` setting `key`."""
        return lambda wc: self.runs[wc.run].ft[key]

    def local_or_ft(self, key: str, setting: str) -> Callable[[Any], Any]:
        """Return a param: this machine's `key`, else the run's `setting`."""
        return lambda wc: self.local.get(key) or self.runs[wc.run].ft[setting]

    def segment_copy_back(self, wc: Any) -> list[str]:
        """Return the files a segment's worker copies back beside it."""
        train = self.train_dir(wc.run)
        return [f"{train}/n{wc.n}/{name}" for name in SEGMENT_COPY_BACK]

    def saved(self, run: str, n: int) -> bool:
        """Return whether checkpoint n of the run is saved."""
        return Path(f"{self.train_dir(run)}/n{n}/settings.json").exists()

    # Evaluation.

    def evaluate_prev(self, wc: Any) -> str | list[str]:
        """Return the settings of the checkpoint before segment n, if any."""
        prev = self.run_of(wc).prev[int(wc.n)]
        if not prev:
            return []
        return f"{self.train_dir(wc.run)}/n{prev}/settings.json"

    def first_prefix(self, run: str) -> int:
        """Return the run's first checkpoint: the one with no previous."""
        return next(n for n, p in self.runs[run].prev.items() if p is None)

    @staticmethod
    def evaluate_prev_dir(input: Any) -> Path | None:
        """Return the previous checkpoint's folder, if it has one."""
        return Path(input.prev).parent if input.prev else None

    def evaluate_copy_in(self, wc: Any) -> list[str]:
        """Return the run's step evaluations and held-out blocks."""
        train = self.train_dir(wc.run)
        return [f"{train}/evals/", f"{train}/{HELD_OUT_BLOCKS}"]

    def earlier_settings(self, wc: Any) -> list[str]:
        """Return the settings of the run's checkpoints before n."""
        train, n = self.train_dir(wc.run), int(wc.n)
        return [
            f"{train}/n{p}/settings.json"
            for p in self.run_of(wc).ft["prefixes"]
            if p < n
        ]

    def evaluate_inputs_from(self, wc: Any) -> list[str]:
        """Return files a checkpoint's evaluation reads but no rule makes."""
        r, n, train = self.run_of(wc), int(wc.n), self.train_dir(wc.run)
        return [
            f"{train}/n{n}/finetuned-ckpt/",
            f"{train}/{HELD_OUT_BLOCKS}",
            *(f"{train}/n{p}/{SIDECAR}" for p in r.ft["prefixes"] if p < n),
        ]

    def offloaded(self, run: str, n: int) -> list[str]:
        """Return checkpoint n's sidecar and step evaluations, if offloaded."""
        if self.runs[run].ft["eval"] != "offload":
            return []
        train = self.train_dir(run)
        return [f"{train}/n{n}/{SIDECAR}", f"{train}/evals/n{n}.done"]

    def validation_curve_inputs(self, wc: Any) -> list[str]:
        """Return each of the run's checkpoints, with its held-out losses."""
        train, ns = self.train_dir(wc.run), self.runs[wc.run].ft["prefixes"]
        return [f"{train}/n{n}/settings.json" for n in ns] + [
            f for n in ns for f in self.offloaded(wc.run, n)
        ]

    def check_report(
        self,
        input: Any,
        output: Any,
        params: Any,  # noqa: ARG002
        wildcards: Any,
    ) -> None:
        """Report whether checkpoint `check_at` beats zero-shot, and say so."""
        text, line = evaluations.check_report(
            Path(input[0]).parent,
            wildcards.run,
            int(wildcards.n),
            self.runs[wildcards.run].ft["prefixes"],
        )
        Path(output[0]).write_text(f"{text}\n")
        notify.post(line, self.local)

    # Seeds and segments.

    def training_seeds(self, run: str) -> list[int]:
        """Return the run's training seeds, in the order it trains on them."""
        f = self.runs[run].ft
        return segments.training_seeds(
            f["seed_file"], f["rounds"], self.rounds.seeds_of
        )

    def held_seeds(self, run: str) -> list[int]:
        """Return the run's held-out seeds, its validation round's."""
        return self.rounds.seeds_of(self.runs[run].ft["validation_round"])

    def held_range(self, run: str) -> str:
        """Return the run's held-out seeds as the trainer takes them."""
        seeds = self.held_seeds(run)
        return f"{seeds[0]} {seeds[-1]}"

    def listed_batches(self, run: str) -> set[int]:
        """Return the batches that can be had now, listed once per run.

        Listed when the run plans, not kept in a file, so a later run sees
        rounds finished since: the rounds host's whole rounds, mirrored in
        the run's `rounds/` (flowlib.hosts.mirrored_seeds), else the
        `sims-r<N>` releases' batches. Written to
        <run>/available-batches.txt for the record.
        """
        if run in self._listed:
            return self._listed[run]
        this = self.runs[run]
        wanted = {
            this.ft["validation_round"],
            *map(self.rounds.round_of, self.training_seeds(run)),
        }
        seeds: set[int] = set()
        for r in sorted(wanted):
            if this.ft["sims_from"] == "remote":
                seeds |= hosts.mirrored_seeds(
                    self.rounds.remote,
                    f"{self.rounds.remote_dir}/r{r}",
                    f"{this.mirror}/r{r}",
                )
            else:
                seeds |= releases.release_seeds(self.rounds.sim["repo"], r)
        record = Path(f"{this.dir}/available-batches.txt")
        record.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_seeds(sorted(seeds), record)
        self._listed[run] = seeds
        return seeds

    def available_seeds(self, run: str) -> list[int]:
        """Return the training seeds whose batches can be had now."""
        listed = self.listed_batches(run) if self.runs[run].elsewhere else None
        return segments.available_seeds(
            self.training_seeds(run), self.held_seeds(run), listed
        )

    def outbreaks_or_full(self, run: str, seed: int) -> int:
        """Return a batch's outbreaks if simulated, else a full batch."""
        return segments.outbreaks_or_full(
            [
                f"{self.runs[run].dir}/sims/settings-{seed}.csv",
                f"{self.rounds.dir}/r{self.rounds.round_of(seed)}/sims/"
                f"settings-{seed}.csv",
            ],
            self.rounds.sim["batch"],
        )

    def segment_seeds(self, run: str, n: int) -> list[int]:
        """Return segment n's seeds: n outbreaks, a spare, the held-out."""
        return segments.segment_seeds(
            n,
            self.available_seeds(run),
            self.held_seeds(run),
            lambda s: self.outbreaks_or_full(run, s),
        )

    def segment_inputs(self, wc: Any) -> dict[str, Any]:
        """Return a segment's batches, seed list and previous checkpoint.

        A run's first segment, if `finetune` does not train the run, also
        takes the final checkpoints of the runs it does train.
        """
        run, n = wc.run, int(wc.n)
        there, train = self.runs[run], self.train_dir(run)
        # A finished checkpoint is final: its batches are not gathered
        # again, so a machine without them never retrains it.
        if Path(f"{train}/n{n}/settings.json").exists():
            return {}
        out: dict[str, Any] = {
            "sims": [
                f"{there.dir}/sims/settings-{seed}.csv"
                for seed in self.segment_seeds(run, n)
            ],
            "seeds": f"{there.dir}/seeds.txt",
        }
        if there.ft["real_training"]:
            out["real"] = runs.real_training_blocks(there)
        if there.prev[n] is not None:
            out["prev"] = f"{train}/n{there.prev[n]}/settings.json"
        elif run not in self.trained_runs():
            # A run outside `train_runs` starts once their final
            # checkpoints exist, so it never takes their GPU.
            out["after"] = [
                f"{self.train_dir(r)}/n{self.runs[r].ft['prefixes'][-1]}"
                "/settings.json"
                for r in self.trained_runs()
            ]
        return out

    def fetch_source(self, wc: Any) -> dict[str, str] | list[str]:
        """Return a batch's files in local rounds, if the run reads them."""
        if self.runs[wc.run].ft["sims_from"] != "local":
            return []
        sims = f"{self.rounds.dir}/r{self.rounds.round_of(wc.seed)}/sims"
        return {
            "counts": f"{sims}/counts-{wc.seed}.csv.gz",
            "settings": f"{sims}/settings-{wc.seed}.csv",
        }

    def trainable(self, run: str) -> list[int]:
        """Return the prefixes done or within reach of the seed list."""
        f = self.runs[run].ft
        return segments.trainable(
            f["prefixes"],
            f["outbreaks"],
            self.rounds.sim["batch"],
            self.available_seeds(run),
            lambda n: self.saved(run, n),
            lambda s: self.outbreaks_or_full(run, s),
        )

    def to_train(self, run: str) -> list[int]:
        """Return the prefixes `finetune` asks for of the run."""
        return segments.to_train(
            self.trainable(run),
            lambda n: self.saved(run, n),
            bool(self.config.get("next_segment_only")),
        )

    def finetune_inputs(self, wc: Any) -> list[str]:  # noqa: ARG002
        """Return each trained run's checkpoints, held-out losses, checks."""
        out = []
        for run in self.trained_runs():
            there, train, ns = (
                self.runs[run],
                self.train_dir(run),
                self.to_train(run),
            )
            out += [f"{train}/n{n}/settings.json" for n in ns]
            # An offloaded run's held-out losses, scored after each segment.
            out += [f for n in ns for f in self.offloaded(run, n)]
            # Each checkpoint's 7-day check (rule seven_day_check).
            out += [f"{train}/n{n}/seven-day.json" for n in ns]
            # A machine that publishes as it trains (PUBLISH in local.env).
            if self.local["PUBLISH"]:
                out += [f"{train}/n{n}/published.json" for n in ns]
            # The check once its checkpoint is asked for, so one segment a
            # run does not train on to it.
            if there.check in ns:
                out.append(f"{there.dir}/reports/n{there.check}-check.txt")
        return out

    def published_checkpoints(self, wc: Any) -> list[str]:  # noqa: ARG002
        """Return every trained run's saved checkpoints' release records."""
        return [
            f"{self.train_dir(run)}/n{n}/published.json"
            for run in self.trained_runs()
            for n in self.runs[run].ft["prefixes"]
            if self.saved(run, n)
        ]
