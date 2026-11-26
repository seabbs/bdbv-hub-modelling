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
    compute,
    evaluations,
    hosts,
    manifest,
    notify,
    releases,
    runs,
    segments,
)
from flowlib.rounds import Rounds
from nowcastfoundation.sidecar import HELD_OUT_BLOCKS, HELD_OUT_KEY, SIDECAR

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
        The workflow config (`runs`, `finetune`).
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
        self._next: frozenset[int] | None = None

    # Runs and their settings.

    def trained_runs(self) -> list[str]:
        """Return the runs `finetune` trains."""
        return runs.trained(self.config, self.runs)

    def trainer_priority(self, wc: Any) -> float:
        """Return a run's trainer priority: the order runs take a GPU.

        `TRAIN_PRIORITY` plus a fraction below one, highest for the
        first of `train_runs`, then the rest of them in order, then the
        other runs in the config's order. Each run's trainer takes one
        `gpu` slot, so with a GPU per run every run trains at once, and
        with fewer they queue in this order.
        """
        trained = self.trained_runs()
        order = trained + [r for r in self.runs if r not in trained]
        rank = order.index(wc.run)
        return compute.TRAIN_PRIORITY + (len(order) - rank) / (len(order) + 1)

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
        """Return the files a segment's worker copies back beside it.

        Its checkpoint's `SEGMENT_COPY_BACK`, the segment's log over
        every attempt and its chunk timings (nowcastfoundation's
        `finetune.stream.write_timings`).
        """
        train = self.train_dir(wc.run)
        return [
            *(f"{train}/n{wc.n}/{name}" for name in SEGMENT_COPY_BACK),
            f"{train}/n{wc.n}.log",
            f"{train}/timings/n{wc.n}.json",
        ]

    def saved(self, run: str, n: int) -> bool:
        """Return whether checkpoint n of the run is saved."""
        return Path(f"{self.train_dir(run)}/n{n}/settings.json").exists()

    # Evaluation.

    def first_prefix(self, run: str) -> int:
        """Return the run's first checkpoint: the one with no previous."""
        return next(n for n, p in self.runs[run].prev.items() if p is None)

    def earlier_settings(self, wc: Any) -> list[str]:
        """Return the settings of the run's checkpoints before n."""
        train, n = self.train_dir(wc.run), int(wc.n)
        return [
            f"{train}/n{p}/settings.json"
            for p in self.run_of(wc).ft["prefixes"]
            if p < n
        ]

    def held_out_key(self, wc: Any) -> list[str]:
        """Return the run's held-out set key (`HELD_OUT_KEY`), once written.

        An input of every evaluation, so a new held-out set reruns them.
        The trainer writes it before step 0; until then there is none.
        """
        path = Path(self.train_dir(wc.run)) / HELD_OUT_KEY
        return [str(path)] if path.is_file() else []

    def evaluate_inputs_from(self, wc: Any) -> list[str]:
        """Return files a checkpoint's evaluation reads but no rule makes."""
        r, n, train = self.run_of(wc), int(wc.n), self.train_dir(wc.run)
        return [
            f"{train}/n{n}/finetuned-ckpt/",
            f"{train}/{HELD_OUT_BLOCKS}",
            *(f"{train}/n{p}/{SIDECAR}" for p in r.ft["prefixes"] if p < n),
        ]

    def offloaded(self, run: str, n: int) -> list[str]:
        """Return checkpoint n's sidecar, if its run evaluates offloaded."""
        if self.runs[run].ft["eval"] != "offload":
            return []
        return [f"{self.train_dir(run)}/n{n}/{SIDECAR}"]

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

    def next_seeds(self) -> frozenset[int]:
        """Return the seeds each trained run's next segment reads.

        Its first segment in reach not yet saved (`trainable`); its
        batches go before the held-out evaluation
        (`compute.NEXT_BATCH_PRIORITY`). Read once a DAG build.
        """
        if self._next is None:
            seeds: set[int] = set()
            for run in self.trained_runs():
                todo = [
                    n for n in self.trainable(run) if not self.saved(run, n)
                ]
                if todo:
                    seeds |= set(self.segment_seeds(run, min(todo)))
            self._next = frozenset(seeds)
        return self._next

    def batch_priority(self, wc: Any) -> float:
        """Return a batch's priority: the next segment's above the rest."""
        base = (
            compute.NEXT_BATCH_PRIORITY
            if int(wc.seed) in self.next_seeds()
            else compute.BATCH_PRIORITY
        )
        return segments.batch_priority(
            int(wc.seed), base, self.rounds.sim["seed_base"]
        )

    def segment_inputs(self, wc: Any) -> dict[str, Any]:
        """Return a segment's batches, seed list and previous checkpoint.

        Its own run's alone: no run waits on another.
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

    def still_training(self) -> list[str]:
        """Return the trained runs with a segment in reach left to train."""
        return [
            run
            for run in self.trained_runs()
            if any(not self.saved(run, n) for n in self.trainable(run))
        ]

    def finetune_inputs(self, wc: Any) -> list[str]:  # noqa: ARG002
        """Return each trained run's checkpoints, held-out losses, checks."""
        out = []
        for run in self.trained_runs():
            there, train, ns = (
                self.runs[run],
                self.train_dir(run),
                self.trainable(run),
            )
            out += [f"{train}/n{n}/settings.json" for n in ns]
            # An offloaded run's held-out losses, scored after each segment.
            out += [f for n in ns for f in self.offloaded(run, n)]
            # Each checkpoint's 7-day check (rule seven_day_check).
            out += [f"{train}/n{n}/seven-day.json" for n in ns]
            # Its validation forecasts and base Chronos-2's, for the
            # dashboard (rules validation_forecasts, _base).
            out += [f"{train}/n{n}/{evaluations.VALIDATION_FILE}" for n in ns]
            out += [f"{train}/{evaluations.VALIDATION_BASE}"] if ns else []
            # A machine that publishes as it trains (PUBLISH in local.env):
            # each release, a saved checkpoint's BDBV check, and its
            # held-out losses once scored.
            if self.local["PUBLISH"]:
                out += [f"{train}/n{n}/published.json" for n in ns]
                out += [
                    f"{train}/n{n}/ft-check.done"
                    for n in ns
                    if self.saved(run, n)
                ]
                out += [f for n in ns for f in self.released_held_out(run, n)]
            # The check once its checkpoint is within reach, so it never
            # pulls the training past the batches there are.
            if there.check in ns:
                out.append(f"{there.dir}/reports/n{there.check}-check.txt")
        return out

    def released_held_out(self, run: str, n: int) -> list[str]:
        """Return checkpoint n's held-out losses' upload, if offloaded."""
        if self.runs[run].ft["eval"] != "offload":
            return []
        return [f"{self.train_dir(run)}/n{n}/held-out-released.txt"]

    def published_checkpoints(self, wc: Any) -> list[str]:  # noqa: ARG002
        """Return the saved checkpoints' releases and held-out uploads."""
        return [
            f
            for run in self.trained_runs()
            for n in self.runs[run].ft["prefixes"]
            if self.saved(run, n)
            for f in [
                f"{self.train_dir(run)}/n{n}/published.json",
                *self.released_held_out(run, n),
            ]
        ]
