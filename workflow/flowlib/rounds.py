"""The simulation rounds' files, seeds and steps, for the rules.

A round `r<N>` is one grid block of `simulate.batches` batches, seeds
`simulate.seed_base` + 100 N on (`segments.round_seeds`), in
<ROOT>/rounds/r<N>/. This host simulates the seeds SIM_SEEDS lists
(config/local.env; empty: all) and waits for the rest from their host.
`Rounds` holds what `simulate.smk`, `report.smk` and the fine-tuning
rules ask of them; `common.smk` builds one and binds its methods.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from flowlib import (
    compute,
    grids,
    hosts,
    manifest,
    notify,
    segments,
    simulator,
)


class Rounds:
    """The rounds of a workflow config on this machine.

    Parameters
    ----------
    config : mapping
        The workflow config (`simulate`, `rounds`, `grid_rounds`,
        `retired_rounds`).
    local : mapping
        This machine's settings (`localconfig.read`).
    rules : object
        Snakemake's `rules`, whose batch outputs `batch_files` names.
    """

    def __init__(
        self, config: Mapping[str, Any], local: Mapping[str, str], rules: Any
    ) -> None:
        self.config = config
        self.local = local
        self.rules = rules
        self.sim: Mapping[str, Any] = config["simulate"]
        self.host = local["HOST_NAME"]
        self.dir = f"{local['ROOT']}/rounds"
        self.remote = local["REMOTE"]
        self.remote_dir = f"{local['REMOTE_ROOT'] or local['ROOT']}/rounds"
        self.grid_file: str = self.sim["grid_file"]
        self.mine = hosts.seeds(local["SIM_SEEDS"], self.seeds_of)

    def seeds_of(self, round_: int | str) -> list[int]:
        """Return a round's batch seeds, its grid block."""
        return segments.round_seeds(
            int(round_), self.sim["seed_base"], self.sim["batches"]
        )

    def round_of(self, seed: int | str) -> int:
        """Return the round of batch `seed`."""
        return (int(seed) - int(self.sim["seed_base"])) // 100

    def here(self, seed: int | str) -> bool:
        """Return whether this host simulates batch `seed`."""
        return hosts.simulates(int(seed), self.mine)

    def batch_priority(self, wc: Any) -> float:
        """Return a batch's priority, earlier seeds a fraction higher."""
        return segments.batch_priority(
            int(wc.seed), compute.BATCH_PRIORITY, self.sim["seed_base"]
        )

    def batch_files(self, wc: Any) -> dict[str, str]:
        """Return a batch's files: simulated here, or pushed by its host."""
        if self.here(wc.seed):
            out = self.rules.simulate_batch.output
            return {
                "counts": out.counts.format(**wc),
                "settings": out.settings.format(**wc),
            }
        return {"arrived": self.rules.await_batch.output[0].format(**wc)}

    def grid_id(self) -> str:
        """Return the grid file the simulator runs, as its manifest id."""
        return simulator.grid_id(self.grid_file)

    def round_records(self, wc: Any) -> list[str]:
        """Return a round's batch manifest rows."""
        return [
            f"{self.dir}/r{wc.round}/records/record-{s}.csv"
            for s in self.seeds_of(wc.round)
        ]

    def round_sims(self, wc: Any) -> list[str]:
        """Return a round's batches' counts, settings and latent files."""
        kinds = (
            ("counts", "csv.gz"),
            ("settings", "csv"),
            ("latent", "csv.gz"),
        )
        return [
            f"{self.dir}/r{wc.round}/sims/{kind}-{seed}.{ext}"
            for seed in self.seeds_of(wc.round)
            for kind, ext in kinds
        ]

    def release_owner(self, wc: Any) -> str | None:
        """Return the grid round `wc.round`'s release belongs to."""
        return grids.owner(
            int(wc.round),
            self.config["grid_rounds"],
            self.config.get("retired_rounds"),
        )

    def release_check(self, wc: Any) -> list[str]:  # noqa: ARG002
        """Raise unless the rounds asked for may be released; no files."""
        grids.check_release(
            self.config["rounds"],
            self.config["grid_rounds"],
            self.config.get("retired_rounds"),
        )
        return []

    def round_sources(self, wc: Any, input: Sequence[str]) -> str:  # noqa: ARG002
        """Return `r<N>=<file>` for each round in `rounds` and its file."""
        return " ".join(
            f"r{r}={path}"
            for r, path in zip(self.config["rounds"], input, strict=False)
        )

    def summary_sources(
        self, rounds: Iterable[int | str], summaries: Iterable[str]
    ) -> str:
        """Return `r<N>=<round's sims>:<summary>` for each round."""
        return " ".join(
            f"r{r}={self.dir}/r{r}/sims:{path}"
            for r, path in zip(rounds, summaries, strict=False)
        )

    def batch_fields(self) -> dict[str, Any]:
        """Return the manifest fields a batch takes from this host and grid."""
        return {
            "grid": self.grid_id(),
            "draws": self.sim["draws"],
            "batch": self.sim["batch"],
            "host": self.host,
        }

    def write_record(
        self,
        input: Any,  # noqa: ARG002
        output: Any,
        params: Any,
        wildcards: Any,
    ) -> None:
        """Write a batch's manifest row, kept if the manifest has it."""
        manifest.record(
            output[0],
            params.old,
            params.sims,
            round_=int(wildcards.round),
            seed=int(wildcards.seed),
            mine=self.here(wildcards.seed),
            **self.batch_fields(),
        )

    def mark_ready(
        self,
        input: Any,
        output: Any,
        params: Any,  # noqa: ARG002
        wildcards: Any,
    ) -> None:
        """Write a round's seed list and READY, and say so."""
        seeds = self.seeds_of(wildcards.round)
        manifest.mark_ready(seeds, output.seeds, output.ready)
        qc = Path(input[0]).read_text().strip()
        notify.post(
            f"round {wildcards.round} ready: {len(seeds)} batches, one grid "
            f"block; {qc}",
            self.local,
        )

    def push_record(
        self,
        input: Any,  # noqa: ARG002
        output: Any,  # noqa: ARG002
        params: Any,
        wildcards: Any,
    ) -> None:
        """Send a helper's batch and its manifest row to the rounds host."""
        seed = int(wildcards.seed)
        row = manifest.new_row(
            params.sims,
            round_=int(wildcards.round),
            seed=seed,
            **self.batch_fields(),
        )
        hosts.push(params.sims, seed, params.there, row, self.remote)
