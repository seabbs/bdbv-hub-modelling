"""The paper workflow's paths and command line arguments.

The paper's rules (`paper/workflow/Snakefile`) take their inputs and
arguments from a `PaperFlow`, so the rules hold no logic of their own.
`PaperFlow.load` reads the workflow config once, when the workflow is
built. Its methods that take `w` are Snakemake input and params
functions: `w` holds the rule's wildcards, `mode` and `study`.
"""

from __future__ import annotations

import os
import shlex
import sys
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

from nowcastfoundation.forecaster import local_checkpoint, newest_segment

from paperpipe.config import run_settings, run_studies
from paperpipe.outputs import (
    JOINT_STUDIES,
    MAIN_TABLES,
    SI_TABLES,
    STUDY_SI_TABLES,
    figure_names,
)

__all__ = [
    "BDBV",
    "DESIGN_TABLES",
    "HELD_OUT_INCLUDES",
    "PYTHON",
    "PaperFlow",
    "Wildcards",
]

HELD_OUT_INCLUDES: Final = (
    ("main", "scores"),
    ("main", "joint"),
    ("si", "scores-by-horizon"),
    ("si", "scores-by-location"),
)
"""The tables main.qmd and si.qmd include for every case study."""

DESIGN_TABLES: Final = (
    "grid-epidemic",
    "grid-reporting",
    "grid-distributions",
    "epinowcast-model",
    "heldout-set",
)
"""The SI tables of the synthetic design (`papertables.synthetic_grid`)."""


class Wildcards(Protocol):
    """The wildcards of a paper rule."""

    mode: str
    study: str


@dataclass(frozen=True)
class PaperFlow:
    """The paper workflow's settings, with its paths and arguments.

    Attributes
    ----------
    config : mapping
        The workflow config (`paper/workflow/config.yaml`), as `load`
        leaves it.
    evidence : mapping
        Each evidence entry that is set: a `ref` and `path` it is read
        at, or a `file` the workflow writes.
    realdata_after : str
        The last BDBV release the real-data models trained on.
    """

    config: Mapping[str, Any]
    evidence: Mapping[str, Mapping[str, str]]
    realdata_after: str

    @classmethod
    def load(
        cls,
        config: dict[str, Any],
        evidence: Mapping[str, Mapping[str, str]],
        root: Path,
        expand: Any,
    ) -> PaperFlow:
        """Check the config and settle what it leaves to the machine.

        The simulation folders are expanded in `config` itself (`expand`,
        `flowlib.localconfig.expand`), so each step that is passed the
        config reads the same folders, as are the final run's and the
        rounds' folders and evidence `file`s. `finetuned` is the spec of the
        final run's newest saved checkpoint, whose folder is put in the
        environment as `BDBV_CHRONOS_FT`, which every `_ft` model loads.

        Parameters
        ----------
        config : dict
            The workflow config.
        evidence : mapping
            Each evidence entry that is set, with its `ref` or `file`.
        root : pathlib.Path
            The repository root, holding `config/finetune.toml`.
        expand : callable
            Expands machine settings such as `$ROOT` in a folder.

        Returns
        -------
        PaperFlow

        Raises
        ------
        ValueError
            If the German benchmark's config splits at another date than
            `heldout.germany_from`.
        """
        germany = config["germany"]
        with Path(germany["config"]).open("rb") as f:
            split = tomllib.load(f)["nowcast"]["split_date"]
        if str(split) != str(config["heldout"]["germany_from"]):
            heldout = config["heldout"]["germany_from"]
            raise ValueError(
                f"{germany['config']} splits at {split} but "
                f"`heldout.germany_from` is {heldout}."
            )
        synthetic = config["synthetic"]
        synthetic["simulations"] = {
            mode: expand(folder)
            for mode, folder in synthetic["simulations"].items()
        }
        # Null leaves the run or rounds out, and what reads them pending.
        for key in ("final_run", "rounds_dir"):
            if config.get(key):
                config[key] = expand(config[key])
        # The `_ft` models load the final run's newest saved checkpoint,
        # pending until it has one. Outputs record its spec, not its folder.
        segment = (
            newest_segment(Path(config["final_run"]) / "train")
            if config.get("final_run")
            else None
        )
        config["finetuned"] = (
            None if segment is None else local_checkpoint(segment).hash
        )
        if segment is not None:
            os.environ.update(BDBV_CHRONOS_FT=str(segment))
        with (root / "config" / "finetune.toml").open("rb") as f:
            after = tomllib.load(f)["realdata_bdbv"]["release"]
        evidence = {
            name: {**entry, "file": expand(entry["file"])}
            if "file" in entry
            else entry
            for name, entry in evidence.items()
        }
        return cls(config, evidence, str(after))

    # Settings ------------------------------------------------------------

    @property
    def results(self) -> str:
        """The folder of every run's outputs."""
        return str(self.config["results"])

    @property
    def build(self) -> str:
        """The folder `publish_paper` copies a run's outputs to."""
        return str(self.config["build"])

    @property
    def germany(self) -> Mapping[str, Any]:
        """The German benchmark's block."""
        return dict(self.config["germany"])

    @property
    def ebola(self) -> Mapping[str, Any]:
        """The Ebola study's block."""
        return dict(self.config["ebola"])

    @property
    def synthetic(self) -> Mapping[str, Any]:
        """The synthetic study's block."""
        return dict(self.config["synthetic"])

    @property
    def triangle(self) -> str:
        """The German hub's reporting triangle."""
        return f"{self.germany['data']}/COVID-19_hospitalizations.csv"

    @property
    def ebola_draws(self) -> str:
        """The paper's own release backtest of `backtest_models`."""
        return f"{self.results}/ebola-backtest/forecasts.csv.gz"

    def studies(self, mode: str) -> list[str]:
        """Return a run's studies: development leaves out the held-out ones."""
        return list(run_studies(self.config["studies"], mode))

    def seeds(self, mode: str) -> str:
        """Return a run's synthetic seeds, as words."""
        seeds = run_settings(self.config, mode).synthetic_seeds
        return " ".join(str(s) for s in seeds)

    # Evidence ------------------------------------------------------------

    def pinned(self) -> dict[str, Mapping[str, str]]:
        """Return the evidence entries read from a git commit."""
        return {n: e for n, e in self.evidence.items() if "file" not in e}

    def evidence_file(self, name: str) -> str:
        """Return an evidence file: the workflow's, or named by its commit."""
        entry = self.evidence[name]
        if "file" in entry:
            return entry["file"]
        suffix = "".join(Path(entry["path"]).suffixes)
        return f"{self.results}/evidence/{name}-{entry['ref'][:12]}{suffix}"

    @staticmethod
    def options(files: Mapping[str, str]) -> str:
        """Return `paperpipe evidence` options naming each file."""
        return " ".join(
            f"--{n.replace('_', '-')} {f}" for n, f in files.items()
        )

    def evidence_args(self, names: Mapping[str, object]) -> str:
        """Return the `paperpipe evidence` options of the entries in `names`."""
        return self.options(
            {n: self.evidence_file(n) for n in self.evidence if n in names}
        )

    @property
    def rounds_dir(self) -> str | None:
        """The simulation rounds' folder, or None to leave them out."""
        return self.config.get("rounds_dir")

    def workflow_evidence(self) -> dict[str, str]:
        """Return the evidence step's inputs from workflow rules.

        The final run's held-out loss curve (rule heldout_curve), the
        published-range table's rows (rule sim_diseases) and the realism
        step's outputs (rule sim_realism), and the walk models' pairing
        scores and rho_p (rules si_copula and pairing_germany). Each is
        left out, and its SI result pending, while its run or rounds folder
        is null, or for the pairing scores while the German benchmark runs
        on dummy data.
        """
        files = {}
        evidence = f"{self.results}/evidence"
        if self.germany["real"]:
            files["si_copula"] = f"{evidence}/si-copula.csv"
            files["copula_rho"] = (
                f"{evidence}/pairing/pairing-germany-rho.csv.gz"
            )
        if final := self.config.get("final_run"):
            files["finetune_heldout"] = f"{final}/heldout-curve.csv"
        if where := self.rounds_dir:
            files["sim_diseases"] = f"{where}/report/sim-diseases.csv"
            files["sim_summaries"] = f"{evidence}/sim-summaries.csv"
            files["sim_realism"] = f"{evidence}/sim-realism.csv"
        return files

    def realism_inputs(self) -> dict[str, Any]:
        """Return the realism step's inputs.

        The grid's training rounds' summaries and the German development
        half's statistics (workflow rules round_summaries and
        germany_dev_stats).
        """
        where = self.rounds_dir
        return {
            "germany": f"{where}/report/germany-dev.csv",
            "rounds": [
                f"{where}/r{r}/summaries/outbreaks.csv"
                for r in self.config["grid_rounds"]["rounds"]
            ],
        }

    def live_arg(self) -> str:
        """Return the live run option, whose newest checkpoints are shown."""
        live = self.config.get("live_run")
        return f"--live-run {live}" if live else ""

    def comparisons_arg(self) -> str:
        """Return the comparisons' options: each name and its variant run."""
        pairs = self.config.get("comparisons") or {}
        return " ".join(f"--comparison {n}={r}" for n, r in pairs.items())

    # Further forecasts ---------------------------------------------------

    def provisional(self, study: str) -> str:
        """Return what is provisional in a study's further forecasts."""
        block = {"germany": self.germany, "ebola": self.ebola}.get(study, {})
        notes = [e.get("provisional") for e in block.get("extra") or []]
        return " ".join(dict.fromkeys(n for n in notes if n))

    def ebola_extra(self) -> list[str]:
        """Further Ebola samples, each kept to its models."""
        return [
            f"{self.results}/evidence/ebola-extra-{i}.csv.gz"
            for i in range(len(self.ebola.get("extra") or []))
        ]

    @staticmethod
    def extra_args(entry: Mapping[str, Any]) -> str:
        """Return the `germany-extra` options of a `germany.extra` entry."""
        args = []
        if entry.get("models"):
            args.append(f"--models {shlex.quote(entry['models'])}")
        if entry.get("suffix"):
            args.append(f"--suffix {shlex.quote(entry['suffix'])}")
        if entry.get("backtest"):
            args.append("--backtest")
        return " ".join(args)

    def germany_tables(self, mode: str) -> list[str]:
        """Return the German benchmark's paper tables a run reads."""
        if not self.germany["real"]:
            return []
        results = self.germany["results"]
        halves = [""] if mode == "development" else ["", "held-out/"]
        tables = [f"{results}/{h}paper-forecasts.csv.gz" for h in halves]
        extra = [
            f"{self.results}/evidence/germany-extra-{i}.csv.gz"
            for i in range(len(self.germany.get("extra") or []))
        ]
        return tables + extra

    def synthetic_models_arg(self) -> str:
        """Return the synthetic runner's models option."""
        if self.config.get("finetuned"):
            return ""
        from paperdata.synthetic_runner import MODELS

        return "--models " + " ".join(m for m in MODELS if "_ft" not in m)

    def synthetic_simulations(self, mode: str) -> list[str]:
        """Return a run's simulation files, or [] if any is missing."""
        folder = self.synthetic["simulations"][mode]
        seeds = run_settings(self.config, mode).synthetic_seeds
        files = [
            f"{folder}/{kind}-{seed}.{ext}"
            for seed in seeds
            for kind, ext in (("counts", "csv.gz"), ("settings", "csv"))
        ]
        return files if all(Path(f).exists() for f in files) else []

    def example_files(self) -> dict[str, list[str]]:
        """Return the SI examples' simulation files, by kind.

        The examples are picked from the workflow's validation round
        (`grid_rounds.validation_round` in `rounds_dir`), which the
        simulate rule builds, so the held-out round is never read for them.
        Each pick names a simulation, `<seed>-<n>`, and the files are its
        seed's. With no rounds folder there are none, and the figure says
        what it waits on.
        """
        if self.rounds_dir is None:
            return {"counts": [], "settings": []}
        round_ = self.config["grid_rounds"]["validation_round"]
        folder = f"{self.rounds_dir}/r{round_}/sims"
        picks = self.synthetic["examples"]["picks"]
        seeds = sorted({str(sim).split("-")[0] for sim, _ in picks})
        return {
            "counts": [f"{folder}/counts-{s}.csv.gz" for s in seeds],
            "settings": [f"{folder}/settings-{s}.csv" for s in seeds],
        }

    # A study's data ------------------------------------------------------

    def sources(self, w: Wildcards) -> str:
        """Return a study's real data, as command line arguments."""
        if w.study == "synthetic":
            fc = self.synthetic["forecasts"][w.mode]
            return f"--synthetic-forecasts {shlex.quote(fc)}" if fc else ""
        if w.study == "germany" and self.germany["real"]:
            tables = " ".join(self.germany_tables(w.mode))
            return (
                f"--germany-forecasts {tables} "
                f"--germany-triangle {self.triangle}"
            )
        ebola = self.ebola
        if w.study == "ebola" and ebola["hub"]:
            args = [
                f"--hub {shlex.quote(ebola['hub'])}",
                f"--ebola-data {shlex.quote(ebola['data'])}",
                f"--ebola-config {shlex.quote(ebola['config'])}",
                f"--ebola-backtest {shlex.quote(ebola['backtest'])}",
            ]
            if ebola.get("extra"):
                files = " ".join(self.ebola_extra())
                args.append(f"--ebola-extra {files}")
            if ebola.get("n_samples"):
                models = " ".join(ebola["backtest_models"])
                args.append(f"--ebola-draws {self.ebola_draws}")
                args.append(f"--ebola-draws-models {models}")
            if ebola["candidates"]:
                files = " ".join(shlex.quote(c) for c in ebola["candidates"])
                args.append(f"--candidates {files}")
            history = self.store_history()
            if history:
                args.append(f"--store-history {history}")
            return " ".join(args)
        return ""

    def store_history(self) -> str | None:
        """Return the fit store's history file, fetched from its release."""
        if not self.ebola.get("store_history"):
            return None
        return f"{self.results}/evidence/fit-store-history.csv.gz"

    def source_files(self, w: Wildcards) -> list[str]:
        """Files a study's real data come from, built by other rules."""
        if w.study == "germany":
            triangle = [self.triangle] if self.germany["real"] else []
            return [*self.germany_tables(w.mode), *triangle]
        ebola = self.ebola
        if w.study == "ebola" and ebola["hub"]:
            hub = f"{ebola['hub']}/target-data/time-series.csv"
            target = f"{ebola['data']}/target.csv"
            draws = [self.ebola_draws] if ebola.get("n_samples") else []
            history = self.store_history()
            stored = [history] if history else []
            return [
                hub,
                target,
                ebola["backtest"],
                *self.ebola_extra(),
                *draws,
                *stored,
            ]
        if w.study == "synthetic" and self.synthetic_simulations(w.mode):
            return [self.synthetic["forecasts"][w.mode]]
        return []

    def limits(self, w: Wildcards) -> str:
        """Return the data limits of a study's loader, as arguments."""
        s = run_settings(self.config, w.mode)
        if w.study == "synthetic":
            return "--seeds " + " ".join(str(x) for x in s.synthetic_seeds)
        if w.study == "germany" and s.germany_before is not None:
            return f"--before {s.germany_before:%Y-%m-%d}"
        if w.study == "ebola" and s.ebola_before is not None:
            return f"--before {s.ebola_before:%Y-%m-%d}"
        return self.study_split(w.mode, w.study)

    def study_split(self, mode: str, study: str) -> str:
        """Return the date a study's scores are split at, as an argument."""
        s = run_settings(self.config, mode)
        if study == "germany" and s.germany_split is not None:
            return f"--split {s.germany_split:%Y-%m-%d}"
        if study == "ebola" and s.ebola_split is not None:
            return f"--split {s.ebola_split:%Y-%m-%d}"
        return ""

    def delay_arg(self, w: Wildcards) -> str:
        """Return the days after a target date its truth is taken."""
        delays = self.config.get("truth_delay", {})
        return f"--delay {delays[w.study]}" if w.study in delays else ""

    def note_arg(self, w: Wildcards) -> str:
        """Return what is provisional in a study's forecasts, as an argument."""
        note = self.provisional(w.study)
        return f"--note {shlex.quote(note)}" if note else ""

    def truth_rule_arg(self, w: Wildcards) -> str:
        """Return `--first-after` for a study whose truth is a later release."""
        first_after = self.config.get("truth_first_after", [])
        return "--first-after" if w.study in first_after else ""

    def study_files(self, study: str) -> dict[str, str]:
        """Return a study's files, with `{mode}` left open."""
        r = self.results
        return {
            "forecasts": f"{r}/{{mode}}/forecasts/{study}.csv.gz",
            "observations": f"{r}/{{mode}}/data/{study}-observations.csv.gz",
            "locations": f"{r}/{{mode}}/data/{study}-locations.csv",
            "meta": f"{r}/{{mode}}/data/{study}-meta.json",
            "scores": f"{r}/{{mode}}/scores/{study}.csv.gz",
            "crps_scores": f"{r}/{{mode}}/scores-crps/{study}.csv.gz",
            "truth": f"{r}/{{mode}}/truth/{study}.csv.gz",
        }

    # A run's outputs -----------------------------------------------------

    def figures(self, mode: str, study: str) -> list[str]:
        """Return a study's main and SI figures."""
        return [
            f"{self.results}/{mode}/figures/{part}/{study}-{name}.png"
            for part in ("main", "si")
            for name in figure_names(study, part)
        ]

    def tables(self, mode: str, study: str) -> list[str]:
        """Return a study's main and SI score tables."""
        r = self.results
        main = [*MAIN_TABLES, *(["grid"] if study == "synthetic" else [])]
        si = (*SI_TABLES, *STUDY_SI_TABLES.get(study, ()))
        return [f"{r}/{mode}/tables/main/{study}-{n}.csv" for n in main] + [
            f"{r}/{mode}/tables/si/{study}-{n}.csv" for n in si
        ]

    def outputs(self, w: Wildcards) -> list[str]:
        """Every figure and table a run publishes."""
        r, mode, studies = self.results, w.mode, self.studies(w.mode)
        design = [f"{r}/{mode}/figures/si/sim-examples.png"] + [
            f"{r}/{mode}/tables/si/{n}.md" for n in DESIGN_TABLES
        ]
        joint = [
            f"{r}/{mode}/tables/main/{study}-joint.md"
            for study in studies
            if study in JOINT_STUDIES
        ]
        real = (
            [f"{r}/{mode}/tables/si/real-finetune.md"]
            if "germany" in studies
            else []
        )
        per_study = [
            path
            for study in studies
            for path in [*self.figures(mode, study), *self.tables(mode, study)]
        ]
        return [f"{r}/{mode}/evidence.yml", *design, *joint, *real, *per_study]

    def ebola_real_finetune(self, w: Wildcards) -> list[str]:
        """Return the Ebola real-data table, which development runs lack."""
        return [
            f"{self.results}/{w.mode}/tables/si/ebola-real-finetune.csv"
            for s in self.studies(w.mode)
            if s == "ebola"
        ]

    def summary_meta(self, w: Wildcards) -> list[str]:
        """Each study's data notes of a run."""
        return [
            self.study_files(s)["meta"].format(mode=w.mode)
            for s in self.studies(w.mode)
        ]

    def summary_scores(self, w: Wildcards) -> list[str]:
        """Each study's main score table of a run."""
        return [
            f"{self.results}/{w.mode}/tables/main/{s}-scores.csv"
            for s in self.studies(w.mode)
        ]

    def summary_args(self) -> str:
        """Return the `paperpipe summary` options the config sets."""
        finetuned = self.config.get("finetuned")
        ft = f"--finetuned {finetuned}" if finetuned else ""
        return f"{ft} --realdata-after {self.realdata_after}"

    def held_out_placeholders(self, w: Wildcards) -> str:
        """Return the included tables of the studies a run leaves out."""
        return " ".join(
            f"{self.build}/tables/{part}/{study}-{name}.md"
            for study in self.config["studies"]
            if study not in self.studies(w.mode)
            for part, name in HELD_OUT_INCLUDES
        )

    def grid_config(self) -> dict[str, Any]:
        """Return the `synthetic` block with both runs' seeds."""
        paper = run_settings(self.config, "paper")
        development = run_settings(self.config, "development")
        return {
            **self.synthetic,
            "heldout_seeds": list(paper.synthetic_seeds),
            "development_seeds": list(development.synthetic_seeds),
        }


PYTHON: Final = f"PYTHONPATH=paper {shlex.quote(sys.executable)}"
"""Each step's Python: the one running the workflow, with paper/ on its path."""

BDBV: Final = shlex.quote(str(Path(sys.executable).parent / "bdbvhubrenewal"))
"""The project's command line, beside that Python."""
