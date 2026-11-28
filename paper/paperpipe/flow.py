"""The paper workflow's paths and command line arguments.

The paper's rules (`paper/workflow/Snakefile`) take their inputs and
arguments from a `PaperFlow`, so the rules hold no logic of their own.
`PaperFlow.load` reads the workflow config once, when the workflow is
built. Its methods that take `w` are Snakemake input and params
functions: `w` holds the rule's wildcards, `mode` and `study`.
"""

from __future__ import annotations

import shlex
import sys
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

from nowcastfoundation.forecaster import FINETUNED_ENV
from nowcastgermany.checkpoint import GERMANY_UNIT, REFERENCE

from paperpipe.config import run_settings, run_studies
from paperpipe.outputs import (
    JOINT_STUDIES,
    MAIN_TABLES,
    SI_TABLES,
    STUDY_SI_TABLES,
    figure_names,
)
from paperpipe.stored import PROGRESS_TABLES

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
    "grid",
    "epinowcast-model",
    "heldout-set",
)
"""The SI tables of the synthetic design (`papertables.synthetic_grid`)."""


class Wildcards(Protocol):
    """The wildcards of a paper rule."""

    mode: str
    study: str


def _prefixes(config: Mapping[str, Any]) -> list[int]:
    """Return the final run's checkpoint prefixes.

    The run's own `prefixes` in the workflow's `runs`, else `finetune`'s,
    which they are with no final run.
    """
    name = Path(config["final_run"]).name if config.get("final_run") else ""
    entry = (config.get("runs") or {}).get(name) or {}
    return [
        int(n) for n in entry.get("prefixes") or config["finetune"]["prefixes"]
    ]


def _final_prefix(config: Mapping[str, Any]) -> int:
    """Return the final run's last checkpoint prefix.

    Its final checkpoint, where the learning rate reaches zero.
    """
    return _prefixes(config)[-1]


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
        rounds' folders and evidence `file`s. `finetuned_record` is the
        release record (`published.json`, rule publish_checkpoint) of the
        final run's final checkpoint, its last prefix, which every rule
        running an `_ft` model takes as an input.

        Parameters
        ----------
        config : dict
            The workflow config.
        evidence : mapping
            Each evidence entry that is set, with its `ref` or `file`.
        root : pathlib.Path
            The repository root, holding
            `packages/nowcastfoundation/config/finetune.toml`.
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
        # A null folder leaves that run's simulations out.
        synthetic["simulations"] = {
            mode: expand(folder) if folder else None
            for mode, folder in synthetic["simulations"].items()
        }
        # Null leaves the run or rounds out, and what reads them pending.
        for key in ("final_run", "rounds_dir", "runs_dir"):
            if config.get(key):
                config[key] = expand(config[key])
        germany["units"] = expand(germany["units"])
        # The `_ft` models load the final run's final checkpoint, a rule
        # input, so each new one reruns their forecasts.
        config["finetuned_record"] = (
            f"{config['final_run']}/train/n{_final_prefix(config)}"
            "/published.json"
            if config.get("final_run")
            else None
        )
        with (
            root / "packages" / "nowcastfoundation" / "config" / "finetune.toml"
        ).open("rb") as f:
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
        """Return the evidence step's development inputs from workflow rules.

        The published-range table's rows (rule sim_diseases) and the realism
        step's outputs (rule sim_realism), or their stored copies. Each is
        left out, and its SI result pending, while the rounds folder is
        null and nothing is stored.
        """
        files = {}
        evidence = f"{self.results}/evidence"
        stored = self.stored
        if where := self.rounds_dir:
            files["sim_diseases"] = f"{where}/report/sim-diseases.csv"
            files["sim_summaries"] = f"{evidence}/sim-summaries.csv"
            files["sim_realism"] = f"{evidence}/sim-realism.csv"
        else:
            for name in ("sim_diseases", "sim_summaries", "sim_realism"):
                if stored.get(name):
                    files[name] = stored[name]
        return files

    def ebola_files(self) -> dict[str, str]:
        """Return the paper run's Ebola analyses, by evidence name.

        The walks' pairing scores, summarised, and their rolling rho_p on
        the Ebola release backtest (rule pairing_ebola), and Chronos-2's
        trajectories under each coupling (rule coupling_ebola), and the
        renewal variants compared (rule renewal_variants).
        """
        evidence = f"{self.results}/paper/evidence"
        return {
            "si_copula": f"{evidence}/si-copula.csv",
            "copula_rho": f"{evidence}/pairing/pairing-bdbv-rho.csv.gz",
            "coupling": f"{evidence}/coupling.csv",
            "renewal_variants": f"{evidence}/renewal-variants.csv",
        }

    def ebola_evidence(self, mode: str) -> dict[str, str]:
        """Return the evidence step's Ebola inputs, a paper run's only.

        A development run holds the Ebola study out, so it has none.
        """
        if mode != "paper" or "ebola" not in self.studies(mode):
            return {}
        return self.ebola_files()

    @property
    def stored(self) -> Mapping[str, str]:
        """Host-made outputs read as files in place of their rules.

        The `stored` block, which the Site build sets from the paper-*
        releases (`paperpipe.stored`); empty on the workflow's hosts.
        """
        return dict(self.config.get("stored") or {})

    @property
    def host_outputs(self) -> bool:
        """Whether this run makes the hosts' outputs (`host_outputs`).

        Pages sets it false: the German development table is read from
        the stored copy, never made.
        """
        return bool(self.config.get("host_outputs", True))

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

    # Each run's progress ----------------------------------------------

    def progress_runs(self) -> list[str]:
        """Return the runs with a progress figure.

        The live run and each comparison's run.
        """
        runs = [
            self.config.get("live_run"),
            *(self.config.get("comparisons") or {}).values(),
        ]
        return list(dict.fromkeys(r for r in runs if r))

    def progress_files(self, run: str) -> dict[str, str]:
        """Return a run's progress tables: made, stored or none.

        With `runs_dir`, the run's own reports from the workflow's rules
        run_progress (losses, the 7-day check and the BDBV check) and
        horizon_table (German relative scores by horizon), of the
        checkpoints it has saved. Without it, the copies the hosts
        published (`paperpipe.stored`), those that exist, which leave out
        the BDBV check. With neither, none, and the figure's panels say so.
        """
        if runs_dir := self.config.get("runs_dir"):
            where = f"{runs_dir}/{run}/reports"
            return {
                "loss": f"{where}/progress/loss.csv",
                "seven_day": f"{where}/progress/seven-day.csv",
                "bdbv": f"{where}/progress/bdbv.csv",
                "horizons": f"{where}/horizon-scores.csv",
            }
        stored = self.stored.get("progress")
        if not stored:
            return {}
        files = {
            key: f"{stored}/{run}/{Path(table).name}"
            for key, table in PROGRESS_TABLES.items()
        }
        return {k: f for k, f in files.items() if Path(f).is_file()}

    def progress_figure(self, mode: str, run: str) -> str:
        """Return a run's progress figure."""
        return f"{self.results}/{mode}/figures/si/progress-{run}.png"

    @staticmethod
    def progress_args(mode: str, files: Mapping[str, str]) -> str:
        """Return the progress step's table options.

        The BDBV check is a paper run's only, as the Ebola study is held
        out of development runs.
        """
        return " ".join(
            f"--{key.replace('_', '-')} {path}"
            for key, path in files.items()
            if key != "bdbv" or mode == "paper"
        )

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
        # Pages reads the host's development table (`paperpipe.stored`).
        if not self.host_outputs and self.stored.get("germany_forecasts"):
            tables[0] = self.stored["germany_forecasts"]
        extra = [
            f"{self.results}/evidence/germany-extra-{i}.csv.gz"
            for i in range(len(self.germany.get("extra") or []))
        ]
        return tables + extra

    def germany_units(self) -> list[str]:
        """Return the German units our development table stacks.

        The reference unit (randomwalknowcast and zero-shot Chronos-2,
        rule score_germany) and the final run's final checkpoint's unit
        (rule score_run_germany), the checkpoint the `_ft` models load;
        the reference alone without a final run. Their done stamps are
        the rule's inputs.
        """
        units = [f"{self.germany['units']}/{REFERENCE}"]
        if run := self.config.get("final_run"):
            n = _final_prefix(self.config)
            units.append(f"{run}/scores/{GERMANY_UNIT}/n{n}")
        return units

    def finetuned_inputs(self) -> list[str]:
        """Return the final checkpoint's release record, if there is a run.

        Only the final run's own checkpoint, so no other run is an input.
        """
        record = self.config.get("finetuned_record")
        return [record] if record else []

    def finetuned_env(self) -> str:
        """Return the environment setting that loads the final checkpoint.

        Its segment folder as `BDBV_CHRONOS_FT`, which every `_ft` model
        loads; empty while the fine-tuned models are pending.
        """
        record = self.config.get("finetuned_record")
        if not record:
            return ""
        return f"{FINETUNED_ENV}={Path(record).parent}"

    def ebola_draws_models(self) -> list[str]:
        """Return the models the paper's own Ebola backtest runs.

        `backtest_models`, less the fine-tuned ones while the final run
        has no checkpoint, as for the synthetic runner. `foundation` is
        not among them: it is a stored model, so `bdbvhub backtest` never
        runs it and its forecasts are the fit store's.
        """
        models = list(self.ebola["backtest_models"])
        if self.config.get("finetuned_record"):
            return models
        from nowcasteval.labels import FINETUNED

        return [m for m in models if m not in FINETUNED]

    def ebola_site_models(self) -> list[str]:
        """Return the models the site's release backtest file is made with.

        The walks only: the paper's own backtest (`ebola_draws_models`)
        replaces every model it runs, so the Chronos-2 ones run there
        alone.
        """
        from nowcasteval.labels import FINETUNED

        chronos = {*FINETUNED, *FINETUNED.values()}
        return [m for m in self.ebola["backtest_models"] if m not in chronos]

    def synthetic_models_arg(self) -> str:
        """Return the synthetic runner's models option."""
        if self.config.get("finetuned_record"):
            return ""
        from nowcasteval.labels import FINETUNED

        from paperdata.synthetic_runner import MODELS

        return "--models " + " ".join(m for m in MODELS if m not in FINETUNED)

    def synthetic_simulations(self, mode: str) -> list[str]:
        """Return a run's simulation files and its round's READY file.

        The simulate rules make whatever is missing, so a run asks for its
        round through the DAG rather than skipping it.
        """
        folder = self.synthetic["simulations"][mode]
        seeds = run_settings(self.config, mode).synthetic_seeds
        files = [
            f"{folder}/{kind}-{seed}.{ext}"
            for seed in seeds
            for kind, ext in (("counts", "csv.gz"), ("settings", "csv"))
        ]
        return [*files, f"{Path(folder).parent}/READY"]

    def example_files(self) -> dict[str, list[str]]:
        """Return the SI examples' simulation files, by kind.

        The examples are picked from the workflow's validation round
        (`grid_rounds.validation_round` in `rounds_dir`), which the
        simulate rule builds, so the held-out round is never read for them.
        Each pick names a simulation, `<seed>-<n>`, and the files are its
        seed's. With no rounds folder they are the stored examples, and
        with neither there are none, and the figure says what it waits on.
        """
        if self.rounds_dir is not None:
            round_ = self.config["grid_rounds"]["validation_round"]
            folder = f"{self.rounds_dir}/r{round_}/sims"
        elif self.stored.get("examples"):
            folder = self.stored["examples"]
        else:
            return {"counts": [], "settings": []}
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
                models = " ".join(self.ebola_draws_models())
                args.append(f"--ebola-draws {self.ebola_draws}")
                args.append(f"--ebola-draws-models {models}")
            releases = self.round_releases()
            if releases:
                args.append(f"--round-releases {releases}")
            return " ".join(args)
        return ""

    def round_releases(self) -> str | None:
        """Return our unsubmitted models' forecasts from the round releases."""
        if not self.ebola.get("round_releases"):
            return None
        return f"{self.results}/evidence/round-releases.csv.gz"

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
            releases = self.round_releases()
            released = [releases] if releases else []
            return [
                hub,
                target,
                ebola["backtest"],
                *self.ebola_extra(),
                *draws,
                *released,
            ]
        if w.study == "synthetic":
            fc = self.synthetic["forecasts"][w.mode]
            return [fc] if fc else []
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
        progress = [
            self.progress_figure(mode, run) for run in self.progress_runs()
        ]
        return [
            f"{r}/{mode}/evidence.yml",
            *design,
            *joint,
            *real,
            *progress,
            *per_study,
        ]

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
        record = self.config.get("finetuned_record")
        ft = f"--finetuned-record {record}" if record else ""
        prefixes = " ".join(str(n) for n in _prefixes(self.config))
        return (
            f"{ft} --realdata-after {self.realdata_after} --prefixes {prefixes}"
        )

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

BDBV: Final = shlex.quote(str(Path(sys.executable).parent / "bdbvhub"))
"""The project's command line, beside that Python."""

GERMANY_CLI: Final = shlex.quote(
    str(Path(sys.executable).parent / "nowcastgermany")
)
"""The German benchmark's command line (`nowcastgermany.cli`)."""
