"""The Snakemake DAG stays valid and the development target runs."""

import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml
from nowcasteval.labels import FINETUNED

from paperpipe.config import last_release, run_studies

pytest.importorskip("snakemake")

PAPER = Path(__file__).parents[1]
# The paper is a module of the project workflow, run from the root.
SNAKEFILE = PAPER.parent / "workflow" / "Snakefile"
CONFIG = yaml.safe_load((PAPER / "workflow" / "config.yaml").read_text())
WORKFLOW = yaml.safe_load(
    (PAPER.parent / "workflow" / "config.yaml").read_text()
)
RULES = (
    "data",
    "publish_paper",
    "summary",
    "forecasts",
    "table",
    "truth",
    "scores",
    "figures",
    "score_tables",
)


def _offline(tmp: Path, **extra) -> Path:
    """A config that runs every study on dummy data, writing under `tmp`."""
    path = tmp / "offline.yaml"
    cfg = {
        "results": str(tmp / "results"),
        "build": str(tmp / "build"),
        "germany": {"real": False},
        "ebola": {"hub": None},
        # Dummy synthetic data: no simulations are made for the run.
        "synthetic": {
            **CONFIG["synthetic"],
            "forecasts": {"development": None, "paper": None},
        },
        # The workflow's runs and rounds are left out, as each is pending.
        "final_run": None,
        "rounds_dir": None,
        "runs_dir": None,
        "evidence": None,
        **extra,
    }
    path.write_text(yaml.safe_dump(cfg))
    return path


def _snakemake(*args: str) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "snakemake", "-s", str(SNAKEFILE), *args]
    return subprocess.run(
        cmd, cwd=PAPER.parent, capture_output=True, text=True, check=False
    )


@pytest.mark.parametrize("target", ["development", "paper"])
def test_dry_run(target: str, tmp_path: Path) -> None:
    res = _snakemake(
        target,
        "-n",
        "--cores",
        "1",
        "--configfile",
        str(PAPER / "workflow" / "config.yaml"),
        str(_offline(tmp_path)),
    )
    assert res.returncode == 0, res.stderr
    log = res.stdout + res.stderr
    for rule in RULES:
        assert f"rule {rule}" in log


@pytest.fixture(scope="module")
def development(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("run")
    res = _snakemake(
        "development",
        "--cores",
        "2",
        "--configfile",
        str(PAPER / "workflow" / "config.yaml"),
        str(_offline(tmp, n_samples=20)),
    )
    assert res.returncode == 0, res.stderr
    return tmp / "results" / "development"


DEV_STUDIES = run_studies(CONFIG["studies"], "development")


def test_development_writes_main_and_si_outputs(development: Path) -> None:
    for study in DEV_STUDIES:
        assert (
            development / "figures" / "main" / f"{study}-results.png"
        ).is_file()
        assert (
            development / "tables" / "main" / f"{study}-scores.csv"
        ).is_file()
        assert list((development / "figures" / "si").glob(f"{study}-*.png"))
        assert list((development / "tables" / "si").glob(f"{study}-*.csv"))
    assert (development / "tables" / "main" / "synthetic-grid.csv").is_file()
    joint = development / "tables" / "main" / "synthetic-joint.md"
    assert "rel_energy_log" in joint.read_text()
    design = development / "tables" / "si" / "grid.md"
    assert "Infections" in design.read_text()
    assert (development / "figures" / "si" / "sim-examples.png").is_file()


def test_development_never_reads_heldout_data(development: Path) -> None:
    split = pd.Timestamp(CONFIG["heldout"]["germany_from"])
    germany = pd.read_csv(
        development / "forecasts" / "germany.csv.gz",
        parse_dates=["origin_date"],
    )
    assert germany["origin_date"].max() < split
    obs = pd.read_csv(
        development / "data" / "germany-observations.csv.gz",
        parse_dates=["target_date", "as_of_date"],
    )
    assert obs["target_date"].max() < split
    delay = CONFIG["truth_delay"]["germany"]
    assert obs["as_of_date"].max() <= last_release(split, delay)
    synthetic = pd.read_csv(development / "forecasts" / "synthetic.csv.gz")
    held = tuple(f"s{s}-" for s in CONFIG["heldout"]["synthetic_seeds"])
    assert not synthetic["location"].str.startswith(held).any()
    # The Ebola study is held out entirely.
    assert DEV_STUDIES == ("synthetic", "germany")
    assert not list(development.rglob("ebola*"))


def test_development_scores_have_truth(development: Path) -> None:
    for study in DEV_STUDIES:
        scores = pd.read_csv(development / "scores" / f"{study}.csv.gz")
        assert len(scores) > 0
        assert scores["score"].notna().all()


def test_development_publishes_for_the_paper(development: Path) -> None:
    build = development.parents[1] / "build"
    assert (build / "figures" / "main" / "germany-results.png").is_file()
    assert (build / "tables" / "main" / "germany-scores.md").is_file()
    results = yaml.safe_load((build / "results.yml").read_text())["results"]
    assert results["mode"] == "development"
    assert results["germany"]["source"] == "dummy"
    assert results["germany"]["label"]
    # No Ebola results, and placeholders for the tables the paper
    # includes inside the hidden Ebola sections.
    assert "ebola" not in results
    held = build / "tables" / "main" / "ebola-scores.md"
    assert "Held out" in held.read_text()


def test_dry_run_with_real_data_runs_the_benchmark(tmp_path: Path) -> None:
    cfg = tmp_path / "real.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "results": str(tmp_path / "results"),
                "build": str(tmp_path / "build"),
                "germany": {
                    "data": str(tmp_path / "germany"),
                    "results": str(tmp_path / "germany-results"),
                },
                "ebola": {"hub": str(tmp_path / "hub")},
            }
        )
    )
    base = str(PAPER / "workflow" / "config.yaml")
    dev = _snakemake("development", "-n", "--configfile", base, str(cfg))
    assert dev.returncode == 0, dev.stderr
    log = dev.stdout + dev.stderr
    for rule in ("germany_fetch", "germany_development"):
        assert f"rule {rule}" in log
    for rule in ("germany_heldout", "ebola_hub"):
        assert f"rule {rule}" not in log
    paper = _snakemake("paper", "-n", "--configfile", base, str(cfg))
    for rule in ("germany_heldout", "ebola_hub"):
        assert f"rule {rule}" in paper.stdout + paper.stderr


def test_only_the_paper_runs_ebola_scoring_every_round(
    tmp_path: Path,
) -> None:
    first = CONFIG["heldout"]["ebola_from"]
    base = str(PAPER / "workflow" / "config.yaml")
    for target in ("development", "paper"):
        res = _snakemake(
            target, "-n", "-p", "--configfile", base, str(_offline(tmp_path))
        )
        assert res.returncode == 0, res.stderr
        ebola = [
            line
            for line in (res.stdout + res.stderr).splitlines()
            if "paperpipe forecasts ebola" in line
            or "paperpipe figures ebola" in line
        ]
        if target == "development":
            assert not ebola
            continue
        assert ebola
        for line in ebola:
            assert f"--split {first}" in line, line


def test_benchmark_commands_run_from_the_repository_root(
    tmp_path: Path,
) -> None:
    # The models read config/ relative to the working directory
    # (packages/nowcastfoundation/config/finetune.toml); the workflow runs from
    # the root.
    cfg = tmp_path / "real.yaml"
    cfg.write_text(yaml.safe_dump({"results": str(tmp_path / "results")}))
    base = str(PAPER / "workflow" / "config.yaml")
    paper = _snakemake(
        "paper", "-n", "-p", "--forceall", "--configfile", base, str(cfg)
    )
    assert paper.returncode == 0, paper.stderr
    commands = [
        line.strip().removeprefix("Shell command:").strip()
        for line in (paper.stdout + paper.stderr).splitlines()
        if "nowcastgermany backtest" in line
    ]
    assert len(commands) == 2
    for line in commands:
        # After the redirect into the rule's log.
        line = re.sub(r"^exec >\S+ 2>&1; ", "", line)
        assert not line.startswith("cd ")
        assert "--config packages/nowcastgermany/config/germany.toml" in line
        assert "--data data/raw/germany" in line


def test_the_synthetic_runner_runs_on_its_round_s_simulations(
    tmp_path: Path,
) -> None:
    # The runner asks for its round's simulations and READY file through
    # the DAG (the simulate rules make missing ones in a real run).
    sims = tmp_path / "r38" / "sims"
    sims.mkdir(parents=True)
    seeds = CONFIG["development"]["synthetic_seeds"]
    for seed in seeds:
        (sims / f"counts-{seed}.csv.gz").write_text("")
        (sims / f"settings-{seed}.csv").write_text("")
    (sims.parent / "READY").write_text("")
    extra = {
        "synthetic": {
            **CONFIG["synthetic"],
            "simulations": {"development": str(sims), "paper": str(sims)},
            "forecasts": {
                "development": str(tmp_path / "out" / "synthetic-dev.csv.gz"),
                "paper": str(tmp_path / "out" / "synthetic.csv.gz"),
            },
        }
    }
    base = str(PAPER / "workflow" / "config.yaml")
    present = _snakemake(
        "development",
        "-n",
        "-p",
        "--configfile",
        base,
        str(_offline(tmp_path, **extra)),
    )
    assert present.returncode == 0, present.stderr
    log = present.stdout + present.stderr
    assert "rule synthetic_forecasts_development" in log
    assert "--seeds " + " ".join(str(s) for s in seeds) in log


def test_the_paper_runs_its_ebola_models_at_its_own_draws(
    tmp_path: Path,
) -> None:
    cfg = tmp_path / "real.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "results": str(tmp_path / "results"),
                "build": str(tmp_path / "build"),
                "germany": {"real": False},
                "ebola": {**CONFIG["ebola"], "hub": str(tmp_path / "hub")},
            }
        )
    )
    base = str(PAPER / "workflow" / "config.yaml")
    res = _snakemake("paper", "-n", "-p", "--configfile", base, str(cfg))
    assert res.returncode == 0, res.stderr
    log = res.stdout + res.stderr
    assert "rule ebola_backtest_draws" in log
    # The draws of the models the backtest runs itself, `[samples]
    # backtest`; the stored fits keep `[forecast] n_samples`.
    assert f"--samples {CONFIG['ebola']['n_samples']}" in log
    assert CONFIG["ebola"]["n_samples"] == 1000
    # The zero-shot Chronos-2 models run there too, and a fine-tuned one
    # loads the final checkpoint, as in the other studies.
    draws = [ln for ln in log.splitlines() if "backtest --config" in ln]
    assert draws
    for line in draws:
        assert "foundation_zeroshot_baselinenowcast" in line
        if "foundation_baselinenowcast" in line.split():
            assert "BDBV_CHRONOS_FT=" in line


def test_the_workflow_pins_and_records_the_fine_tuned_checkpoint(
    tmp_path: Path,
) -> None:
    base = str(PAPER / "workflow" / "config.yaml")
    sims = tmp_path / "sims"
    sims.mkdir()
    for seed in CONFIG["development"]["synthetic_seeds"]:
        (sims / f"counts-{seed}.csv.gz").write_text("")
        (sims / f"settings-{seed}.csv").write_text("")
    # The round's READY file, which the runner asks for with its sims.
    (sims.parent / "READY").write_text("")
    synthetic = {
        **CONFIG["synthetic"],
        "simulations": {"development": str(sims), "paper": str(sims)},
        "forecasts": {
            "development": str(tmp_path / "out" / "synthetic-dev.csv.gz"),
            "paper": str(tmp_path / "out" / "synthetic.csv.gz"),
        },
    }
    # With no final run the fine-tuned models are pending: not run, and no
    # checkpoint recorded.
    pending = _snakemake(
        "development",
        "-n",
        "-p",
        "--configfile",
        base,
        str(_offline(tmp_path, synthetic=synthetic)),
    )
    assert pending.returncode == 0, pending.stderr
    log = pending.stdout + pending.stderr
    assert "--finetuned-record" not in log
    assert "BDBV_CHRONOS_FT" not in log
    runner = [line for line in log.splitlines() if "run_synthetic.py" in line]
    assert runner
    assert not {w for line in runner for w in line.split()} & set(FINETUNED)
    # With one, they load its final checkpoint, whose release record is
    # the runner's and the summary's input.
    run = tmp_path / "run"
    run.mkdir()
    segment = run / "train" / f"n{WORKFLOW['finetune']['prefixes'][-1]}"
    segment.mkdir(parents=True)
    (segment / "settings.json").write_text('{"hash": "abcdef123456"}')
    (segment / "published.json").write_text('{"hash": "abcdef123456"}')
    pinned = _snakemake(
        "development",
        "-n",
        "-p",
        "--configfile",
        base,
        str(_offline(tmp_path, synthetic=synthetic, final_run=str(run))),
    )
    assert pinned.returncode == 0, pinned.stderr
    log = pinned.stdout + pinned.stderr
    runner = [line for line in log.splitlines() if "run_synthetic.py" in line]
    assert runner
    assert all(f"BDBV_CHRONOS_FT={segment}" in r for r in runner)
    assert f"--finetuned-record {segment}/published.json" in log


def test_rules_that_need_the_network_or_git_run_on_the_driver():
    # On a cluster a non-local rule runs on a node, which may have no
    # network, git or quarto.
    import re
    from pathlib import Path

    text = (Path(__file__).parents[1] / "workflow" / "Snakefile").read_text()
    blocks = re.split(r"\n(?=\s*rule\b)", text)
    for block in blocks:
        shell = block.split("shell:", 1)[-1]
        # `bdbvhub data` downloads the hub's releases.
        pattern = (
            r"git (clone|fetch)|gh release download|quarto"
            r"|\{params\.bdbv\} data"
        )
        if re.search(pattern, shell):
            assert "localrule: True" in block, block.split("\n", 1)[0]


def test_every_forecasting_rule_declares_its_threads():
    # A scheduler gives a job the cores its rule declares; a forecasting
    # rule sets FORECAST.threads in its environment, so it declares them.
    from pathlib import Path

    text = (Path(__file__).parents[1] / "workflow" / "Snakefile").read_text()
    assert text.count("gpu=FORECAST.gpu") == text.count(
        "threads: FORECAST.threads"
    )


def test_the_pages_render_computes_nothing_the_workflow_hosts_make(
    tmp_path: Path,
) -> None:
    # Pages renders with paper/workflow/pages.yaml: the figures, tables
    # and summary over the German benchmark's stored forecasts, never the
    # benchmark, a simulation, a training run or the pairing sensitivity.
    # The fetched data newer than the forecasts would rerun the benchmark
    # if a rule made them.
    (tmp_path / "germany-results").mkdir()
    forecasts = tmp_path / "germany-results" / "paper-forecasts.csv.gz"
    forecasts.write_text("")
    os.utime(forecasts, (1, 1))
    (tmp_path / "germany").mkdir()
    (tmp_path / "germany" / "COVID-19_hospitalizations.csv").write_text("")
    cfg = tmp_path / "real.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "results": str(tmp_path / "results"),
                "build": str(tmp_path / "build"),
                "germany": {
                    "data": str(tmp_path / "germany"),
                    "results": str(tmp_path / "germany-results"),
                },
            }
        )
    )
    base = str(PAPER / "workflow" / "config.yaml")
    pages = str(PAPER / "workflow" / "pages.yaml")
    res = _snakemake(
        "render",
        "-n",
        "--config",
        "render=development",
        "--configfile",
        base,
        str(cfg),
        pages,
    )
    assert res.returncode == 0, res.stderr
    log = res.stdout + res.stderr
    assert "rule publish_paper" in log
    for rule in (
        "germany_fetch",
        "germany_development",
        "simulate_batch",
        "fetch_batch",
        "train_segment",
        "evaluate_checkpoint",
        "synthetic_forecasts_development",
        "pairing_synthetic",
        "pairing_germany",
        "si_copula",
        "sim_realism",
    ):
        assert f"rule {rule}:" not in log


def test_the_pages_render_reads_stored_outputs_as_inputs(
    tmp_path: Path,
) -> None:
    # With the stored overlay the host outputs are plain inputs: no rule
    # makes them, and the synthetic study is real, not dummy.
    store = tmp_path / "stored"
    store.mkdir()
    overlay = {
        "synthetic": {
            "forecasts": {"development": str(store / "synthetic-dev.csv.gz")}
        },
        "stored": {
            "pairing_synthetic": str(store / "pairing.csv.gz"),
            "pairing_germany": str(store / "pairing-germany.csv.gz"),
            "copula_rho": str(store / "rho.csv.gz"),
        },
    }
    for name in (
        "synthetic-dev.csv.gz",
        "pairing.csv.gz",
        "pairing-germany.csv.gz",
        "rho.csv.gz",
    ):
        (store / name).write_text("")
    (tmp_path / "germany-results").mkdir()
    (tmp_path / "germany-results" / "paper-forecasts.csv.gz").write_text("")
    (tmp_path / "germany").mkdir()
    (tmp_path / "germany" / "COVID-19_hospitalizations.csv").write_text("")
    cfg = tmp_path / "real.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "results": str(tmp_path / "results"),
                "build": str(tmp_path / "build"),
                "germany": {
                    "data": str(tmp_path / "germany"),
                    "results": str(tmp_path / "germany-results"),
                },
            }
        )
    )
    (tmp_path / "stored.yaml").write_text(yaml.safe_dump(overlay))
    base = str(PAPER / "workflow" / "config.yaml")
    pages = str(PAPER / "workflow" / "pages.yaml")
    res = _snakemake(
        "render",
        "-n",
        "--config",
        "render=development",
        "--configfile",
        base,
        str(cfg),
        pages,
        str(tmp_path / "stored.yaml"),
    )
    assert res.returncode == 0, res.stderr
    log = res.stdout + res.stderr
    assert "rule si_copula:" in log
    for rule in (
        "simulate_batch",
        "synthetic_forecasts_development",
        "pairing_synthetic",
        "pairing_germany",
        "train_segment",
    ):
        assert f"rule {rule}:" not in log


def test_figures_wait_when_no_shown_model_has_forecasts(tmp_path: Path) -> None:
    # Until the shown models' forecasts are stored, their nowcast panels
    # say what they wait on rather than failing the run.
    nowcasts = {
        **CONFIG["nowcasts"],
        "models": {**CONFIG["nowcasts"]["models"], "germany": ["not_a_model"]},
    }
    res = _snakemake(
        "development",
        "--cores",
        "2",
        "--configfile",
        str(PAPER / "workflow" / "config.yaml"),
        str(_offline(tmp_path, n_samples=20, nowcasts=nowcasts)),
    )
    assert res.returncode == 0, res.stderr
    figures = tmp_path / "results" / "development" / "figures"
    assert (figures / "main" / "germany-nowcasts.png").is_file()
