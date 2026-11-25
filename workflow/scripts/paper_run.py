"""Run the paper on this machine, niced and reported (`task paper`).

    python workflow/scripts/paper_run.py [development|paper]
    python workflow/scripts/paper_run.py --targets TARGET...
    python3 workflow/scripts/paper_run.py --render [development|paper] [--pages]

The first form runs the third (`render`, the paper workflow and render,
as Pages CI does) at nice 19 with this machine's cores less its headroom
(`flowlib.machine.budget`). The workflow asks for every input it needs,
the held-out simulations and German forecasts included, and makes what is
missing. The paper workflow loads the final fine-tuning run's newest
checkpoint itself (`paperpipe.flow`).
The second form runs those targets of the paper workflow alone. Each run
logs to PAPER_LOGS and posts start, done or FAILED with the log's first
error line (mailbox lines when NOTIFY is set, `flowlib.notify`). The
third form is the one command Pages CI runs for the paper (`render`),
in the foreground with no log or notes. `--pages` adds the Pages config
(paper/workflow/pages.yaml), which leaves out what the workflow's hosts
make: simulations, fine-tuning runs and their scores.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "workflow"))

from flowlib import localconfig, machine, monitor, notify  # noqa: E402

RENDERS = ("development", "paper")

PAPER_OUTPUTS = (
    "main.html",
    "main.pdf",
    "main_files",
    "si.html",
    "si.pdf",
    "si_files",
)
"""The rendered paper's files and folders, copied into docs/paper/."""

PAGES_CONFIG = "paper/workflow/pages.yaml"
"""The paper workflow's settings on Pages CI, over its config."""


def render(target: str, cfg: Mapping[str, str], *, pages: bool = False) -> int:
    """Run the paper's workflow for TARGET and copy the paper into docs/.

    `development` never reads the held-out data; `paper` needs it
    (paper/README.md). CORES (environment or config/local.env) sets
    Snakemake's cores (default 4). Snakemake runs the target's missing steps
    (`--rerun-triggers mtime` reuses the backtest forecasts CI has
    already made) and renders main.qmd and si.qmd, which are copied into
    docs/paper/ for the site. The chronos group is installed too, as the
    German rule runs the Chronos-2 models. Needs uv, git and Quarto, and a
    GitHub token (GITHUB_TOKEN, GH_TOKEN or a gh login). With `pages`
    the workflow also reads PAGES_CONFIG.
    """
    if target not in RENDERS:
        print(
            f"Unknown target '{target}': give development or paper.",
            file=sys.stderr,
        )
        return 2
    env = dict(os.environ)
    # The data steps call the GitHub API, which rate limits anonymous
    # calls. CI sets GITHUB_TOKEN; locally take GH_TOKEN or gh's login.
    if not env.get("GITHUB_TOKEN"):
        token = env.get("GH_TOKEN", "")
        if not token and shutil.which("gh"):
            token = subprocess.run(
                ["gh", "auth", "token"], capture_output=True, text=True
            ).stdout.strip()
        env["GITHUB_TOKEN"] = token
    cores = env.get("CORES") or cfg.get("CORES") or "4"
    subprocess.run(
        [
            "uv", "run", "--group", "paper", "--group", "chronos",
            "snakemake", "-s", "workflow/Snakefile", "render",
            "--config", f"render={target}",
            *(["--configfile", PAGES_CONFIG] if pages else []),
            "--cores", cores,
            "--resources", "heavy=1", "--rerun-triggers", "mtime",
            *monitor.args(cfg),
        ],
        cwd=ROOT, env=env, check=True,
    )  # fmt: skip
    paper, docs = ROOT / "paper", ROOT / "docs" / "paper"
    (docs / "figures").mkdir(parents=True, exist_ok=True)
    for name in PAPER_OUTPUTS:
        src = paper / name
        if src.is_dir():
            shutil.copytree(src, docs / name, dirs_exist_ok=True)
        else:
            shutil.copy2(src, docs / name)
    for pattern in ("*.svg", "*.pdf", "*.png"):
        for f in (paper / "figures").glob(pattern):
            shutil.copy2(f, docs / "figures" / f.name)
    shutil.copytree(
        paper / "build" / "figures",
        docs / "build" / "figures",
        dirs_exist_ok=True,
    )
    return 0


def say(text: str, cfg: dict[str, str]) -> None:
    """Print a line and post it as a paper line."""
    print(text, flush=True)
    notify.post(f"paper: {text}", cfg)


def first_error(log: Path) -> str:
    """Return the log's first error line, cut short."""
    for line in log.read_text(errors="replace").splitlines():
        if re.search(r"error|exception", line, re.I):
            return line.strip()[:250]
    return ""


def main(argv: list[str]) -> int:
    """Run one paper job; return its exit code."""
    cfg = localconfig.read()
    if argv[:1] == ["--render"]:
        rest = [a for a in argv[1:] if a != "--pages"]
        target = rest[0] if rest else "development"
        return render(target, cfg, pages="--pages" in argv[1:])
    cores = str(machine.budget(cfg)["cores"])
    if argv[:1] == ["--targets"]:
        name, targets = "targets", argv[1:]
        cmd = [
            "uv",
            "run",
            "--group",
            "paper",
            "--group",
            "chronos",
            "snakemake",
            "-s",
            "workflow/Snakefile",
            *targets,
            "--cores",
            cores,
            "--resources",
            "heavy=1",
            "--rerun-triggers",
            "mtime",
            *monitor.args(cfg),
        ]
        where = ROOT
    else:
        name = argv[0] if argv else "development"
        if name not in RENDERS:
            raise SystemExit(f"give one of {RENDERS} or --targets, not {name}")
        cmd = [sys.executable, str(Path(__file__).resolve()), "--render", name]
        where = ROOT
    logs = Path(cfg["PAPER_LOGS"]).expanduser()
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / f"paper-{name}.log"
    say(f"started {name} (log {log})", cfg)
    with log.open("w") as out:
        rc = subprocess.run(
            ["nice", "-n", "19", *cmd],
            cwd=where,
            env={**os.environ, "CORES": cores},
            stdout=out,
            stderr=subprocess.STDOUT,
        ).returncode
    if rc:
        say(f"FAILED {name} (exit {rc}): {first_error(log)}", cfg)
    else:
        say(f"done {name}", cfg)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
