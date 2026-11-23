"""Machine settings from `config/local.env`.

The file is untracked and holds `KEY=VALUE` lines (see
`config/local.env.example`). Blank lines, comments and lines whose key
is not upper case are skipped, and one pair of surrounding quotes is
dropped. A value may use `$NAME` or `${NAME}` of another setting, in any
order, or of the environment; a line that uses itself (`ROOT=$ROOT/x`)
takes the environment's value, empty if unset. Nothing else of the
shell's syntax is read (no `~`, `${X:-y}` or `$(...)`). A variable set in
the environment wins over the file. With no file everything runs on this
machine. The workflow refuses a file setting something nothing reads
(`check`).

Shell scripts read one setting with
`python3 workflow/flowlib/localconfig.py KEY`.
"""

from __future__ import annotations

import os
import re
import socket
from collections.abc import Mapping
from pathlib import Path
from string import Template

ROOT = Path(__file__).parents[2]
DEFAULT = ROOT / "config" / "local.env"
KEY = re.compile(r"^[A-Z_][A-Z0-9_]*$")
USES = re.compile(r"\$(?:\{([A-Z_][A-Z0-9_]*)\}|([A-Z_][A-Z0-9_]*))")

DEFAULTS = (
    ("HOST_NAME", None),
    ("ROOT", "$HOME/bdbv-rounds"),
    # An optional local file of one-line notes (off unless NOTIFY is set).
    ("MAILBOX", ""),
    # The workflow's own settings.
    ("SCORE_CACHE", "$ROOT/scores"),
    ("JOB_PREFIX", ""),
    # The prefix every simulation batch runs under, with no cap of its
    # own (such as one systemd slice they all share; empty: none).
    ("SIM_PREFIX", ""),
    # The folder simulation batches share their JAX compiles in (empty:
    # none) and the size in GB past which a batch empties it
    # (flowlib.simulator.jax_cache_env).
    ("JAX_CACHE", "$ROOT/jax-cache"),
    ("JAX_CACHE_GB", "2"),
    # The rounds host, as ssh names it (empty: this machine), and its
    # rounds folder (empty: ROOT), where a helping host pushes its
    # batches; the seeds this host simulates (flowlib.hosts; empty: all);
    # and a Snakemake profile of this machine's own (empty:
    # workflow/profiles/default with flowlib.machine's budget).
    ("REMOTE", ""),
    ("REMOTE_ROOT", ""),
    ("SIM_SEEDS", ""),
    ("SNAKEMAKE_PROFILE", ""),
    # This machine's budget for workflow/profiles/default
    # (flowlib.machine): its cores and memory less the headroom kept free,
    # unless CORES and MEM_MB set them; simulation batches at once; and
    # the device (auto: CUDA, MPS, else the CPU).
    ("HEADROOM_CORES", "0"),
    ("HEADROOM_GB", "0"),
    ("CORES", ""),
    ("MEM_MB", ""),
    ("SIM_SLOTS", ""),
    ("DEVICE", "auto"),
    # Series above which a training step recomputes activations in
    # backward (empty: the trainer's memory budget sets it); lower on a
    # small GPU. Memory only, outside the checkpoint's hash.
    ("CHECKPOINT_ABOVE", ""),
    # Mailbox lines (any value turns them on), publishing as it trains,
    # the targets `all` runs here, and where batches come from (empty:
    # workflow/config.yaml's `finetune.sims_from`).
    ("NOTIFY", ""),
    ("PUBLISH", ""),
    ("TARGETS", "simulate finetune score"),
    ("SIMS_FROM", ""),
    # Scheduled runs (workflow/scripts/tick.py): the targets each tick runs
    # (empty: none; flowlib.schedule), every how many minutes, and the timer
    # (cron, systemd or launchd; empty: by platform).
    ("SCHEDULE_TARGETS", ""),
    # Where `hub_rounds` copies the hub rounds report (empty: nowhere).
    ("REVIEW_DIR", ""),
    # The status each tick writes, every how many seconds
    # (flowlib.runstatus).
    ("STATUS_EVERY", "60"),
    # The workflow monitor (flowlib.monitor): the Panoptes server every
    # Snakemake run reports to (empty: off), and where
    # `task nowcast-workflow:monitor` serves it; the targets whose rule
    # graph and job DAG it draws (flowlib.dag, NAME=TARGET), the Snakemake
    # options they are built with, and every how many seconds.
    ("MONITOR_URL", "http://$MONITOR_HOST:$MONITOR_PORT"),
    ("MONITOR_HOST", "127.0.0.1"),
    ("MONITOR_PORT", "8795"),
    ("MONITOR_DAG", ""),
    ("MONITOR_DAG_OPTIONS", ""),
    ("MONITOR_DAG_EVERY", "900"),
    ("SCHEDULE_EVERY", "60"),
    # The coordinator run's pause between ticks, and after a failed one
    # (seconds; workflow/scripts/coordinate.py).
    ("COORDINATE_PAUSE", "60"),
    ("COORDINATE_BACKOFF", "600"),
    ("SCHEDULE_KIND", ""),
    # Where jobs run (flowlib.compute): single (this machine) or ssh
    # (this machine coordinates its WORKERS, name=host:ROOT:CHECKOUT,
    # with rule groups placed on them, group=worker; not GROUPS, which
    # the shells keep for themselves), and the command running
    # Snakemake in a worker's checkout.
    ("COMPUTE", "single"),
    ("WORKERS", ""),
    ("WORKER_GROUPS", ""),
    ("WORKER_SNAKEMAKE", "uv run --group paper snakemake"),
    # Memory and disk a job leaves free when it starts (GB).
    ("MIN_FREE_GB", "20"),
    ("MIN_DISK_GB", "16"),
    # The load average below which a heavy job starts (empty: the core
    # count).
    ("MAX_LOAD", ""),
    # Logs of the paper's detached jobs (paper/scripts).
    ("PAPER_LOGS", "$HOME/paper-jobs"),
)
"""Each setting and its default."""


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]
    return value


OPTIONAL = (
    "BLOCK_LIMIT",
    "CHRONOS_BATCH",
    "FORECAST_DEVICE",
    "FORECAST_THREADS",
    "HF_HOME",
    "MPS_HIGH_WATERMARK",
    "PRECISION",
    "RUN",
    "SCORE_DATES",
)
"""Settings read where set, with no default (`RUN` only to refuse it)."""

KNOWN = frozenset(key for key, _ in DEFAULTS) | frozenset(OPTIONAL)
"""Every setting something reads."""


def lines_of(text: str) -> list[tuple[str, str]]:
    """Return a local config's settings as written, as (key, value) pairs."""
    out = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if sep and KEY.match(key):
            out.append((key, value))
    return out


def _lines(path: Path) -> list[tuple[str, str]]:
    """Return the file's settings as written, as (key, value) pairs."""
    return lines_of(path.read_text()) if path.is_file() else []


def check(
    path: Path | str | None = None, environ: Mapping[str, str] | None = None
) -> None:
    """Raise if the local config sets something nothing reads.

    A setting outside `KNOWN` is allowed only where another line uses it
    (`$NAME` or `${NAME}`), so a mistyped name fails here instead of
    being ignored.

    Parameters
    ----------
    path
        The local config; by default `$LOCAL_CONFIG`, else
        `config/local.env`.
    environ
        The environment; by default `os.environ`.

    Raises
    ------
    ValueError
        Naming each setting nothing reads.
    """
    env = os.environ if environ is None else environ
    if path is None:
        path = env.get("LOCAL_CONFIG") or DEFAULT
    pairs = _lines(Path(path))
    used = {
        m.group(1) or m.group(2)
        for _, value in pairs
        for m in USES.finditer(value)
    }
    unread = sorted({k for k, _ in pairs} - KNOWN - used)
    if unread:
        raise ValueError(
            f"{path} sets {', '.join(unread)}, which nothing reads: "
            "config/local.env.example lists the settings"
        )


def read(
    path: Path | str | None = None, environ: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Return the machine settings.

    Parameters
    ----------
    path
        The local config; by default `$LOCAL_CONFIG`, else
        `config/local.env`.
    environ
        The environment; by default `os.environ`.

    Returns
    -------
    dict
        Every key of the file and of `DEFAULTS`.

    Raises
    ------
    ValueError
        If settings use each other in a loop.
    """
    env = dict(os.environ if environ is None else environ)
    if path is None:
        path = env.get("LOCAL_CONFIG") or DEFAULT
    # Each setting as written, and whether it is filled in: the
    # environment's values and single-quoted ones are kept as they are,
    # as the shell does.
    raw: dict[str, tuple[str, bool]] = {}
    for key, value in _lines(Path(path)):
        single = value.startswith("'") and value.endswith("'")
        raw[key] = (_unquote(value), not single)
    for key, default in DEFAULTS:
        if key not in raw:
            if default is None:
                default = socket.gethostname().split(".")[0]
            raw[key] = (default, True)
    for key in raw:
        if key in env:
            raw[key] = (env[key], False)
    return _resolve(raw, env)


def _resolve(
    raw: dict[str, tuple[str, bool]], env: Mapping[str, str]
) -> dict[str, str]:
    """Fill each setting's `$NAME`s in, so a value may use any other."""
    done: dict[str, str] = {}
    open_: list[str] = []

    class Names(dict):
        def __missing__(self, key: str) -> str:
            return value(key)

    def value(key: str) -> str:
        if key in done:
            return done[key]
        if key not in raw:
            return env[key]
        if open_ and open_[-1] == key:
            # A line that uses itself, as the shell reads it: the
            # environment has no value, or it would have won.
            return env.get(key, "")
        if key in open_:
            loop = " -> ".join([*open_[open_.index(key) :], key])
            raise ValueError(f"local config settings use each other: {loop}")
        text, fill = raw[key]
        open_.append(key)
        if fill:
            text = Template(text).safe_substitute(Names())
        open_.pop()
        done[key] = text
        return text

    return {key: value(key) for key in raw}


def expand(value: str, settings: Mapping[str, str] | None = None) -> str:
    """Return `value` with `$NAME` or `${NAME}` filled from the settings.

    Parameters
    ----------
    value
        A path or other value, such as `$ROOT/rounds/r1/sims`.
    settings
        The machine settings; by default `read()`.
    """
    names = read() if settings is None else settings
    return Template(value).safe_substitute(names)


if __name__ == "__main__":
    import sys

    cfg = read()
    for key in sys.argv[1:]:
        print(cfg.get(key, ""))
