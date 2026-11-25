"""Serve the fine-tuning dashboard with a workflow run, or alone.

One entry for both: `ensure` starts `workflow/scripts/dashboard.py`
detached, listening on DASHBOARD_HOST:DASHBOARD_PORT, unless the one in
the pid file (`pidfile`) is still alive. The server ends by itself when
the process it was started for ends (`--while-pid`): the coordinator
run (`DRIVER_ENV`, set by workflow/scripts/coordinate.py), else the
Snakemake run. The workflow calls it from its `onstart` when
`dashboard.serve` is true (the default); with no runs named it shows
every run under ROOT. The same hook keeps one long-lived monitor
serving (`ensure_monitor`), which serves the dashboard under
/finetune/. `task workflow:dashboard` runs the dashboard in the
foreground.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "scripts" / "dashboard.py"
"""The dashboard's server (workflow/scripts/dashboard.py)."""

MONITOR = Path(__file__).parents[1] / "monitor_server.py"
"""The monitor's server (flowlib/monitor_server.py), the machine's one
entry, with the dashboard under /finetune/."""

DRIVER_ENV = "WORKFLOW_DRIVER_PID"
"""The process a served dashboard lives as long as, when set."""


def root(local: Mapping[str, str]) -> Path:
    """Return the ROOT whose runs the dashboard shows (DASHBOARD_ROOT)."""
    return Path(local.get("DASHBOARD_ROOT") or local["ROOT"]).expanduser()


def pidfile(local: Mapping[str, str]) -> Path:
    """Return the pid file of the dashboard served for `root`."""
    return root(local) / "dashboard" / "serve.pid"


def alive(pid: int) -> bool:
    """Return whether process `pid` is running (not a zombie)."""
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    status = Path(f"/proc/{pid}/status")
    if status.is_file():
        return "\nState:\tZ" not in status.read_text()
    return True


def serving(pid: int, script: Path | str) -> bool:
    """Return whether `pid` is alive and running `script`.

    A pid the system has since given to another process is not ours.
    Where there is no /proc (macOS), alive is enough.
    """
    if not alive(pid):
        return False
    cmdline = Path(f"/proc/{pid}/cmdline")
    if not cmdline.parent.is_dir():
        return True
    try:
        return Path(script).name.encode() in cmdline.read_bytes()
    except OSError:
        # It ended between the two checks.
        return False


def _recorded(path: Path) -> int:
    """Return the pid a pid file holds, 0 if none or unreadable."""
    try:
        return int(path.read_text().split()[0])
    except (OSError, IndexError, ValueError):
        return 0


JOB_ENV = "WORKFLOW_JOB"
"""Set in a worker's job (flowlib.compute.job_env), which serves nothing."""


def wanted(config: Mapping[str, object], targets: list[str]) -> bool:
    """Return whether this run serves the dashboard.

    It does when `dashboard.serve` is true, its targets include
    `finetune`, and it is not one job of another machine's run.
    """
    dash = config.get("dashboard") or {}
    on = bool(dash.get("serve")) if isinstance(dash, Mapping) else False
    return on and "finetune" in targets and not os.environ.get(JOB_ENV)


def driver_pid() -> int:
    """Return the process a served dashboard should outlive no longer."""
    return int(os.environ.get(DRIVER_ENV) or os.getpid())


def _start(path: Path, argv: list[str], env: Mapping[str, str] | None) -> int:
    """Start `argv` detached, logging beside the pid file `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with (path.parent / "serve.log").open("ab") as log:
        proc = subprocess.Popen(
            argv,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env=None if env is None else {**os.environ, **env},
        )
    path.write_text(f"{proc.pid}\n")
    return proc.pid


def monitor_pidfile(local: Mapping[str, str]) -> Path:
    """Return the pid file of the monitor serving this machine."""
    return Path(local["ROOT"]).expanduser() / "monitor" / "serve.pid"


def ensure_monitor(
    local: Mapping[str, str], script: Path | str = MONITOR
) -> int:
    """Start the monitor unless one is serving; return its pid.

    One per machine and long-lived: it outlives the runs that start it,
    with its database in `<ROOT>/monitor/`.
    """
    path = monitor_pidfile(local)
    old = _recorded(path)
    if old and serving(old, script):
        return old
    db = Path(local["ROOT"]).expanduser() / "monitor" / "panoptes.db"
    return _start(
        path,
        [sys.executable, str(script)],
        {"PANOPTES_DB_URL": f"sqlite:///{db}"},
    )


def ensure(
    local: Mapping[str, str],
    driver: int,
    script: Path | str = SCRIPT,
    runs: Sequence[str] = (),
) -> int:
    """Start the dashboard unless one is serving; return its pid.

    Parameters
    ----------
    local : mapping
        This machine's settings (`localconfig.read`).
    driver : int
        The process the dashboard ends with.
    script : path
        The server to run.
    runs : sequence of str
        The runs it shows, the first live (`--run`); empty: the config's.

    Returns
    -------
    int
        The pid of the dashboard serving `root(local)`.
    """
    path = pidfile(local)
    old = _recorded(path)
    if old and serving(old, script):
        return old
    return _start(
        path,
        [
            sys.executable,
            str(script),
            "--while-pid",
            str(driver),
            *(a for run in runs for a in ("--run", run)),
        ],
        None,
    )
