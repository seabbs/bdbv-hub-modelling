"""Serve the fine-tuning dashboard with a workflow run, or alone.

One entry for both: `ensure` starts `workflow/scripts/dashboard.py`
detached, listening on DASHBOARD_HOST:DASHBOARD_PORT, unless the one in
the pid file (`pidfile`) is still alive. The server ends by itself when
the process it was started for ends (`--while-pid`): the coordinator
run (`DRIVER_ENV`, set by workflow/scripts/coordinate.py), else the
Snakemake run. The workflow calls it from its `onstart` when
`dashboard.serve` is true; `task nowcast-foundation:dashboard` runs the
same script in the foreground.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "scripts" / "dashboard.py"
"""The dashboard's server (workflow/scripts/dashboard.py)."""

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
    if cmdline.is_file():
        return Path(script).name.encode() in cmdline.read_bytes()
    return True


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


def ensure(
    local: Mapping[str, str], driver: int, script: Path | str = SCRIPT
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

    Returns
    -------
    int
        The pid of the dashboard serving `root(local)`.
    """
    path = pidfile(local)
    old = _recorded(path)
    if old and serving(old, script):
        return old
    path.parent.mkdir(parents=True, exist_ok=True)
    with (path.parent / "serve.log").open("ab") as log:
        proc = subprocess.Popen(
            [sys.executable, str(script), "--while-pid", str(driver)],
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    path.write_text(f"{proc.pid}\n")
    return proc.pid
