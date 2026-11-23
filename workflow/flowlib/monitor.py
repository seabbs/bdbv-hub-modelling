"""The workflow monitor: every Snakemake run reports to one Panoptes server.

Panoptes (`panoptes-ui`) serves every run's jobs, their state, what is
running and what has failed; Snakemake sends them through its logger
plugin (`snakemake-logger-plugin-panoptes`). `args` gives a run's logger
options from the machine's `MONITOR_URL` (config/local.env), and
`task nowcast-workflow:monitor` serves the page on `MONITOR_HOST` and
`MONITOR_PORT`. The plugin drops events it cannot send, so a run goes on
when no server is listening.

    python3 workflow/flowlib/monitor.py

prints the options, for the tasks that run Snakemake themselves.
"""

from __future__ import annotations

import shlex
import sys
from collections.abc import Mapping
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flowlib import localconfig

TIMEOUT = 2
"""Seconds a run waits on the server per event."""


def args(cfg: Mapping[str, str]) -> list[str]:
    """Return the Snakemake options that report a run to the monitor.

    None (an empty list) when `MONITOR_URL` is empty, which switches the
    monitor off.
    """
    url = cfg.get("MONITOR_URL", "")
    if not url:
        return []
    return [
        "--logger",
        "panoptes",
        "--logger-panoptes-address",
        url,
        "--logger-panoptes-timeout",
        str(TIMEOUT),
    ]


if __name__ == "__main__":
    print(" ".join(shlex.quote(a) for a in args(localconfig.read())))
