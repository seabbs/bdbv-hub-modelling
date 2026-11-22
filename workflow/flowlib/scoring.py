"""Checkpoint scoring units' commands and the German report's arguments.

A unit is one `nowcastfoundation.checkpoint` study for one checkpoint
label, cached under its study folder (rules/score.smk).
"""

from __future__ import annotations

import shlex
from collections.abc import Iterable

EVALUATE = "uv run --group chronos python -m nowcastfoundation.checkpoint"
"""The scoring CLI."""

REFERENCE_NAMES = {
    "reference": "Zero-shot",
    "randomwalknowcast": "Random walk nowcast",
}
"""The German report's names of the reference unit's models."""


def unit_shell(study: str, room: str, segment: str = "") -> str:
    """Return a unit's shell command, as a rule's `shell:` template.

    The unit of `study` caches under {params.cache} as {params.label}
    once `room` (a wait-for-room prefix ending `&& `, or empty) passes;
    `segment`, if given, is the run's own checkpoint folder it loads.
    """
    return (
        f"mkdir -p {{params.copy_back}} && {room}{{params.prefix}} nice -n 19 "
        f"env {{params.env}} {EVALUATE} {study} {{params.cache}} "
        f"{{params.label}} {segment}> {{log}} 2>&1 && touch {{output}}"
    )


def report_args(
    run: str, saved: Iterable[int], references: Iterable[str], unit_dir: str
) -> str:
    """Return the German report's LABEL=NAME arguments and reference folders.

    `saved` are the run's saved prefixes; each reference label's rows are
    read from `unit_dir`/<label>.
    """
    names = REFERENCE_NAMES | {f"n{n}": f"{run} n{n}" for n in saved}
    args = [shlex.quote(f"{k}={v}") for k, v in names.items()]
    args += [f"--from {lb}={unit_dir}/{lb}" for lb in references]
    return " ".join(args)
