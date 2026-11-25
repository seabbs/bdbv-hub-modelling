"""Assemble a dashboard from the workflow config, the machine and the CLI.

The runs are the workflow's (`runs` in workflow/config.yaml), each in
`<ROOT>/runs/<name>`; `--run NAME` picks some of them by name.
`workflow/config.yaml`'s `dashboard` section holds the rest (references,
the scoring rule, the mailbox filter, the datasets); `config/local.env`
holds this machine's `ROOT`, scoring cache (`SCORE_CACHE`) and mailbox;
`--ref` adds or replaces references.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from nowcasteval import labels

from . import logs, paperstyle
from . import runs as spec
from .page import Context
from .sources import Sources

RUN_MODEL = "foundation"
"""The model a run's checkpoints are, for their display name and colour."""


def references(
    entries: Sequence[Mapping[str, Any]], cli: Sequence[str] = ()
) -> list[spec.Reference]:
    """Return the configured references, then `--ref` ones.

    A reference with `paper_model` takes that model's name, colour and
    line style from the paper (`paperstyle`) unless it sets its own.
    """
    resolved = [
        {
            **e,
            "label": e.get("label") or paperstyle.model_label(e["paper_model"]),
            "colour": e.get("colour")
            or paperstyle.model_colour(e["paper_model"]),
            "dash": e.get("dash") or paperstyle.model_dash(e["paper_model"]),
        }
        if e.get("paper_model")
        else e
        for e in entries
    ]
    return spec.load_references(resolved, cli)


def workflow_runs(
    config: Mapping[str, Any], local: Mapping[str, str], names: Sequence[str]
) -> list[spec.Run]:
    """Return the runs `names` picks, else every run under ROOT.

    Each run is its folder, `<ROOT>/runs/<name>` (flowlib.runs). With no
    names, the workflow's runs (`runs`) that have trained here, newest
    activity first (`active_runs`), else all of them in the config's
    order. The first is live.

    Raises
    ------
    SystemExit
        If a name is not one of the workflow's runs, or there are none.
    """
    known = list(config.get("runs") or {})
    unknown = [n for n in names if n not in known]
    if unknown:
        raise SystemExit(
            f"no run {', '.join(unknown)}: the workflow's runs are "
            f"{', '.join(known)}"
        )
    root = Path(local.get("DASHBOARD_ROOT") or local.get("ROOT") or ".")
    root = root.expanduser()
    chosen = list(names) or active_runs(known, root) or known
    if not chosen or not local.get("ROOT"):
        raise SystemExit(
            "no runs to show: the dashboard shows the workflow's runs "
            "(runs in workflow/config.yaml) under ROOT in config/local.env"
        )
    return spec.load_runs([(n, root / "runs" / n) for n in chosen])


def active_runs(known: Sequence[str], root: Path) -> list[str]:
    """Return the runs of `known` that have trained under `root`.

    The most recently active first, by the newest file in its `train/`.
    """

    def last(name: str) -> float:
        train = root / "runs" / name / "train"
        return max((p.stat().st_mtime for p in train.iterdir()), default=0.0)

    found = [n for n in known if (root / "runs" / n / "train").is_dir()]
    return sorted(found, key=last, reverse=True)


def build(
    config: Mapping[str, Any],
    local: Mapping[str, str],
    runs: Sequence[str] = (),
    refs: Sequence[str] = (),
    once: bool = False,
) -> Context:
    """Return the dashboard's context.

    Raises
    ------
    SystemExit
        If `runs` names no run of the workflow's, or it has none.
    """
    ft = config["finetune"]
    dash = dict(config["dashboard"])
    # Held out of model development (BDBV): never shown.
    held = dict(dash.get("held_out") or {})
    hidden = set(held.get("datasets") or ())
    datasets = [d for d in dash.get("datasets") or [] if d["key"] not in hidden]
    hidden_studies = set(held.get("studies") or ())
    studies = {
        k: v
        for k, v in (dash.get("studies") or {}).items()
        if k not in hidden_studies
    }
    run_list = named_runs(workflow_runs(config, local, runs))
    source = Sources(local, datasets)
    return Context(
        runs=run_list,
        references=references(dash.get("references") or [], refs),
        source=source,
        rule=dash["rule"],
        outbreaks=int(ft["outbreaks"]),
        prefixes=tuple(int(p) for p in ft["prefixes"]),
        learning_rate=float(ft["learning_rate"]),
        datasets=datasets,
        mailbox=source.mailbox,
        mail_filter=dash.get("mail_filter", ""),
        mail_hide=held.get("mail", ""),
        hidden_studies=hidden_studies,
        once=once,
        view={
            "studies": studies,
            "levels": dash.get("levels") or [],
            "defaults": dash.get("score_defaults") or {},
        },
        excluded={
            str(h) for h in (dash.get("exclude") or {}).get("hashes") or ()
        },
    )


RUN_SHADES = labels.RUN_SHADES
"""Colours of earlier runs, from the shared map (`labels.RUN_SHADES`)."""


def _identifier(run: spec.Run) -> str:
    """Return an earlier run's suffix: its settings' grid, else its key."""
    if run.folder is not None:
        for _, s, _ in reversed(logs.checkpoints(run.folder)):
            if s.get("grid"):
                return f"{s['grid']} grid"
    return run.key


def named_runs(run_list: Sequence[spec.Run]) -> list[spec.Run]:
    """Name and colour every run as the fine-tuned model.

    The live run is the model's display name in its colour; earlier runs
    are `<name> - <identifier>` (their grid, else their key)
    in shades of it (`RUN_SHADES`). A configured label or colour is not
    shown, so the toggles and every panel name runs alike.
    """
    name = paperstyle.model_label(RUN_MODEL)
    colour = paperstyle.model_colour(RUN_MODEL)
    # The live run, as `runs.load_runs` marks it, else the first.
    live = next((r.key for r in run_list if r.live), None)
    live = live or (run_list[0].key if run_list else None)
    ids = {r.key: _identifier(r) for r in run_list if r.key != live}
    # Two earlier runs of one grid are told apart by their keys.
    dup = {i for i in ids.values() if list(ids.values()).count(i) > 1}
    out = []
    shade = iter(RUN_SHADES * (1 + len(run_list) // len(RUN_SHADES)))
    for r in run_list:
        if r.key == live:
            r = dataclasses.replace(r, label=name, colour=colour)
        else:
            suffix = r.key if ids[r.key] in dup else ids[r.key]
            r = dataclasses.replace(
                r, label=f"{name} - {suffix}", colour=next(shade)
            )
        out.append(r)
    return out
