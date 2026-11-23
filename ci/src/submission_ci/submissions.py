"""Write the table of what each round submitted under each hub id.

One row per round and hub id: the model that went out, the model its
chain started at that round, whether it was substituted (a fallback went
out instead of the chain's start), which candidates were ruled out ahead
of it and the gate's reason. Our evaluation and the paper score
substituted weeks apart from chain-start weeks with it, so one model's
results are never mixed with another's under the same hub id
(`config/hub-comparison.toml`).

The facts come from the ledger: each hub model's record in `state.json`
and its gate's `provenance.json` in the round's archive. The chain's
start is the gate's first attempt, so it is the start at that round,
whatever `config/hub.toml` says now. A run that left no provenance (a
manual run) is read from its label, and its chain start is the hub id's
model in `config/hub.toml` (`--hub-config`), or else the policy's `main`
candidate; `source` says which rows were read this way.

`ci/steps/record.sh` writes the table beside `state.json` on the ledger
branch each time a round is recorded.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from submission_ci.policy import Candidate, Policy, load_policy
from submission_ci.tracks import load_tracks

COLUMNS: tuple[str, ...] = (
    "reference_date",
    "hub_id",
    "status",
    "model",
    "chain_start",
    "substituted",
    "ruled_out",
    "gate_status",
    "reason",
    "source",
    "pr_url",
    "run",
    "code_commit",
    "data_commit",
    "as_of",
)
"""The table's columns, in order."""


def _model(cand: Candidate) -> str:
    """The model a candidate runs (its id when it names none)."""
    return cand.model or cand.id


def _from_provenance(
    prov: dict[str, Any], by_id: dict[str, Candidate]
) -> dict[str, str]:
    """The model, chain start and what was ruled out, from a gate."""

    def key(cid: str) -> str:
        return _model(by_id[cid]) if cid in by_id else cid

    attempts = [a["id"] for a in prov.get("attempts", [])]
    selected = prov.get("selected") or ""
    start = key(attempts[0]) if attempts else ""
    ahead = attempts[: attempts.index(selected)] if selected in attempts else []
    model = key(selected) if selected else ""
    return {
        "model": model,
        "chain_start": start,
        "substituted": _substituted(model, start),
        "ruled_out": ";".join(key(c) for c in ahead),
        "gate_status": str(prov.get("status", "")),
        "reason": str(prov.get("reason", "")),
        "source": "provenance",
    }


def _from_label(label: str, start: str, policy: Policy) -> dict[str, str]:
    """The model from a run's label, for a run with no provenance.

    `start` is the hub id's chain start; empty for the policy's `main`.
    """
    cand = next((c for c in policy.candidates if c.label == label), None)
    model = _model(cand) if cand else ""
    if not start:
        main = next((c for c in policy.candidates if c.kind == "main"), None)
        start = _model(main) if main else ""
    return {
        "model": model,
        "chain_start": start,
        "substituted": _substituted(model, start),
        "ruled_out": "",
        "gate_status": "",
        "reason": "",
        "source": "label",
    }


def _substituted(model: str, start: str) -> str:
    """`true` when a fallback went out; empty when either is unknown."""
    if not model or not start:
        return ""
    return "true" if model != start else "false"


def rows(
    state: dict[str, Any],
    ledger: Path,
    policy: Policy,
    starts: Mapping[str, str] | None = None,
) -> list[dict[str, str]]:
    """Return one row per round and hub id.

    Parameters
    ----------
    state : dict
        The ledger's `state.json`.
    ledger : pathlib.Path
        The ledger branch's checkout, holding the rounds' archives.
    policy : Policy
        The chain's candidates, to map gate ids and labels to models.
    starts : mapping, optional
        Hub id to its chain's first model, for runs with no provenance
        (`config/hub.toml`). A hub id not in it starts at the policy's
        `main` candidate.

    Returns
    -------
    list of dict
        Rows with `COLUMNS`, by round and then hub id.
    """
    by_id = {c.id: c for c in policy.candidates}
    out = []
    for ref, rnd in sorted(state.get("rounds", {}).items()):
        for hub_id, rec in sorted(rnd.get("models", {}).items()):
            prov_file = ledger / rec.get("archive", "") / "provenance.json"
            if rec.get("archive") and prov_file.is_file():
                facts = _from_provenance(
                    json.loads(prov_file.read_text(encoding="utf-8")), by_id
                )
            else:
                facts = _from_label(
                    rec.get("produced_by", ""),
                    (starts or {}).get(hub_id, ""),
                    policy,
                )
            out.append(
                {
                    "reference_date": ref,
                    "hub_id": hub_id,
                    "status": rec.get("status", ""),
                    **facts,
                    "pr_url": rec.get("pr_url", ""),
                    "run": rec.get("run", ""),
                    "code_commit": rec.get("code_commit", ""),
                    "data_commit": rnd.get("data_commit", ""),
                    "as_of": rnd.get("as_of", ""),
                }
            )
    return out


def write_table(rows_: list[dict[str, str]], out: Path) -> None:
    """Write the rows as CSV with `COLUMNS`."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows_)


def main(argv: Sequence[str] | None = None) -> int:
    """Command line entry point."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--state", type=Path, required=True)
    p.add_argument(
        "--ledger",
        type=Path,
        default=None,
        help="The ledger checkout (default: the folder of --state)",
    )
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument(
        "--hub-config",
        type=Path,
        default=None,
        help="config/hub.toml: each hub id's model, for runs with no gate",
    )
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    state = json.loads(args.state.read_text(encoding="utf-8"))
    ledger = args.ledger or args.state.parent
    starts = (
        {t.model_id: t.model for t in load_tracks(args.hub_config)}
        if args.hub_config
        else None
    )
    table = rows(state, ledger, load_policy(args.policy), starts)
    write_table(table, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
