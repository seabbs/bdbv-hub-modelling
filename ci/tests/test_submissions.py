"""The table of what each round submitted under each hub id, and why."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from submission_ci.policy import load_policy
from submission_ci.submissions import COLUMNS, main, rows

POLICY = Path(__file__).resolve().parents[1] / "policy.toml"
HUB = Path(__file__).resolve().parents[2] / "config" / "hub.toml"
PR = "https://github.com/InsightNet-US/BDBV-Modeling-Hub/pull/"


def attempt(cid: str, status: str) -> dict[str, object]:
    return {"id": cid, "kind": "", "status": status, "reasons": []}


def provenance(selected: str, chain: list[str], reason: str) -> dict[str, Any]:
    """A gate's provenance: `chain` from its start, `selected` passing."""
    return {
        "selected": selected,
        "status": "warn",
        "reason": reason,
        "attempts": [
            attempt(c, "warn" if c == selected else "fail")
            if chain.index(c) <= chain.index(selected)
            else attempt(c, "not needed")
            for c in chain
        ],
    }


def model(pr: int, label: str, archive: str) -> dict[str, str]:
    return {
        "status": "submitted",
        "produced_by": label,
        "pr_url": f"{PR}{pr}",
        "run": "run",
        "code_commit": "abc",
        "archive": archive,
    }


def ledger(tmp_path: Path) -> Path:
    """Three rounds: chain start, a substituted week, a manual run."""
    chain = ["bvd", "renewalsp", "main", "randomwalknowcast", "randomwalk"]
    gates = {
        "archive/2026-11-07/1": provenance(
            "bvd", chain, "The bvd model cleared every check."
        ),
        "archive/2026-11-07/1/baseline": provenance(
            "randomwalknowcast", ["randomwalknowcast"], "Cleared."
        ),
        "archive/2026-11-21/2": provenance(
            "renewalsp", chain, "bvd failed plausible."
        ),
    }
    for folder, prov in gates.items():
        (tmp_path / folder).mkdir(parents=True)
        (tmp_path / folder / "provenance.json").write_text(json.dumps(prov))
    (tmp_path / "archive/2026-11-14/manual").mkdir(parents=True)
    bvd = "BVDOutbreakSize-informed renewal model"
    state = {
        "rounds": {
            "2026-11-07": {
                "as_of": "2026-11-01",
                "data_commit": "d1",
                "models": {
                    "seabbs-renewal": model(95, bvd, "archive/2026-11-07/1"),
                    "seabbs-rwnowcast": model(
                        94,
                        "random walk on a nowcast",
                        "archive/2026-11-07/1/baseline",
                    ),
                },
            },
            "2026-11-14": {
                "as_of": "2026-11-08",
                "data_commit": "d2",
                "models": {
                    "seabbs-renewal": model(
                        120, bvd, "archive/2026-11-14/manual"
                    ),
                    "seabbs-rwnowcast": model(
                        121,
                        "random walk on a nowcast",
                        "archive/2026-11-14/manual/baseline",
                    ),
                },
            },
            "2026-11-21": {
                "as_of": "2026-11-15",
                "data_commit": "d3",
                "models": {
                    "seabbs-renewal": model(
                        136,
                        "single-patch province renewal models",
                        "archive/2026-11-21/2",
                    ),
                },
            },
        }
    }
    (tmp_path / "state.json").write_text(json.dumps(state))
    return tmp_path


def by_round(
    table: list[dict[str, str]],
) -> dict[tuple[str, str], dict[str, str]]:
    return {(r["reference_date"], r["hub_id"]): r for r in table}


def test_each_round_and_hub_id_has_its_model_and_chain_start(
    tmp_path: Path,
) -> None:
    root = ledger(tmp_path)
    state = json.loads((root / "state.json").read_text())
    table = by_round(rows(state, root, load_policy(POLICY)))
    first = table["2026-11-07", "seabbs-renewal"]
    assert first["model"] == "renewalbvd"
    assert first["chain_start"] == "renewalbvd"
    assert first["substituted"] == "false"
    assert first["reason"] == "The bvd model cleared every check."
    assert first["source"] == "provenance"
    walk = table["2026-11-07", "seabbs-rwnowcast"]
    assert walk["model"] == "randomwalknowcast"
    assert walk["substituted"] == "false"


def test_a_fallback_is_marked_substituted_with_what_failed(
    tmp_path: Path,
) -> None:
    root = ledger(tmp_path)
    state = json.loads((root / "state.json").read_text())
    row = by_round(rows(state, root, load_policy(POLICY)))[
        "2026-11-21", "seabbs-renewal"
    ]
    assert row["model"] == "renewalsp"
    assert row["chain_start"] == "renewalbvd"
    assert row["substituted"] == "true"
    assert row["ruled_out"] == "renewalbvd"
    assert row["reason"] == "bvd failed plausible."


def test_a_run_without_provenance_reads_the_model_from_its_label(
    tmp_path: Path,
) -> None:
    """The manual 2026-11-14 run left no gate record: its label names the
    chain's first candidate, so it was not substituted."""
    root = ledger(tmp_path)
    state = json.loads((root / "state.json").read_text())
    row = by_round(rows(state, root, load_policy(POLICY)))[
        "2026-11-14", "seabbs-renewal"
    ]
    assert row["model"] == "renewalbvd"
    assert row["chain_start"] == "renewalbvd"
    assert row["substituted"] == "false"
    assert row["source"] == "label"


def test_the_command_writes_the_table(tmp_path: Path) -> None:
    root = ledger(tmp_path)
    out = tmp_path / "submissions.csv"
    code = main(
        [
            "--state",
            str(root / "state.json"),
            "--ledger",
            str(root),
            "--policy",
            str(POLICY),
            "--hub-config",
            str(HUB),
            "--out",
            str(out),
        ]
    )
    assert code == 0
    with out.open(newline="") as f:
        reader = csv.DictReader(f)
        assert tuple(reader.fieldnames or ()) == COLUMNS
        table = list(reader)
    assert [(r["reference_date"], r["hub_id"]) for r in table] == [
        ("2026-11-07", "seabbs-renewal"),
        ("2026-11-07", "seabbs-rwnowcast"),
        ("2026-11-14", "seabbs-renewal"),
        ("2026-11-14", "seabbs-rwnowcast"),
        ("2026-11-21", "seabbs-renewal"),
    ]


def test_a_hub_id_of_its_own_starts_at_its_own_model(tmp_path: Path) -> None:
    """With no gate record, seabbs-rwnowcast's start is its own model in
    config/hub.toml, not the renewal chain's."""
    root = ledger(tmp_path)
    state = json.loads((root / "state.json").read_text())
    starts = {
        "seabbs-renewal": "renewalbvd",
        "seabbs-rwnowcast": "randomwalknowcast",
    }
    row = by_round(rows(state, root, load_policy(POLICY), starts))[
        "2026-11-14", "seabbs-rwnowcast"
    ]
    assert row["model"] == "randomwalknowcast"
    assert row["chain_start"] == "randomwalknowcast"
    assert row["substituted"] == "false"
