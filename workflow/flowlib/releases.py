"""What the repository's releases hold.

`labels`: checkpoint labels of releases published from a cut-off, the
final run's `chronos-slots-<hash>` first as `slots-<hash>`, then
`chronos-ft-<hash>` as `<hash>`, each group oldest first. Other releases
are left out. The labels must be the ones `nowcastgermany.checkpoint`
takes.

`by_time` orders the same labels oldest first across both kinds, and
`priority` gives newer checkpoints' jobs a higher priority.

`released_seeds`: the simulation batches a `sims-r<N>` release holds.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

FT = "chronos-ft-"
SLOTS = "chronos-slots-"


def labels(
    listing: Iterable[Mapping[str, str]], published_from: str
) -> list[str]:
    """Return the checkpoint labels of `gh release list --json` rows."""
    order = {}
    for r in listing:
        tag, at = r["tagName"], r["publishedAt"]
        if at < published_from:
            continue
        if tag.startswith(SLOTS):
            order["slots-" + tag.removeprefix(SLOTS)] = (0, at)
        elif tag.startswith(FT):
            order[tag.removeprefix(FT)] = (1, at)
    return sorted(order, key=order.__getitem__)


def by_time(
    listing: Iterable[Mapping[str, str]], published_from: str
) -> list[str]:
    """Return the checkpoint labels of `labels`, oldest published first."""
    at = {r["tagName"]: r["publishedAt"] for r in listing}
    tag = {
        lb: (SLOTS + lb.removeprefix("slots-"))
        if lb.startswith("slots-")
        else FT + lb
        for lb in labels(listing, published_from)
    }
    return sorted(tag, key=lambda lb: at[tag[lb]])


def priority(order: Sequence[str], label: str, base: float) -> float:
    """Return a job's priority, newer checkpoints higher, within `base`.

    The labels of `order` (oldest first) take `base` plus their rank as
    a fraction below one, so scoring keeps its place among the rules;
    any other label (the reference) takes `base`.
    """
    if label not in order:
        return base
    return base + (list(order).index(label) + 1) / (len(order) + 1)


def released_seeds(names: Iterable[str]) -> set[int]:
    """Return the batches whose counts and settings are both assets."""
    ext = {"counts": ".csv.gz", "settings": ".csv"}
    have: dict[int, set[str]] = {}
    for name in names:
        m = re.fullmatch(r"(counts|settings)-(\d+)(\.csv(?:\.gz)?)", name)
        if m and m[3] == ext[m[1]]:
            have.setdefault(int(m[2]), set()).add(m[1])
    return {s for s, kinds in have.items() if len(kinds) == 2}


def release_seeds(repo: str, round_: int) -> set[int]:
    """Return the batches release `sims-r<round_>` of `repo` holds.

    None for a round not released yet.
    """
    got = subprocess.run(
        ["gh", "release", "view", f"sims-r{round_}", "-R", repo,
         "--json", "assets"],
        capture_output=True,
        text=True,
    )  # fmt: skip
    if got.returncode:
        return set()
    return released_seeds(a["name"] for a in json.loads(got.stdout)["assets"])


def _listing(path: Path | str) -> list[dict[str, Any]] | None:
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def chosen(listing: Path | str, cfg: Mapping[str, Any]) -> list[str]:
    """Return the checkpoints to score: `cfg["labels"]`, else the listed.

    `cfg` is the workflow's `score` settings; listed releases published
    from `published_from` count, less `skip`.
    """
    if cfg["labels"] is not None:
        return list(cfg["labels"])
    rows = _listing(listing) or []
    skip = set(cfg.get("skip") or ())
    found = labels(rows, cfg["published_from"])
    return [lb for lb in found if lb not in skip]


def oldest_first(listing: Path | str, cfg: Mapping[str, Any]) -> list[str]:
    """Return `cfg["labels"]`, else the listed checkpoints oldest first.

    None if the listing is not there yet: a job's priority cannot wait
    for it.
    """
    if cfg["labels"] is not None:
        return list(cfg["labels"])
    rows = _listing(listing)
    return [] if rows is None else by_time(rows, cfg["published_from"])
