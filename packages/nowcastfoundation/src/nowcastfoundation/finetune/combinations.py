"""The realised combinations of block layout and options.

Each training block has a layout (its `source`), the levels it holds
(its `mix`: the whole system, the country alone, a view's top or a
subtree) and how its country relates to its places (its `parent`). The
trainer counts each combination as it trains (`count`) and saves the
counts with every checkpoint (`FILE`); `shares` gives the SI's table
of them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

FILE = "block-combinations.json"
"""A checkpoint's combination counts, beside its settings (not hashed)."""

COLUMNS = ("layout", "levels", "parents")
"""A combination's parts, in its key's order (`source|mix|parent`)."""


def count(seen: dict[str, Any], blocks: Iterable[Mapping[str, Any]]) -> None:
    """Add each block to its combination's count in `seen`."""
    counts = seen.setdefault("combinations", {})
    for block in blocks:
        key = f"{block['source']}|{block['mix']}|{block.get('parent', 'exact')}"
        counts[key] = counts.get(key, 0) + 1


def shares(counts: Mapping[str, int]) -> pd.DataFrame:
    """Return each combination's blocks and share, largest first."""
    rows = [(*key.split("|"), n) for key, n in counts.items()]
    out = pd.DataFrame(rows, columns=[*COLUMNS, "blocks"])
    out["share"] = out["blocks"] / out["blocks"].sum()
    return out.sort_values(
        ["blocks", *COLUMNS], ascending=[False, True, True, True]
    ).reset_index(drop=True)
