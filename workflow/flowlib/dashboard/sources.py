"""Where the dashboard reads scores, forecasts and the mailbox from.

All on this machine: the score table and each dataset's per-checkpoint
forecast folders are in its scoring cache (`SCORE_CACHE`), and the
mailbox is `MAILBOX`. Nothing is written.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

TABLE = "tables/checkpoint-scores.csv"
"""The score table, under a scoring cache."""


class Sources:
    """The paths the dashboard reads."""

    def __init__(
        self,
        cfg: Mapping[str, str],
        datasets: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        self.cache = Path(cfg.get("SCORE_CACHE") or "scores").expanduser()
        self._units = {d["key"]: d.get("unit") or d["key"] for d in datasets}
        box = cfg.get("MAILBOX") or ""
        self.mailbox = Path(box).expanduser() if box else None

    @property
    def scores(self) -> Path:
        """Return the score table's path."""
        return self.cache / TABLE

    def dataset(self, key: str) -> Path:
        """Return a dataset's per-checkpoint forecast folders.

        Its unit's folder in the scoring cache (`unit`, else its key).
        """
        return self.cache / self._units.get(key, key)
