"""A training run folder with saved checkpoints, for the dashboard tests."""

import json
from pathlib import Path

from flowlib.dashboard import runs


def saved_run(
    folder: Path,
    key: str,
    hashes: tuple[str, ...],
    live: bool = True,
    colour: str = "",
) -> runs.Run:
    """Return a run whose folder has saved a checkpoint of each hash."""
    for i, h in enumerate(hashes):
        d = Path(folder) / "train" / f"n{i + 1}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "settings.json").write_text(json.dumps({"hash": h}))
    return runs.make_run(key, folder, live=live, colour=colour)
