"""An interim checkpoint from a running segment's saved trainer state.

The trainer saves its state (`train/state/`: the adapter, optimiser and
`state.json`) every `stream.SAVE_EVERY` steps. `export` writes that
adapter as a checkpoint folder, `train/interim/s<step>/`, laid out as a
saved segment (`settings.json` and `finetuned-ckpt/`), so the scoring and
evaluation that load a segment folder load it too. It reads files only,
so the trainer runs on undisturbed. The folder is not an `n<N>` name, so
nothing takes it for a saved checkpoint.

Run as `python -m nowcastfoundation.finetune.interim TRAIN`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

INTERIM = "interim"
"""The folder under `train/` holding the interim checkpoints."""

TRIES = 5
"""Reads of the state before giving up while the trainer keeps saving."""


def _read_state(state: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the saved adapter and `state.json` of one save.

    The trainer replaces the state folder by renaming, so `state.json` is
    read before and after the adapter; a step that moved means a save
    landed between them, and the read is taken again.
    """
    import torch

    for _ in range(TRIES):
        before = json.loads((state / "state.json").read_text())
        adapter = torch.load(
            state / "adapter.pt", weights_only=True, map_location="cpu"
        )
        after = json.loads((state / "state.json").read_text())
        if before["step"] == after["step"]:
            return adapter, after
    raise RuntimeError(f"{state} kept changing over {TRIES} reads")


def _last_checkpoint(train: Path) -> Path:
    """Return the run's largest saved `n<N>` segment folder."""
    found = [
        (int(m.group(1)), d)
        for d in train.glob("n*")
        if (m := re.fullmatch(r"n(\d+)", d.name))
        and (d / "settings.json").is_file()
    ]
    if not found:
        raise FileNotFoundError(f"{train} has no saved checkpoint to follow")
    return max(found)[1]


def export(train: Path | str) -> Path:
    """Write the saved state's adapter as `train/interim/s<step>/`.

    Its `settings.json` is the last saved checkpoint's with `interim`,
    the state's `steps` and `n_outbreaks`, the checkpoint it follows
    (`from_checkpoint`) and a `hash` naming the step and the adapter's
    SHA-256, so each export is its own checkpoint. The adapter config is
    the last checkpoint's: the same LoRA.

    Returns
    -------
    pathlib.Path
        The interim checkpoint's folder.
    """
    from safetensors.torch import save_file

    train = Path(train)
    adapter, state = _read_state(train / "state")
    last = _last_checkpoint(train)
    step = int(state["step"])
    where = train / INTERIM / f"s{step}"
    ckpt = where / "finetuned-ckpt"
    ckpt.mkdir(parents=True, exist_ok=True)
    tensors = {k: v.contiguous() for k, v in adapter.items()}
    save_file(tensors, ckpt / "adapter_model.safetensors")
    shutil.copy(
        last / "finetuned-ckpt" / "adapter_config.json",
        ckpt / "adapter_config.json",
    )
    digest = hashlib.sha256(
        (ckpt / "adapter_model.safetensors").read_bytes()
    ).hexdigest()
    settings = json.loads((last / "settings.json").read_text())
    settings |= {
        "interim": True,
        "steps": step,
        "n_outbreaks": int(state["done"]),
        "from_checkpoint": last.name,
        "hash": f"interim-s{step}-{digest[:12]}",
    }
    (where / "settings.json").write_text(json.dumps(settings, indent=1))
    return where


def main() -> None:
    """Export the interim checkpoint of a run's `train` folder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train", type=Path)
    print(export(parser.parse_args().train))


if __name__ == "__main__":
    main()
