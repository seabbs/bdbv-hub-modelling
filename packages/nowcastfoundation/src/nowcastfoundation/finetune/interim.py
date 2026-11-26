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


NOT_ITS_OWN = (
    "blocks",
    "series",
    "level_mix",
    "cadence_mix",
    "tokens",
    "seconds",
    "train_seconds",
    "wall_seconds",
    "held_out_loss_start",
    "held_out_loss",
    "held_out_loss_nowcast",
    "held_out_loss_forecast",
    "held_out_loss_by_level",
    "held_out_loss_by_source",
)
"""The last checkpoint's counts and losses, which the interim drops."""


def export(train: Path | str) -> Path:
    """Write the saved state's adapter as `train/interim/s<step>/`.

    The settings are `write_settings`'s. The adapter config is the last
    checkpoint's: the same LoRA.

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
    write_settings(where, train, step, int(state["done"]))
    return where


def write_settings(where: Path, train: Path, step: int, done: int) -> None:
    """Write an interim checkpoint's lists and `settings.json`.

    As a saved checkpoint's (`finetune.stream`): `outbreaks.txt` holds
    the first `done` outbreaks in the order trained (the state's
    `trained`) and `consumed.txt` the run's first `step` steps. The
    settings are the last saved checkpoint's, less the counts and losses
    that are its own (`NOT_ITS_OWN`), with this one's outbreaks, steps
    and list hashes, `interim` and the checkpoint it follows
    (`from_checkpoint`). Its `hash` is taken from them less those
    outside the hash (`verify.NOT_HASHED`), so it verifies
    (`finetune-verify`) and the workflow releases it as any checkpoint.
    """
    from nowcastfoundation.finetune import chronos as ft
    from nowcastfoundation.finetune import verify

    last = _last_checkpoint(train)
    trained = json.loads((train / "state" / "state.json").read_text())[
        "trained"
    ]
    if len(trained) < done:
        raise ValueError(f"the state lists {len(trained)} of {done} outbreaks")
    (where / "outbreaks.txt").write_text("\n".join(trained[:done]) + "\n")
    with (train / "consumed.txt").open() as f:
        consumed = [next(f) for _ in range(step)]
    (where / "consumed.txt").write_text("".join(consumed))
    settings = json.loads((last / "settings.json").read_text())
    kept = {
        k: settings.pop(k)
        for k in ("hash", "code_commit", "code_paths", "device")
        if k in settings
    }
    for key in NOT_ITS_OWN:
        settings.pop(key, None)
    settings |= {
        "n_outbreaks": done,
        "sims_done": done,
        "steps": step,
        "outbreaks_sha256": ft._sha256(where / "outbreaks.txt"),
        "consumed_sha256": ft._sha256(where / "consumed.txt"),
        "interim": True,
        "from_checkpoint": last.name,
    }
    settings["hash"] = verify.settings_hash(settings)
    for key in ("code_commit", "code_paths", "device"):
        if key in kept:
            settings[key] = kept[key]
    (where / "settings.json").write_text(json.dumps(settings, indent=1))


def main() -> None:
    """Export the interim checkpoint of a run's `train` folder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train", type=Path)
    print(export(parser.parse_args().train))


if __name__ == "__main__":
    main()
