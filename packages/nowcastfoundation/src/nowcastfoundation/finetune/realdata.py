# Copyright (c) 2026 The bdbv-hub-modelling authors. MIT licence, see LICENSE.
"""Fine-tune Chronos-2 on real data alone, for `foundation_real`.

The workflow's `realdata` run (workflow/config.yaml `runs`, `data:
real`). It trains a LoRA adapter from base Chronos-2 on the German and
BDBV real-data blocks (`nowcastgermany.finetune_blocks --realdata`,
`nowcastbdbv.finetune_blocks`) with the simulation fine-tune's recipe
(`finetune.stream`: its LoRA, series bias, optimiser and batches packed
by size). It trains in passes over every training block, each pass
shuffled with its own seed, with the learning rate falling linearly to
zero over the passes. The passes are as many as give the main run's
steps (`--target-steps`), at most `--max-passes` (`choose_passes`). The
blocks of the last `--held-out-dates` German nowcast dates are held out
and their loss is logged; they are not trained on, and no checkpoint is
chosen by it: the final one is reported.

    DEVICE=cuda uv run --group chronos finetune-real OUT \
        --blocks germany.pkl bdbv.pkl --prefixes 125 250 500 1000 \
        --until 125

Checkpoints are fractions of the whole training, in thousandths
(`checkpoint_steps`: 1/8, 1/4, 1/2 and all). Each `--until N` is saved
in `OUT/n<N>` as the simulation runs' are (settings with its hash,
adapter, `consumed.txt`), with the blocks it trained on in `blocks.txt`;
the next resumes from the state kept with it (`--prev OUT/n<N>/state`).
The workflow verifies, scores and releases them as it does the other
runs' (`finetune-verify`, `publish`, here as `realdata-ft-<hash>`).
"""

from __future__ import annotations

import argparse
import inspect
import json
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from nowcastfoundation import series_bias
from nowcastfoundation.backends.chronos2 import torch_device
from nowcastfoundation.finetune import chronos as ft
from nowcastfoundation.finetune import stream as fs
from nowcastfoundation.finetune import verify
from nowcastfoundation.forecaster import (
    CHRONOS_MODEL,
    CHRONOS_REVISION,
    REALDATA_PREFIX,
    SUMS_FILE,
)

TRAINER = (
    "packages/nowcastfoundation/src/nowcastfoundation/finetune/realdata.py"
)
"""The trainer's path, its name in `code_sha256`."""

CODE = (
    TRAINER,
    ft.TRAINER,
    "packages/nowcastfoundation/src/nowcastfoundation/finetune/chronos.py",
    "packages/nowcastfoundation/src/nowcastfoundation/finetune/__init__.py",
    "packages/nowcastfoundation/src/nowcastfoundation/finetune/realtasks.py",
    "packages/nowcastgermany/src/nowcastgermany/finetune_blocks.py",
    "packages/nowcastbdbv/src/nowcastbdbv/finetune_blocks.py",
    "packages/nowcastfoundation/config/finetune.toml",
)
"""Files a checkpoint depends on, hashed into its settings by path: the
trainer, the simulation trainer whose recipe it shares, the block layout
and the real-data blocks' layout, the block cutters and the real-data
cut (`[realdata_germany]`, `[realdata_bdbv]`)."""

TARGET_STEPS = 82040
"""The main run's steps over its 4,800 simulations. Still the old main
run's final checkpoint (`chronos-slots-66a7a630d729`, n4800) until the
retrained `finetuned` run's final checkpoint gives its own."""

MAX_PASSES = 20
"""Most passes over the real-data blocks, however few steps a pass is."""

WHOLE = 1000
"""A checkpoint prefix of the whole training, in thousandths."""

LOG_EVERY = 10
"""Steps between logged training losses."""

BLOCKS_FILE = "blocks.txt"
"""The training blocks' ids, in a checkpoint's folder and its release."""


def log(message: str) -> None:
    """Print a timestamped line."""
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{stamp} [realdata] {message}", flush=True)


def block_id(block: Mapping[str, Any]) -> str:
    """Return a block's id: data, cut, source, places and lag."""
    places = block["places"]
    names = "+".join(places) if isinstance(places, tuple) else str(places)
    return (
        f"{block.get('data', 'germany')}:{block['cut']}:{block['source']}:"
        f"{names}:{block['lag']}"
    )


def split_held_out(
    blocks: Sequence[dict[str, Any]], held_out_dates: Mapping[str, int]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return the training blocks and the held-out blocks.

    Each dataset holds out the blocks of its last `held_out_dates[data]`
    cut dates (none where unset or 0).
    """
    held_cuts: dict[str, set[str]] = {}
    for data in {b.get("data", "germany") for b in blocks}:
        n = held_out_dates.get(data, 0)
        cuts = sorted({b["cut"] for b in blocks if b.get("data") == data})
        held_cuts[data] = set(cuts[len(cuts) - n :]) if n else set()
    train, held = [], []
    for b in blocks:
        if b["cut"] in held_cuts[b.get("data", "germany")]:
            held.append(b)
        else:
            train.append(b)
    return train, held


def choose_passes(steps_per_pass: int, target: int, most: int) -> int:
    """Return the passes whose steps come nearest `target`, 1 to `most`."""
    return int(min(most, max(1, round(target / max(1, steps_per_pass)))))


def pass_plans(
    lengths: np.ndarray,
    sizes: np.ndarray,
    batch_size: int,
    seed: int,
    passes: int,
) -> list[list[tuple[int, list[int]]]]:
    """Return each pass's batches: every block once, shuffled per pass.

    As a simulation chunk is packed (`chronos.plan_by_length`), each pass
    with its own seed.
    """
    return [
        ft.plan_by_length(
            lengths,
            sizes,
            batch_size,
            np.random.default_rng([seed, p]),
            ft.MAX_SERIES,
        )
        for p in range(passes)
    ]


def checkpoint_steps(total: int, prefixes: Sequence[int]) -> dict[int, int]:
    """Return each checkpoint's step: its thousandths of `total` steps."""
    return {n: round(n * total / WHOLE) for n in prefixes}


def check_until(until: int, prefixes: Sequence[int]) -> None:
    """Raise if `until` is not one of the run's checkpoints."""
    if until not in prefixes:
        raise ValueError(f"--until {until} is not a checkpoint of {prefixes}")


def is_real(settings: Mapping[str, Any]) -> bool:
    """Return whether a checkpoint was trained on real data alone."""
    return settings.get("data") == "real"


def checkpoint_settings(
    base: Mapping[str, Any],
    steps: int,
    consumed_sha256: str,
    epochs: float,
) -> dict[str, Any]:
    """Return a checkpoint's hashed settings and its hash.

    `base` is the run's settings with its total `steps`, the learning
    rate's schedule (`lr_steps`); the checkpoint's are the steps trained.
    """
    settings = {
        **base,
        "lr_steps": base["steps"],
        "steps": steps,
        "epochs": epochs,
        "consumed_sha256": consumed_sha256,
    }
    settings["hash"] = verify.settings_hash(settings)
    return settings


def release_tag(key: str) -> str:
    """Return the release of a real-data checkpoint, `realdata-ft-<hash>`."""
    return f"{REALDATA_PREFIX}-{key}"


def publish(where: Path, run: Any = subprocess.run) -> str:
    """Release checkpoint `where` as `realdata-ft-<hash>` and return the tag.

    As the simulation checkpoints are released (`finetune.stream.publish`),
    with the blocks it trained on in place of the outbreaks: the adapter,
    the settings, `blocks.txt` and `sha256sums.txt`. Safe to run again: a
    release whose sums match is left as it is, one missing assets gets
    them, and one whose sums differ is an error. The fine-tuned models'
    latest lookup never takes it.
    """
    settings = json.loads((where / "settings.json").read_text())
    key = str(settings["hash"])
    tag = release_tag(key)
    files = [
        where / "finetuned-ckpt" / "adapter_config.json",
        where / "finetuned-ckpt" / "adapter_model.safetensors",
        where / "settings.json",
        where / BLOCKS_FILE,
    ]
    sums = where / SUMS_FILE
    sums.write_text("".join(f"{ft._sha256(f)}  {f.name}\n" for f in files))
    files.append(sums)

    def gh(*command: str) -> subprocess.CompletedProcess[str]:
        done: subprocess.CompletedProcess[str] = run(
            ["gh", "release", *command, "-R", fs.REPO],
            capture_output=True,
            text=True,
        )
        return done

    view = gh("view", tag, "--json", "assets")
    if view.returncode != 0:
        result = gh(
            "create",
            tag,
            "--latest=false",
            "--title",
            f"Chronos-2 fine-tuned on real data {key} "
            f"({settings['steps']} steps, SI only)",
            "--notes",
            "LoRA adapter of Chronos-2 fine-tuned from base Chronos-2 on "
            "real data alone, with the main run's recipe: the German "
            "hospitalisation data as available on the last day of the "
            "development half and the public BDBV data up to the release "
            "in [realdata_bdbv] (packages/nowcastfoundation/config/"
            "finetune.toml), nowcastfoundation.finetune.realdata. "
            "`foundation_real` loads it once pinned; the _ft models never "
            "do. Check the files with sha256sums.txt.\n\nThis was opened by "
            "a bot. Please ping @seabbs for any questions.",
            *map(str, files),
        )
        if result.returncode != 0:
            raise RuntimeError(f"could not publish {tag}: {result.stderr}")
        log(f"published {tag}")
        return tag
    present = {a["name"] for a in json.loads(view.stdout)["assets"]}
    if sums.name in present:
        theirs = gh("download", tag, "-p", sums.name, "-O", "-").stdout
        if theirs != sums.read_text():
            raise RuntimeError(f"{tag} exists and its sha256sums differ")
    missing = [f for f in files if f.name not in present]
    if missing:
        result = gh("upload", tag, *map(str, missing))
        if result.returncode != 0:
            raise RuntimeError(f"could not upload to {tag}: {result.stderr}")
    return tag


def code_sha256(bias: bool) -> dict[str, str]:
    """Return the SHA-256 of each file in `CODE`, by its path."""
    files = list(CODE) + (list(fs.BIAS_CODE) if bias else [])
    return {path: ft.code_sha256(path) for path in files}


def resume_from(live: Path, prev: Path | None, until: int) -> None:
    """Leave the state segment `until` resumes from in `live`.

    The run's own state if it is between the previous checkpoint and
    `until` (a restarted segment), else the previous checkpoint's
    (`prev`), else none (the first segment).
    """
    found = live / "state.json"
    step = json.loads(found.read_text())["step"] if found.exists() else None
    floor = 0
    if prev is not None and (prev / "state.json").exists():
        floor = json.loads((prev / "state.json").read_text())["step"]
    if step is not None and floor <= step <= until:
        return
    shutil.rmtree(live, ignore_errors=True)
    if prev is not None:
        shutil.copytree(prev, live)


def build_parser() -> argparse.ArgumentParser:
    """Return the trainer's command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--blocks", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--prefixes", type=int, nargs="+", default=[125, 250, 500, WHOLE]
    )
    parser.add_argument("--target-steps", type=int, default=TARGET_STEPS)
    parser.add_argument("--max-passes", type=int, default=MAX_PASSES)
    parser.add_argument("--until", type=int, required=True)
    parser.add_argument("--prev", type=Path)
    parser.add_argument("--held-out-dates", type=int, default=14)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--series-bias", action="store_true")
    parser.add_argument("--bias-lr", type=float, default=fs.BIAS_LR)
    parser.add_argument("--batch-size", type=int, default=256)
    # Series above which a step recomputes activations in backward, off
    # CUDA (memory only: not in the hash).
    parser.add_argument("--checkpoint-above", type=int, default=733)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--note", default="")
    return parser


def main() -> None:
    """Train the real-data checkpoint to checkpoint `--until` and save it."""
    import torch
    from chronos import Chronos2Pipeline
    from chronos.chronos2.dataset import DatasetMode
    from chronos.chronos2.model import Chronos2Model

    args = build_parser().parse_args()
    prefixes = sorted(args.prefixes)
    try:
        check_until(args.until, prefixes)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    commit = fs.code_commit()
    device = torch_device(args.device)
    every = [b for path in args.blocks for b in fs.load_blocks(path)]
    train, held = split_held_out(every, {"germany": args.held_out_dates})
    ids = [block_id(b) for b in train]
    (out / BLOCKS_FILE).write_text("\n".join(ids) + "\n")
    sizes = np.array([int(b["context"].shape[0]) for b in train])
    lengths = np.array([b["prediction_length"] for b in train])
    one = pass_plans(lengths, sizes, args.batch_size, args.seed, 1)[0]
    passes = choose_passes(len(one), args.target_steps, args.max_passes)
    plans = pass_plans(lengths, sizes, args.batch_size, args.seed, passes)
    total = sum(len(plan) for plan in plans)
    until = checkpoint_steps(total, prefixes)[args.until]
    log(
        f"{len(train)} training blocks, {len(held)} held out, on {device}; "
        f"{passes} passes of about {len(one)} steps, {total} steps; "
        f"checkpoint n{args.until} at step {until}"
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)  # noqa: NPY002
    pipeline = Chronos2Pipeline.from_pretrained(
        CHRONOS_MODEL, revision=CHRONOS_REVISION, device_map=device
    )
    if args.series_bias:
        series_bias.attach(pipeline.model)
    model = fs.lora_model(pipeline.model, fs.lora_config(args.series_bias))
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimiser = fs.make_optimiser(model, args.learning_rate, args.bias_lr)
    accepted = inspect.signature(Chronos2Model.forward).parameters
    base = {
        "data": "real",
        "grid": "realdata",
        "chronos_revision": CHRONOS_REVISION,
        "code_sha256": code_sha256(args.series_bias),
        "blocks_sha256": [ft._sha256(p) for p in args.blocks],
        "blocks_list_sha256": ft._sha256(out / BLOCKS_FILE),
        "n_blocks": {
            d: sum(b.get("data") == d for b in train)
            for d in ("germany", "bdbv")
        },
        "n_validation": len(held),
        "held_out_dates": {"germany": args.held_out_dates},
        "lora": {
            "r": 8,
            "alpha": 16,
            **(
                {"modules_to_save": [series_bias.BIAS_MODULE]}
                if args.series_bias
                else {}
            ),
        },
        **fs.bias_settings(args.series_bias, args.bias_lr),
        "learning_rate": args.learning_rate,
        "lr_decay": "linear to zero over the passes",
        "passes": passes,
        "pass_steps": [len(plan) for plan in plans],
        "target_steps": args.target_steps,
        "max_passes": args.max_passes,
        "steps": total,
        "batch_size": args.batch_size,
        "context": ft.CONTEXT,
        "seed": args.seed,
        "note": args.note,
    }
    live = out / "state"
    resume_from(live, args.prev, until)
    state = fs.load_state(live, model, optimiser)
    for group, base_lr in zip(
        optimiser.param_groups,
        (args.learning_rate, args.bias_lr),
        strict=False,
    ):
        group["base_lr"] = base_lr
    rule = fs.memory_rule(model, device, (state or {}).get("memory"))
    above = fs.series_threshold(args.checkpoint_above, rule)
    fs.checkpoint_large_batches(above)
    consumed = out / "consumed.txt"
    if state is None:
        consumed.write_text("")
        start = fs.held_out_parts(model, held, args.batch_size)
        state = {
            "step": 0,
            "epoch": 0,
            "position": 0,
            "start": start,
            "history": [{"step": 0, "held_out_loss": start["loss"]}],
        }
        log(f"held-out loss at the start {start['loss']:.4f}")
    else:
        lines = consumed.read_text().splitlines()[: state["step"]]
        consumed.write_text("".join(f"{line}\n" for line in lines))
        log(f"resumed at step {state['step']}")
    data = fs.ByLength(
        train, args.batch_size, DatasetMode.TRAIN, args.series_bias
    )

    def backward(part: list[int], share: float) -> float:
        batch = data._build_batch(part)
        value = fs.train_step(model, batch, accepted, device, share, rule)
        del batch
        return value

    def zero() -> None:
        optimiser.zero_grad()
        fs.free_memory()

    def snapshot() -> None:
        memory = {"memory": rule.state()} if rule is not None else {}
        fs.save_state(live, model, optimiser, state | memory)

    t0 = time.perf_counter()
    plan: list[tuple[int, list[int]]] = []
    while state["step"] < until:
        plan = plans[state["epoch"]]
        while state["position"] < len(plan) and state["step"] < until:
            _, indices = plan[state["position"]]
            fs.set_learning_rates(optimiser, state["step"], total)
            optimiser.zero_grad()
            value = fs.accumulate(list(indices), backward, zero)
            if not fs.finite_step(params, optimiser, value):
                log(f"step {state['step'] + 1}: no update (loss {value})")
            fs.free_memory()
            state["step"] += 1
            state["position"] += 1
            with consumed.open("a") as f:
                f.write(
                    f"{state['step']}\t"
                    + ",".join(ids[i] for i in indices)
                    + "\n"
                )
            if state["step"] % LOG_EVERY == 0:
                state["history"].append({"step": state["step"], "loss": value})
                log(
                    f"step {state['step']}: loss {value:.4f}, epoch "
                    f"{state['epoch']}, {time.perf_counter() - t0:.0f} s"
                )
            if state["step"] % fs.SAVE_EVERY == 0:
                snapshot()
        if state["position"] >= len(plan):
            state["epoch"] += 1
            state["position"] = 0
    where = out / f"n{args.until}"
    where.mkdir(parents=True, exist_ok=True)
    shutil.copy(consumed, where / "consumed.txt")
    shutil.copy(out / BLOCKS_FILE, where / BLOCKS_FILE)
    model.save_pretrained(where / "finetuned-ckpt")
    parts = fs.held_out_parts(model, held, args.batch_size)
    epochs = state["epoch"] + state["position"] / max(1, len(plan))
    settings = checkpoint_settings(
        base, until, ft._sha256(where / "consumed.txt"), epochs
    )
    settings["fraction"] = args.until / WHOLE
    start = state["start"]
    settings |= {
        "code_commit": commit,
        "code_paths": {p: p for p in settings["code_sha256"]},
        "device": device,
        "checkpoint_above": above,
        "held_out_loss_start": start["loss"],
        "held_out_loss": parts["loss"],
        **fs.loss_parts("held_out_loss", parts),
        "seconds": time.perf_counter() - t0,
    }
    (where / "settings.json").write_text(json.dumps(settings, indent=1))
    state["history"].append(
        {
            "step": until,
            "held_out_loss": parts["loss"],
            "hash": settings["hash"],
        }
    )
    (out / "history.json").write_text(json.dumps(state["history"]))
    snapshot()
    fs.keep_state(live, where)
    log(
        f"checkpoint n{args.until}: hash {settings['hash']}, held-out loss "
        f"{parts['loss']:.4f} (start {start['loss']:.4f})"
    )


if __name__ == "__main__":
    main()
