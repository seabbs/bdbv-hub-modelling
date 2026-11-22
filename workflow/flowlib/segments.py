"""Training segments: one per checkpoint prefix, resumed in turn.

The continuous run (`scripts/finetune_stream.py`) checkpoints after the
first N outbreaks of its seed list, for each prefix N. The workflow runs
it once per prefix: segment N resumes from the trainer's state after the
prefix before and stops (`--until N`) once prefix N is saved. The
learning-rate schedule, step and batch position are in that state, so
the segments in turn are the one continuous run.

A batch (`simulate_batch`: `<n>` simulations from `<seed>`) holds the outbreaks
`<seed>-<index>` its settings file lists, so how many batches a prefix
needs is known only once they are simulated.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path


def segments(prefixes: Iterable[int]) -> list[tuple[int | None, int]]:
    """Return each prefix with the prefix it resumes from (None: base)."""
    ordered = sorted(set(prefixes))
    return list(zip([None, *ordered[:-1]], ordered, strict=True))


def check_chunks(
    prefixes: Iterable[int], chunk: int, outbreaks: int | None = None
) -> None:
    """Raise if a prefix does not end a training chunk.

    The trainer takes outbreaks in chunks of `chunk`, the last ending at
    the run's `outbreaks`, so it can stop cleanly only at a chunk's end.
    """
    bad = [p for p in prefixes if p % chunk and p != outbreaks]
    if bad:
        raise ValueError(
            f"prefixes {bad} do not end a chunk of {chunk} outbreaks"
        )


BATCH_SPAN = 10_000
"""Seeds after `seed_base` that `batch_priority` orders: rounds 0 to 99."""


def batch_priority(seed: int, base: float, seed_base: int) -> float:
    """Return a batch's job priority, earlier seeds higher, within `base`.

    Training reads batches in seed order, so a run over several rounds
    simulates them in that order. The fraction below one keeps batches
    between the trainer and scoring; seeds past `BATCH_SPAN` take `base`.
    """
    rank = min(max(seed - seed_base, 0), BATCH_SPAN)
    return base + 1 - (rank + 1) / (BATCH_SPAN + 1)


EVAL_MODES = ("local", "offload")
"""Where the trainer's held-out evaluation runs: in the trainer, or in the
`evaluate` rules on whichever machine `WORKER_GROUPS` gives that group.
Must match `EVAL_MODES` in nowcastfoundation.finetune.stream (tested)."""


def check_eval(mode: str) -> None:
    """Raise if `mode` is not one of `EVAL_MODES`."""
    if mode not in EVAL_MODES:
        raise ValueError(
            f"finetune.eval: {mode} is not one of {', '.join(EVAL_MODES)}"
        )


ROUND_SEEDS = 100
"""Batch seeds each round has, `base` + 100 r on."""


def round_seeds(round_: int, base: int, batches: int) -> list[int]:
    """Return a round's first `batches` batch seeds, `base` + 100 r on.

    A round's seeds are its batches' seeds, each batch drawing its
    simulations independently from the grid.
    """
    first = base + ROUND_SEEDS * round_
    return list(range(first, first + batches))


def seeds_for(
    n: int, seeds: Sequence[int], kept: Callable[[int], int]
) -> list[int]:
    """Return the first seeds of `seeds` whose batches hold `n` outbreaks.

    `kept(seed)` gives a batch's outbreaks. It is asked
    in order and only until there are enough, so in a Snakemake input
    function it can wait on each batch through a checkpoint.
    """
    total = 0
    out = []
    for seed in seeds:
        if total >= n:
            break
        total += kept(seed)
        out.append(seed)
    if total < n:
        raise ValueError(f"the seed list holds only {total} outbreaks, not {n}")
    return out


def batch_outbreaks(settings: Path | str) -> int:
    """Return how many outbreaks a batch's settings file holds."""
    rows = Path(settings).read_text().splitlines()[1:]
    return len({r.split(",")[0] for r in rows if r.strip()})


def _done(state: Path) -> int:
    return int(json.loads((state / "state.json").read_text())["done"])


def resume_state(
    run: Path | str, prev: Path | str | None, prev_n: int | None
) -> str:
    """Put the state a segment resumes from in the trainer's `run` folder.

    The trainer resumes from `run` (its `OUT/state`). That is kept if it
    has reached the prefix before, `prev_n`, as a segment that stopped
    part way, or a live run whose checkpoints kept no state, resumes
    from its own saves. Otherwise the state kept with that prefix
    (`prev`, `n<P>/state`) is copied in.

    Returns
    -------
    str
        "base" (first segment), "kept" or "copied".

    Raises
    ------
    ValueError
        If there is no state at or past the prefix before.
    """
    run = Path(run)
    if prev_n is None:
        return "base"
    if (run / "state.json").exists() and _done(run) >= prev_n:
        return "kept"
    if prev is None or not (Path(prev) / "state.json").exists():
        raise ValueError(
            f"no state at or past n{prev_n} to resume from: neither {run} "
            f"nor {prev}"
        )
    shutil.rmtree(run, ignore_errors=True)
    shutil.copytree(prev, run)
    return "copied"


def reachable(n: int, seeds: Sequence[int], kept: Callable[[int], int]) -> bool:
    """Whether the seed list can hold `n` outbreaks."""
    try:
        seeds_for(n, seeds, kept)
    except ValueError:
        return False
    return True


def training_seeds(
    seed_file: Path | str | None,
    rounds: Iterable[int],
    seeds_of: Callable[[int], list[int]],
) -> list[int]:
    """Return a run's training seeds: `seed_file`'s, else its rounds'."""
    if seed_file:
        return [int(s) for s in Path(seed_file).read_text().split()]
    return [s for r in rounds for s in seeds_of(r)]


def available_seeds(
    order: Sequence[int], held: Iterable[int], listed: set[int] | None
) -> list[int]:
    """Return the training seeds whose batches can be had now, in order.

    `listed` None means this machine simulates them all. Else those
    before the first batch not listed, and none until every held-out
    batch is listed.
    """
    if listed is None:
        return list(order)
    if not set(held) <= listed:
        return []
    for i, seed in enumerate(order):
        if seed not in listed:
            return list(order[:i])
    return list(order)


def outbreaks_or_full(settings: Iterable[Path | str], full: int) -> int:
    """Return the outbreaks of the first settings file found, else `full`."""
    for path in settings:
        if Path(path).exists():
            return batch_outbreaks(path)
    return full


def segment_seeds(
    n: int,
    order: Sequence[int],
    held: Sequence[int],
    kept: Callable[[int], int],
) -> list[int]:
    """Return segment n's seeds: n outbreaks, one spare, the held-out."""
    need = seeds_for(n, order, kept)
    spare = list(order[len(need) : len(need) + 1])
    return need + spare + list(held)


def trainable(
    prefixes: Iterable[int],
    outbreaks: int,
    batch: int,
    order: Sequence[int],
    saved: Callable[[int], bool],
    kept: Callable[[int], int],
) -> list[int]:
    """Return the prefixes saved or within reach of the seed list.

    Each prefix but the run's last (`outbreaks`, every training
    simulation) needs a spare batch past it.
    """
    return [
        n
        for n in prefixes
        if saved(n) or reachable(n + batch * (n < outbreaks), order, kept)
    ]


def to_train(
    prefixes: Sequence[int], saved: Callable[[int], bool], next_only: bool
) -> list[int]:
    """Return the prefixes to ask for: all, or the saved and the next."""
    if not next_only:
        return list(prefixes)
    done = [n for n in prefixes if saved(n)]
    rest = [n for n in prefixes if n not in done]
    return done + rest[:1]
