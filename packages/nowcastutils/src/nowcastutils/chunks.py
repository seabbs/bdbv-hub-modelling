"""Split a run into blocks that run apart and stack again."""

from __future__ import annotations

from collections.abc import Sequence

CHUNK_HELP = (
    "Run only block K of N (`K/N`, from 0) of the {what}, in order, so the "
    "blocks can run apart and their results be stacked (`concat-results`)."
)
"""The `--chunk` option's help, with `{what}` the items split."""


def chunk(items: Sequence[str], spec: str) -> list[str]:
    """Return block K of N of `items`, with `spec` as `K/N` (K from 0).

    The blocks are contiguous and in order, sizes differing by at most
    one, so stacking blocks 0 to N - 1 gives `items` again.

    Parameters
    ----------
    items : sequence of str
        Such as forecast dates or origins.
    spec : str
        `K/N`.

    Returns
    -------
    list of str
        The block, empty if N exceeds the number of items.

    Raises
    ------
    ValueError
        If `spec` is not `K/N` with 0 <= K < N.
    """
    try:
        k, n = (int(x) for x in spec.split("/"))
    except ValueError:
        raise ValueError(f"chunk must be K/N, got {spec!r}") from None
    if not 0 <= k < n:
        raise ValueError(f"chunk {spec!r} needs 0 <= K < N")
    size, extra = divmod(len(items), n)
    start = k * size + min(k, extra)
    return list(items[start : start + size + (k < extra)])
