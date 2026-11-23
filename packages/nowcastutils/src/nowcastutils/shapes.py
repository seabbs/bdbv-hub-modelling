"""Shared array shapes, so JAX compiles a function once for many sizes."""

from __future__ import annotations


def round_up(n: int, bucket: int) -> int:
    """Return `n` rounded up to a multiple of `bucket`.

    Parameters
    ----------
    n : int
        A length.
    bucket : int
        The multiple.

    Returns
    -------
    int
        The padded length.
    """
    return -(-int(n) // bucket) * bucket
