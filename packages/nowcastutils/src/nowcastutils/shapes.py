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


def bucket(n: int, steps: int = 8, least: int = 8) -> int:
    """Return `n` rounded up to one of a few sizes, for padding.

    The sizes are the multiples of `1 / steps` of the power of two at or
    below `n`, and at least `least`, so a padded length is at most
    `1 / steps` longer than `n`. A length that grows by one at a time
    takes about `steps` sizes for each doubling.

    Parameters
    ----------
    n : int
        A length.
    steps : int, default 8
        Sizes for each doubling, a power of two.
    least : int, default 8
        The smallest size.

    Returns
    -------
    int
        The padded length.

    Examples
    --------
    >>> [bucket(n) for n in (1, 9, 100, 1000)]
    [8, 9, 104, 1024]
    """
    n = max(int(n), least)
    unit = max((1 << (n.bit_length() - 1)) // steps, 1)
    return round_up(n, unit)
