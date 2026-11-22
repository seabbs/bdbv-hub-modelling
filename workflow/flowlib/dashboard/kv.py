"""One component for every structured value the dashboard shows.

A mapping (a status file, settings, a stage's progress) is a small box
of labelled items; a nested mapping is a box inside an item, and a list
is its items side by side. Labels come from the keys and units from
their suffixes (`_s` seconds, `_kb`/`_mb`/`_gb` sizes), so no field
needs a string of its own.
"""

from __future__ import annotations

import html
import re
from collections.abc import Mapping, Sequence
from typing import Any

_UNIT = re.compile(r"_(s|kb|mb|gb)$")
_BYTES = {"kb": 1e3, "mb": 1e6, "gb": 1e9}
_ACRONYMS = {"eta": "ETA", "gpu": "GPU", "cpu": "CPU", "lr": "LR"}


def duration(seconds: float) -> str:
    """Return a time in readable units, rounded."""
    s = float(seconds)
    if s < 1:
        return f"{s:.2g} s"
    if s < 60:
        return f"{s:.0f} s"
    if s < 3600:
        return f"{s / 60:.0f} min"
    if s < 86400:
        return f"{int(s // 3600)} h {int(s % 3600 // 60):02d} min"
    return f"{int(s // 86400)} d {int(s % 86400 // 3600)} h"


def size(n: float) -> str:
    """Return a size in bytes in readable units."""
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if n >= scale:
            return f"{n / scale:.3g} {unit}"
    return f"{n:.0f} B"


def label(key: str) -> str:
    """Return a key as a label: no unit suffix, words, a capital.

    A name with digits (a round `r34`, a checkpoint `n800`) is kept.
    """
    words = " ".join(
        _ACRONYMS.get(w.lower(), w)
        for w in _UNIT.sub("", str(key)).split("_")
        if w
    )
    if any(c.isdigit() for c in words):
        return words
    return words[:1].upper() + words[1:]


def value(key: str, v: Any) -> str:
    """Return a scalar as text, in the unit its key names."""
    unit = _UNIT.search(str(key))
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int | float) and unit:
        if unit[1] == "s":
            return duration(v)
        return size(v * _BYTES[unit[1]])
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return f"{v:.0f}" if abs(v) >= 1000 else f"{v:.3g}"
    return str(v)


def _empty(v: Any) -> bool:
    return v is None or v == "" or (isinstance(v, Mapping | list) and not v)


def box(data: Mapping[str, Any] | Sequence[Any], key: str = "") -> str:
    """Return `data` as a box of labelled items (html)."""
    if isinstance(data, Mapping):
        items = "".join(
            f"<div class=kvi><span class=kvk>{html.escape(label(k))}</span> "
            f"{_inner(k, v)}</div>"
            for k, v in data.items()
            if not _empty(v)
        )
        return f"<div class=kv>{items}</div>"
    items = "".join(
        f"<div class=kvi>{_inner(key, v)}</div>" for v in data if not _empty(v)
    )
    return f"<div class=kv>{items}</div>"


def _inner(key: str, v: Any) -> str:
    if isinstance(v, Mapping) or (
        isinstance(v, Sequence) and not isinstance(v, str)
    ):
        return box(v, key)
    return f"<b>{html.escape(value(key, v))}</b>"
