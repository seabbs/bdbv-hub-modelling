"""The R simulator's latent cases, run live (`r/latent_cases.R`)."""

from __future__ import annotations

import functools
from typing import Any

from .r_live import run_r

PACKAGES = ("pkgload", "data.table", "yaml")


@functools.cache
def load(name: str) -> Any:
    """Return `latent_<name>()` from the R engine, once per session.

    Parameters
    ----------
    name : str
        One of pmfs, paths, mixing, imports, renewal or runs.

    Returns
    -------
    object
        The cases, read back from JSON.
    """
    code = f"""
    source("tests/model/r/latent_cases.R")
    latent_{name}()
    """
    return run_r(code, {}, packages=PACKAGES, na_as_null=True)
