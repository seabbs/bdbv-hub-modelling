# Copyright (c) 2026 The bdbv-hub-modelling authors. MIT licence, see LICENSE.
"""MkDocs hook of every package site: links back to the project's site.

Each package site (`INHERIT: ../mkdocs-base.yml`) ends its navigation with
the project's home page and its list of packages, from `extra.homepage`.
"""

from __future__ import annotations

from typing import Any


def on_config(config: Any, **_: Any) -> Any:
    """Add the project site and its package list to the navigation."""
    home = config["extra"]["homepage"]
    config["nav"] = [
        *(config["nav"] or []),
        {"Project site": home},
        {"All packages": f"{home}packages/"},
    ]
    return config
