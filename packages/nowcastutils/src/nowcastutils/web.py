# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Downloads of the data the studies read (hub data, benchmark files)."""

from __future__ import annotations

import urllib.request

__all__ = ["download"]


def download(url: str, token: str | None = None) -> bytes:
    """Return the bytes at `url`.

    Parameters
    ----------
    url : str
        The address.
    token : str, optional
        A GitHub token, sent to the GitHub API only.

    Returns
    -------
    bytes
    """
    headers = {"User-Agent": "bdbvhubrenewal"}
    if token and url.startswith("https://api.github.com"):
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        data: bytes = resp.read()
    return data
