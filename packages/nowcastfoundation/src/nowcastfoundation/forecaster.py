# Copyright (c) 2026 The bdbv-hub-modelling authors. MIT licence, see LICENSE.
r"""The Chronos-2 foundation model on the hub target's releases.

See the Foundation models methods page for the model.
One registered model per method the paper compares, each reading its
cadence from the task (`FROM_DATA`: the daily releases when there are any,
so one pass gives the daily counts and the 7-day hub target):
`foundation` (fine-tuned on simulated outbreaks, ragged layout, the hub's
model; `finetuned_checkpoint`), `foundation_zeroshot` (pretrained
weights), `foundation_complete` and `foundation_zeroshot_complete`
(complete layout), and the two-step `foundation_twostep` and
`foundation_zeroshot_twostep`. For the SI,
`foundation_real` is the ragged model with the checkpoint fine-tuned on real
data alone (the German development half and the public BDBV data to date,
`scripts/finetune_realdata.py`), pinned in `config/finetune.toml` and
released as `realdata-ft-<hash>`, which the latest lookup never takes.
`hub_triangle` lays out the hub target by place, time and release lag,
`FoundationModel` gives it to Chronos-2 as one group of series, and
`chronos_sampler` draws trajectories from Chronos-2's quantiles.
Chronos-2 is an optional dependency group (`uv sync --group chronos`).
Without it, or without the fine-tuned checkpoint's release, a fit raises
and the fit store stores it as failed.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import http.client
import json
import os
import shutil
import tempfile
import time
import tomllib
import urllib.request
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cache, partial
from pathlib import Path
from typing import Any, ClassVar, Final

import numpy as np
import pandas as pd

from nowcastbaselines.chain import start_weeks
from nowcastbaselines.randomwalk import RandomWalk
from nowcastfoundation import series_bias
from nowcastfoundation.backends import Series
from nowcastfoundation.backends.chronos2 import (
    CHRONOS_KEYS,
    CHRONOS_MODEL,
    CHRONOS_REVISION,
    _predict,
    apply_threads,
    torch_device,
)
from nowcastutils.base import (
    NATIONAL,
    SAMPLE_COLUMNS,
    ForecastTask,
    ModelUnavailableError,
    register_model,
)
from nowcastutils.dates import as_date, epiweek_end
from nowcastutils.hierarchy import province_of
from nowcastutils.trajectories import (
    choose_coupling,
    couple,
    coupling_releases,
    fit_coupling,
)

__all__ = [
    "ADAPTER_FILES",
    "CHRONOS_MODEL",
    "CHRONOS_REVISION",
    "FINETUNED_CACHE",
    "FINETUNED_CONFIG",
    "FINETUNED_ENV",
    "FINETUNED_PREFIX",
    "FINETUNED_RELEASES",
    "FINETUNED_URL",
    "LAG_QUANTILE",
    "LAYOUTS",
    "MAX_LAG",
    "PERIODS",
    "REALDATA_PREFIX",
    "SLOTS_PREFIX",
    "SLOTS_RELEASE_PREFIX",
    "SOURCES",
    "SUMS_FILE",
    "Checkpoint",
    "FoundationModel",
    "HubTriangle",
    "Known",
    "Sampler",
    "checkpoint_spec",
    "chronos_sampler",
    "finetuned_adapter",
    "finetuned_releases",
    "finetuned_spec",
    "group_key",
    "hub_triangle",
    "lag_count",
    "layout_block",
    "local_checkpoint",
    "native_sampler",
    "prefetch",
    "realdata_pinned",
    "realdata_spec",
    "register_realdata_models",
    "release_tag",
    "resolve_checkpoint",
]

SOURCES: Final = ("weekly", "daily")
"""The hub target's releases by onset week (`target.csv`) or onset day
(`daily.csv`)."""

FROM_DATA: Final = "data"
"""The source of a model that reads its cadence from each task
(`FoundationModel.for_task`): the daily releases when the task has them,
else the weekly ones. Every registered model reads it so."""

LAYOUTS: Final = ("complete", "ragged")
"""The ways the releases are given to Chronos-2."""

MAX_LAG: Final = {"weekly": 4, "daily": 28}
"""A fixed release lag $D$ in the source's steps, 4 weeks or 28 days: the
cap the SI compares the estimated lag count (`lag_count`) with, and the
triangle's default width."""

LAG_QUANTILE: Final = 0.99
"""The share of a count reported by the lag `lag_count` takes as $D$."""

COUPLINGS: Final = ("online", "independent", "comonotone")
"""How a foundation model joins its cells (`FoundationModel.coupling`):
`online` chooses and fits the coupling at each forecast
(`nowcastutils.trajectories.choose_coupling`); the other two need no fit
and are the supplement's comparisons."""

PERIODS: Final = {"daily": "rolling7"}
"""The aggregate row each place has in a block, ragged or complete, by
source (`FoundationModel.period`): the daily source's 7-day rolling sums,
from which the hub's 7-day target is read."""

FINETUNED_CONFIG: Final = Path("config/finetune.toml")
"""Which fine-tuned checkpoint the fine-tuned models load: `hash`, `"latest"`
or a settings hash."""

FINETUNED_URL: Final = (
    "https://github.com/seabbs/bdbv-hub-modelling/releases/download"
)
"""Where the fine-tuned checkpoints' releases are downloaded from."""

FINETUNED_RELEASES: Final = (
    "https://api.github.com/repos/seabbs/bdbv-hub-modelling/releases"
)
"""The repository's releases, listed to find the latest checkpoint."""

FINETUNED_PREFIX: Final = "chronos-ft"
"""Tag prefix of the simulation-trained checkpoints' releases, the only
ones `"latest"` takes."""

REALDATA_PREFIX: Final = "realdata-ft"
"""Tag prefix of the real-data checkpoints' releases."""

FINETUNED_ENV: Final = "BDBV_CHRONOS_FT"
"""Environment variable with the settings hash a run resolved once, which
then overrides `FINETUNED_CONFIG`, so its jobs never list the releases."""

SUMS_FILE: Final = "sha256sums.txt"
"""A checkpoint release's asset with the SHA-256 of its other assets."""

RETRY_SECONDS: Final = (2, 10, 30)
"""Waits before each retry of a GitHub request that failed in passing."""

ADAPTER_FILES: Final = ("adapter_config.json", "adapter_model.safetensors")
"""The LoRA adapter's files. A checkpoint's release holds them and its
`settings.json`."""

FINETUNED_CACHE: Final = "~/.cache/bdbv-hub-modelling"
"""Directory each downloaded checkpoint is kept in, as
`chronos-ft-<hash>`."""

SLOTS_PREFIX: Final = "slots-"
"""Prefix of the spec of a checkpoint with the series bias
(`nowcastfoundation.series_bias`). Such a checkpoint is released as
`chronos-slots-<hash>`, the paper's name for the run, so the `latest`
lookup, which takes only `chronos-ft-*` releases, never gives it to code
without the bias."""

SLOTS_RELEASE_PREFIX: Final = "chronos-slots"
"""Tag prefix of the releases of checkpoints with the series bias."""

FINETUNED_SOURCES: Final = ("huggingface", "github", "local")
"""Where the fine-tuned models' checkpoint is read from (`source` in
`FINETUNED_CONFIG`, `finetuned_checkpoint`): a Hugging Face Hub repository
at a pinned revision, a GitHub release, or a training segment's folder."""


def release_tag(spec: str) -> str:
    """Return the release tag of a checkpoint.

    `chronos-ft-<hash>`, or `chronos-slots-<hash>` for `slots-<hash>`.
    """
    if spec.startswith(SLOTS_PREFIX):
        return f"{SLOTS_RELEASE_PREFIX}-{spec.removeprefix(SLOTS_PREFIX)}"
    return f"{FINETUNED_PREFIX}-{spec}"


@dataclass(frozen=True)
class Checkpoint:
    """A fine-tuned checkpoint of Chronos-2, from its release.

    Attributes
    ----------
    hash : str
        Settings hash from `nowcastfoundation.finetune.stream`. The release
        is `tag`: `release_tag(hash)` for a simulation checkpoint, else
        `<prefix>-<hash>`.
    sha256 : tuple of (str, str)
        SHA-256 of each of `ADAPTER_FILES`, by file name.
    prefix : str
        The release's tag prefix, `FINETUNED_PREFIX` or `REALDATA_PREFIX`.
        Left out of the repr, which keys stored fits; the hash tells the
        checkpoints apart.
    path : str or None
        A run's own adapter folder (`local_checkpoint`), read in place
        instead of the release. Left out of the repr and of equality, so
        it is the same checkpoint as its release.
    hub : tuple of (str, str) or None
        The Hugging Face Hub repository and pinned revision the adapter is
        read from instead of the release (`hub_checkpoint`). Left out of
        the repr and of equality, as `path` is.
    """

    hash: str
    sha256: tuple[tuple[str, str], ...]
    prefix: str = field(default=FINETUNED_PREFIX, repr=False)
    path: str | None = field(default=None, repr=False, compare=False)
    hub: tuple[str, str] | None = field(default=None, repr=False, compare=False)

    @property
    def tag(self) -> str:
        """The checkpoint's release tag.

        A simulation checkpoint's is `release_tag(hash)`, so `slots-<hash>`
        is `chronos-slots-<hash>`.
        """
        if self.prefix == FINETUNED_PREFIX:
            return release_tag(self.hash)
        return f"{self.prefix}-{self.hash}"


def finetuned_spec(path: Path | str = FINETUNED_CONFIG) -> str:
    """Return the checkpoint the fine-tuned models load: `"latest"` or a hash.

    `FINETUNED_ENV`, when set, overrides the file.

    Parameters
    ----------
    path : pathlib.Path or str
        A TOML file with `hash`.

    Returns
    -------
    str
    """
    if os.environ.get(FINETUNED_ENV):
        return os.environ[FINETUNED_ENV]
    return str(tomllib.loads(Path(path).read_text())["hash"])


def active_zone_rule(
    path: Path | str = FINETUNED_CONFIG,
) -> tuple[int | None, int, int]:
    """Return the fine-tuned models' active-zone rule (`active_places`).

    `active_zone_cases` missing gives None: every zone in one block.
    `active_zone_weeks` is 8 and `active_zone_recent_weeks` 4 when
    missing.

    Parameters
    ----------
    path : pathlib.Path or str
        A TOML file, `config/finetune.toml`.

    Returns
    -------
    tuple
        The cases, the weeks and the recent weeks.
    """
    conf = tomllib.loads(Path(path).read_text())
    cases = conf.get("active_zone_cases")
    weeks = int(conf.get("active_zone_weeks", 8))
    recent = int(conf.get("active_zone_recent_weeks", 4))
    return (None if cases is None else int(cases)), weeks, recent


def _passing(err: BaseException) -> bool:
    """Whether a failed request may succeed if tried again."""
    if isinstance(err, urllib.error.HTTPError):
        return err.code >= 500 or err.code == 429
    if isinstance(err, urllib.error.URLError):
        return not isinstance(err.reason, FileNotFoundError)
    return isinstance(err, (OSError, http.client.HTTPException))


def _missing(err: BaseException) -> bool:
    """Whether a failed request means the file is not there (a 404)."""
    if isinstance(err, urllib.error.HTTPError):
        return err.code == 404
    reason = getattr(err, "reason", None)
    return isinstance(err, FileNotFoundError) or isinstance(
        reason, FileNotFoundError
    )


def _unreachable(url: str, err: BaseException) -> ConnectionError:
    """Return the error for a request refused or dropped, not missing."""
    return ConnectionError(
        f"Cannot read {url}: {err!r}. If GitHub is rate limiting, set "
        "GH_TOKEN or GITHUB_TOKEN to a GitHub token."
    )


def _get(url: str) -> bytes:
    """Read a URL, with a GitHub token from the environment if there is one.

    A server error or a dropped connection is tried again after each of
    `RETRY_SECONDS`.
    """
    request = urllib.request.Request(url)
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token and url.startswith("https://api.github.com/"):
        request.add_header("Authorization", f"Bearer {token}")
    for wait in (*RETRY_SECONDS, None):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return bytes(response.read())
        except (OSError, http.client.HTTPException) as err:
            if wait is None or not _passing(err):
                raise
            time.sleep(wait)
    raise AssertionError("unreachable")


def finetuned_releases() -> list[dict[str, Any]]:
    """List the repository's releases from the GitHub API.

    Returns
    -------
    list of dict
        Each release's `tag_name`, `published_at` and `draft`, among
        others.

    Raises
    ------
    ConnectionError
        If GitHub cannot be reached.
    """
    out: list[dict[str, Any]] = []
    for page in range(1, 11):
        url = f"{FINETUNED_RELEASES}?per_page=100&page={page}"
        try:
            batch = json.loads(_get(url))
        except (OSError, http.client.HTTPException) as err:
            raise ConnectionError(
                f"Cannot list the releases at {url} to find the fine-tuned "
                f"checkpoint: {err!r}. Set GH_TOKEN or GITHUB_TOKEN to a "
                "GitHub token (anonymous requests are rate limited), or "
                f"set {FINETUNED_ENV} to a settings hash."
            ) from err
        out += batch
        if len(batch) < 100:
            break
    return out


def realdata_spec(path: Path | str = FINETUNED_CONFIG) -> str:
    """Return the settings hash `foundation_real` load.

    No environment variable overrides it, so a run that pins the
    simulation-trained checkpoint leaves it as it is.

    Parameters
    ----------
    path : pathlib.Path or str
        A TOML file with `realdata`.

    Returns
    -------
    str

    Raises
    ------
    ValueError
        If the pin is pending (empty) until the release is published.
    """
    spec = str(tomllib.loads(Path(path).read_text())["realdata"])
    if not spec:
        raise ValueError(
            f"No {REALDATA_PREFIX} release is pinned yet: set `realdata` in "
            f"{path} to its settings hash."
        )
    return spec


@cache
def resolve_checkpoint(spec: str, prefix: str = FINETUNED_PREFIX) -> Checkpoint:
    """Return a fine-tuned checkpoint from its release.

    `"latest"` is the newest published `chronos-ft-*` release by publish
    date; releases of other prefixes, such as `realdata-ft-*`, are never
    taken. The release's `SUMS_FILE` gives the SHA-256 of its adapter
    files. Each process resolves once.

    Parameters
    ----------
    spec : str
        `"latest"` or a settings hash.
    prefix : str
        The release's tag prefix. `"latest"` needs `FINETUNED_PREFIX`.

    Returns
    -------
    Checkpoint

    Raises
    ------
    ConnectionError
        If GitHub cannot be reached. No cached checkpoint is used instead.
    LookupError
        If there is no checkpoint release.
    FileNotFoundError
        If the release has no `SUMS_FILE`.
    ValueError
        If `SUMS_FILE` lacks an adapter file.
    """
    checkpoint = spec
    if spec == "latest" and prefix != FINETUNED_PREFIX:
        raise ValueError(f"Only {FINETUNED_PREFIX} releases resolve latest.")
    if spec == "latest":
        tags = sorted(
            (r["published_at"], r["tag_name"])
            for r in finetuned_releases()
            if str(r["tag_name"]).startswith(f"{FINETUNED_PREFIX}-")
            and not r.get("draft")
            and r.get("published_at")
        )
        if not tags:
            raise LookupError("No chronos-ft-* release holds a checkpoint.")
        checkpoint = tags[-1][1].removeprefix(f"{FINETUNED_PREFIX}-")
    tag = (
        release_tag(checkpoint)
        if prefix == FINETUNED_PREFIX
        else f"{prefix}-{checkpoint}"
    )
    url = f"{FINETUNED_URL}/{tag}/{SUMS_FILE}"
    try:
        text = _get(url).decode()
    except (OSError, http.client.HTTPException) as err:
        if _missing(err):
            raise FileNotFoundError(
                f"The release {tag} has no {SUMS_FILE} ({url})."
            ) from err
        raise _unreachable(url, err) from err
    return Checkpoint(
        checkpoint, _adapter_sums(text, f"the release {tag}"), prefix
    )


def _adapter_sums(text: str, where: str) -> tuple[tuple[str, str], ...]:
    """Return the SHA-256 of each of `ADAPTER_FILES` from a `SUMS_FILE`.

    Raises `ValueError` if the file lacks one; `where` names its source.
    """
    sums = {}
    for line in text.splitlines():
        parts = line.split(maxsplit=1)
        # A line that is not `<sha> <name>` is left to the check below.
        if len(parts) == 2:
            sums[parts[1].lstrip("*").strip()] = parts[0]
    missing = [f for f in ADAPTER_FILES if f not in sums]
    if missing:
        raise ValueError(
            f"{SUMS_FILE} of {where} has no SHA-256 of {', '.join(missing)}."
        )
    return tuple((f, sums[f]) for f in ADAPTER_FILES)


def _hub_file(repo: str, revision: str, name: str) -> Path:
    """Return one file of a Hub repository at a revision, cached locally.

    `huggingface_hub` downloads it once into its own cache.

    Raises
    ------
    FileNotFoundError
        If the repository has no such file at the revision.
    LookupError
        If there is no such repository or revision.
    ConnectionError
        If the Hub cannot be reached.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import (
        EntryNotFoundError,
        LocalEntryNotFoundError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    where = f"{repo}@{revision}"
    try:
        return Path(
            hf_hub_download(repo_id=repo, filename=name, revision=revision)
        )
    except LocalEntryNotFoundError as err:
        # The Hub could not be reached and the file is not cached.
        raise ConnectionError(
            f"Cannot read {name} of {where} from the Hugging Face Hub, and "
            "it is not cached."
        ) from err
    except EntryNotFoundError as err:
        raise FileNotFoundError(f"{where} has no {name}.") from err
    except (RepositoryNotFoundError, RevisionNotFoundError) as err:
        raise LookupError(
            f"No Hugging Face repository and revision {where}."
        ) from err
    except OSError as err:
        raise ConnectionError(
            f"Cannot read {name} of {where} from the Hugging Face Hub: {err!r}."
        ) from err


@cache
def hub_checkpoint(repo: str, revision: str) -> Checkpoint:
    """Return a fine-tuned checkpoint from the Hugging Face Hub.

    The repository holds, at `revision`, the files a release holds:
    `ADAPTER_FILES`, `settings.json` and `SUMS_FILE`. The checkpoint is
    the one its release would give (`checkpoint_spec`, the SHA-256 of
    `ADAPTER_FILES` in `SUMS_FILE`), read from the Hub. Each process
    resolves once.

    Parameters
    ----------
    repo : str
        The repository id, `<owner>/<name>`.
    revision : str
        The pinned revision: a tag or a commit hash.

    Returns
    -------
    Checkpoint

    Raises
    ------
    FileNotFoundError, LookupError, ConnectionError
        As `_hub_file`.
    ValueError
        If `SUMS_FILE` lacks an adapter file.
    """
    where = f"{repo}@{revision}"
    sums = _adapter_sums(
        _hub_file(repo, revision, SUMS_FILE).read_text(), where
    )
    settings = json.loads(
        _hub_file(repo, revision, "settings.json").read_text()
    )
    return Checkpoint(checkpoint_spec(settings), sums, hub=(repo, revision))


def finetuned_checkpoint(path: Path | str = FINETUNED_CONFIG) -> Checkpoint:
    """Return the checkpoint the fine-tuned models load, from its source.

    `source` in `path` is one of `FINETUNED_SOURCES`: `huggingface`
    reads `[huggingface] repo` at `revision` (`hub_checkpoint`), `github`
    the release `hash` names (`resolve_checkpoint`), `local` the training
    segment folder `local` (`local_checkpoint`). `FINETUNED_ENV`, when
    set, overrides the file with a segment folder or a settings hash, as a
    run pins the checkpoint it resolved once; a hash is read from the Hub
    when the configured Hub revision holds it, else from its release.
    Every source is checked against the adapter's SHA-256.

    Parameters
    ----------
    path : pathlib.Path or str
        A TOML file, `config/finetune.toml`.

    Returns
    -------
    Checkpoint

    Raises
    ------
    ValueError
        For an unknown source, or a source without its settings.
    """
    pinned = os.environ.get(FINETUNED_ENV, "")
    if pinned and Path(pinned).is_dir():
        return local_checkpoint(pinned)
    conf = tomllib.loads(Path(path).read_text())
    source = str(conf.get("source", "github"))
    if source == "huggingface":
        hub = conf.get("huggingface", {})
        for key in ("repo", "revision"):
            if not hub.get(key):
                raise ValueError(
                    f"source is huggingface, but [huggingface] {key} in "
                    f"{path} is empty."
                )
        checkpoint = hub_checkpoint(str(hub["repo"]), str(hub["revision"]))
        if not pinned or pinned == checkpoint.hash:
            return checkpoint
    if pinned:
        return resolve_checkpoint(pinned)
    if source == "github":
        return resolve_checkpoint(str(conf["hash"]))
    if source == "local":
        if not conf.get("local"):
            raise ValueError(f"source is local, but local in {path} is empty.")
        return local_checkpoint(str(conf["local"]))
    raise ValueError(
        f"Unknown checkpoint source {source!r} in {path}: one of "
        f"{FINETUNED_SOURCES}."
    )


def checkpoint_spec(settings: Mapping[str, Any]) -> str:
    """Return a checkpoint's spec from its training settings.

    `slots-<hash>` for a checkpoint with the series bias, else the hash.
    """
    key = str(settings["hash"])
    return f"{SLOTS_PREFIX}{key}" if settings.get("series_bias") else key


def newest_segment(train: Path | str) -> Path | None:
    """Return a run's newest saved segment folder, or None before any.

    The largest `n<N>` under the run's `train` folder whose settings and
    adapter files (`ADAPTER_FILES`) are saved; a segment still training
    has neither. The paper's fine-tuned models load it (`paperpipe.flow`).
    """
    saved = [
        d
        for d in Path(train).glob("n*")
        if d.name[1:].isdigit()
        and (d / "settings.json").is_file()
        and all((d / "finetuned-ckpt" / f).is_file() for f in ADAPTER_FILES)
    ]
    return max(saved, key=lambda d: int(d.name[1:]), default=None)


def local_checkpoint(segment: Path | str) -> Checkpoint:
    """Return a training segment's checkpoint, read from its folder.

    The segment folder holds `settings.json` and its adapter in
    `finetuned-ckpt/`. The checkpoint is the one its release would give
    (`checkpoint_spec`, the SHA-256 of `ADAPTER_FILES`), so a run scores
    its segments before, or without, publishing them.

    Parameters
    ----------
    segment : pathlib.Path or str
        The segment folder, `<run>/train/n<N>`.

    Returns
    -------
    Checkpoint
    """
    segment = Path(segment)
    settings = json.loads((segment / "settings.json").read_text())
    adapter = segment / "finetuned-ckpt"
    sums = tuple(
        (name, hashlib.sha256((adapter / name).read_bytes()).hexdigest())
        for name in ADAPTER_FILES
    )
    return Checkpoint(checkpoint_spec(settings), sums, path=str(adapter))


def _finetuned(
    name: str, layout: str = "ragged", two_step: bool = False
) -> FoundationModel:
    """Build a fine-tuned model with the checkpoint `finetuned_checkpoint`.

    It builds the block its zero-shot model of `layout` builds, with the
    cadence read from each task (`FROM_DATA`); only the weights differ.
    """
    cases, weeks, recent = active_zone_rule()
    return FoundationModel(
        name,
        FROM_DATA,
        layout,
        finetuned_checkpoint(),
        two_step=two_step,
        active_zone_cases=cases,
        active_zone_weeks=weeks,
        active_zone_recent_weeks=recent,
    )


def _real_data(name: str) -> FoundationModel:
    """Build a model with the real-data checkpoint `realdata_spec`."""
    return FoundationModel(
        name,
        FROM_DATA,
        "ragged",
        resolve_checkpoint(realdata_spec(), REALDATA_PREFIX),
    )


Sampler = Callable[[Sequence[Series], int, int, int], list[np.ndarray]]
"""Draws trajectories: `(groups, horizon, n, seed)` to one
`[n, V, horizon]` array per group."""


@dataclass(frozen=True)
class HubTriangle:
    """The hub target by place, time step and release lag.

    Attributes
    ----------
    times : list of str
        Consecutive weeks (ISO Saturdays) or days.
    final : numpy.ndarray
        The count of each place and step at the latest release, `[L, T]`.
    partial : numpy.ndarray
        The count of each place and step at release lag $d < D$, from the
        release whose cut-off is $d$ steps after it, NaN where no release
        is, `[L, T, D]`.
    last : int
        Index of the step holding the latest release's cut-off.
    """

    times: list[str]
    final: np.ndarray
    partial: np.ndarray
    last: int

    @property
    def n_context(self) -> int:
        """Steps whose count at the latest release is at lag $D$ or more."""
        return self.last - int(self.partial.shape[2]) + 1


def _step(source: str, day: str) -> dt.date:
    """Return the step holding a day: its week's Saturday, or the day."""
    return epiweek_end(day) if source == "weekly" else as_date(day)


@dataclass(frozen=True)
class PeriodRows:
    """Each place's period totals on a daily grid, for its total row.

    Attributes
    ----------
    ends : numpy.ndarray
        Whether each day of the grid ends a whole period, `[T]`.
    totals : numpy.ndarray
        Each place's total over the period ending on each such day, NaN on
        every other day, `[L, T]`.
    """

    ends: np.ndarray
    totals: np.ndarray


def rolling_totals(days: np.ndarray, k: int) -> np.ndarray:
    """Return each place's `k`-day sums ending on each day.

    Parameters
    ----------
    days : numpy.ndarray
        Daily counts, `[L, T]`.
    k : int
        Days in each sum.

    Returns
    -------
    numpy.ndarray
        `[L, T]`, the sum of days `t - k + 1` to `t` on day `t`, NaN on
        the first `k - 1` days.
    """
    out = np.full(days.shape, np.nan)
    if days.shape[1] >= k:
        sums = np.cumsum(np.pad(days, ((0, 0), (1, 0))), axis=1)
        out[:, k - 1 :] = sums[:, k:] - sums[:, :-k]
    return out


def rolling_rows(tri: HubTriangle, k: int = 7) -> PeriodRows:
    """Return each place's `k`-day sums of a daily triangle's counts.

    Every day from the grid's `k`-th holds the sum of its `k` days at the
    latest release (`rolling_totals`), so a Saturday holds its week's
    total and every other day its own 7-day sum, as the `daily+rolling7`
    training blocks have.
    """
    ends = np.arange(len(tri.times)) >= k - 1
    return PeriodRows(ends, rolling_totals(tri.final, k))


def hub_triangle(
    rows: pd.DataFrame,
    source: str,
    places: Sequence[str],
    last_time: str,
    max_lag: int | None = None,
    *,
    all_lags: bool = False,
) -> HubTriangle:
    """Lay out the hub target of every release by place, step and lag.

    Parameters
    ----------
    rows : pandas.DataFrame
        The releases. The latest gives the final counts. Weekly rows have
        `target_end_date` and `total`; daily rows have `onset_date`,
        `observed` and `imputed`. Both have `vintage`, `location` and
        `cutoff`.
    source : str
        One of `SOURCES`.
    places : sequence of str
        Places, in the order of the output.
    last_time : str
        Last step of the grid.
    max_lag : int, optional
        Release lag $D$, `MAX_LAG` of the source by default.
    all_lags : bool
        Every lag the grid holds, its whole release history, in place of
        `max_lag` (what `lag_count` reads).

    Returns
    -------
    HubTriangle
        Steps from the first in `rows` to `last_time`.
    """
    width = 7 if source == "weekly" else 1
    time_col = "target_end_date" if source == "weekly" else "onset_date"
    rows = rows.loc[rows["location"].isin(places)]
    if rows.empty:
        raise ValueError(f"no release rows for {list(places)}")
    value = (
        rows["total"]
        if source == "weekly"
        else rows["observed"] + rows["imputed"]
    ).to_numpy()
    first = _step(source, str(rows[time_col].min()))
    n_times = (as_date(last_time) - first).days // width + 1
    times = [
        (first + dt.timedelta(days=width * i)).isoformat()
        for i in range(n_times)
    ]
    index_of = {c: i for i, c in enumerate(places)}
    codes, unique = pd.factorize(rows["location"])
    place = np.array([index_of[c] for c in unique], dtype=int)[codes]
    # Each distinct date and release is parsed once; the rows of a release
    # are found by sorting, not by comparing every row to it.
    codes, unique = pd.factorize(rows[time_col])
    step = (pd.to_datetime(unique) - pd.Timestamp(first)).days.to_numpy()[
        codes
    ] // width
    release_of, releases = pd.factorize(rows["vintage"], sort=True)
    order = np.argsort(release_of, kind="stable")
    bounds = np.searchsorted(release_of[order], np.arange(len(releases) + 1))
    cutoffs = rows["cutoff"].to_numpy()
    lags = (
        n_times if all_lags else MAX_LAG[source] if max_lag is None else max_lag
    )
    partial = np.full((len(places), n_times, lags), np.nan)
    final = np.zeros((len(places), n_times))
    last = 0
    for r in range(len(releases)):
        mine = order[bounds[r] : bounds[r + 1]]
        cutoff = str(cutoffs[mine[0]])
        at = (_step(source, cutoff) - first).days // width
        counts = np.zeros((len(places), n_times))
        keep = mine[(step[mine] >= 0) & (step[mine] < n_times)]
        np.add.at(counts, (place[keep], step[keep]), value[keep])
        d = np.arange(lags)
        d = d[(at - d >= 0) & (at - d < n_times)]
        partial[:, at - d, d] = counts[:, at - d]
        # Sorted, so the last release's counts and cut-off are the latest.
        final, last = counts, at
    return HubTriangle(times, final, partial, last)


def lag_count(
    tri: HubTriangle, source: str, quantile: float = LAG_QUANTILE
) -> int:
    """Return the block's lag count $D$ from the data's reporting delay.

    Looking back from the cut-off over every release, each step first
    published after releases began (not backfilled by the first release)
    settles at the first lag at which its reported count reaches
    `quantile` of its latest count. $D - 1$ is the `quantile` (99th
    percentile) of the settling lags over steps and places with a count,
    using only steps old enough to have settled within the release
    history: seen for at least $2D$ lags, so their latest count stands
    for their final one and recent steps, which have not yet seen their
    long delays, do not bias it short (right truncation). $D$ is the
    smallest count that meets this. A lag no release fell on carries the
    count of the lag before (weekly releases of daily data). Steps before
    the first release are not evidence: a first release that backfills
    earlier onsets (BDBV's daily releases begin months after its first
    onsets) measures when releases began, not a delay. With no such
    step, the block takes its release history, every step from the
    first release to the cut-off. $D$ is capped only by the history: two
    steps of context before the longest lag, and the lags `tri` holds.
    Training blocks (`nowcastfoundation.finetune.training_block`) and
    forecasts (`FoundationModel`) use this one rule.

    Parameters
    ----------
    tri : HubTriangle
        The triangle, with every lag of its history (`hub_triangle` with
        `all_lags`).
    source : str
        One of `SOURCES`; the rule is the same in either's steps.
    quantile : float
        The share of its latest count at which a step settles, and the
        percentile of the settling lags, in (0, 1].

    Returns
    -------
    int
        $D$, at least 1.

    Raises
    ------
    ValueError
        If `quantile` is not in (0, 1].
    """
    if not 0 < quantile <= 1:
        raise ValueError(f"lag quantile {quantile} is not in (0, 1]")
    del source
    width = int(tri.partial.shape[2])
    most = int(max(1, min(width, tri.last - 1)))
    # The first release: the earliest step any cell was released on.
    _, t, d = np.nonzero(~np.isnan(tri.partial))
    if t.size == 0:
        return most
    released = int((t + d).min())
    history = int(max(1, min(most, tri.last - released + 1)))
    # Steps on the grid: a grid of targets before the cut-off ends early.
    steps = np.arange(released, min(tri.last, tri.final.shape[1] - 1) + 1)
    if steps.size == 0:
        return history
    final = tri.final[:, steps]
    seen = np.nan_to_num(
        pd.DataFrame(tri.partial[:, steps, :width].reshape(-1, width))
        .ffill(axis=1)
        .to_numpy()
    ).reshape(final.shape[0], len(steps), width)
    # Each step's settling lag: the first at which it reaches `quantile`
    # of its latest count (its age, past the lags it was seen at, if not).
    age = tri.last - steps
    settled = seen >= quantile * final[:, :, None] - 1e-9
    first = np.where(settled.any(axis=2), settled.argmax(axis=2), width)
    first = np.minimum(first, age[None, :])
    counted = final > 0
    for lags in range(1, history + 1):
        use = counted & (age >= 2 * lags)[None, :]
        if not use.any():
            break
        if np.quantile(first[use], quantile, method="higher") + 1 <= lags:
            return lags
    return history


def finetuned_adapter(checkpoint: Checkpoint) -> Path:
    """Return a fine-tuned checkpoint's LoRA adapter, downloading it once.

    The release `<prefix>-<hash>` at `FINETUNED_URL` holds
    `ADAPTER_FILES` and `settings.json`. They are kept in
    `FINETUNED_CACHE` only once all are downloaded, the settings hash
    matches and each adapter file has its SHA-256. A cached copy is
    checked against the SHA-256 again on each use. A run's own
    checkpoint (`local_checkpoint`) is its folder, checked the same way.

    Parameters
    ----------
    checkpoint : Checkpoint
        The checkpoint.

    Returns
    -------
    pathlib.Path
        The adapter directory.

    Raises
    ------
    FileNotFoundError
        If the release has no such asset.
    ConnectionError
        If GitHub refuses the request or the connection fails.
    ValueError
        If the release's settings hash or an adapter file's SHA-256 is not
        the checkpoint's.
    """
    if checkpoint.hub is not None:
        return _hub_adapter(checkpoint)
    if checkpoint.path is not None:
        local = Path(checkpoint.path)
        for name, expected in checkpoint.sha256:
            sha = hashlib.sha256((local / name).read_bytes()).hexdigest()
            if sha != expected:
                raise ValueError(
                    f"{name} in {local} has SHA-256 {sha}, not {expected}."
                )
        return local
    tag = checkpoint.tag
    cache = Path(FINETUNED_CACHE).expanduser()
    out = cache / tag
    if out.is_dir():
        if all(
            (out / name).is_file()
            and hashlib.sha256((out / name).read_bytes()).hexdigest() == sha
            for name, sha in checkpoint.sha256
        ):
            return out
        # A cached copy that has changed is downloaded again.
        shutil.rmtree(out)
    cache.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=cache, prefix=f".{tag}-"))
    try:
        for name in (*ADAPTER_FILES, "settings.json"):
            url = f"{FINETUNED_URL}/{tag}/{name}"
            try:
                (tmp / name).write_bytes(_get(url))
            except (OSError, http.client.HTTPException) as err:
                if _missing(err):
                    raise FileNotFoundError(
                        f"The release {tag} has no {name} ({url})."
                    ) from err
                raise _unreachable(url, err) from err
        found = json.loads((tmp / "settings.json").read_text()).get("hash")
        if found != checkpoint.hash.removeprefix(SLOTS_PREFIX):
            raise ValueError(
                f"The release {tag} holds the checkpoint with settings "
                f"hash {found}."
            )
        for name, expected in checkpoint.sha256:
            sha = hashlib.sha256((tmp / name).read_bytes()).hexdigest()
            if sha != expected:
                raise ValueError(
                    f"{name} of the release {tag} has SHA-256 {sha}, not "
                    f"{expected}."
                )
        try:
            tmp.rename(out)
        except OSError:
            # Another process cached it first.
            if not out.is_dir():
                raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def _hub_adapter(checkpoint: Checkpoint) -> Path:
    """Return a Hub checkpoint's adapter folder, checked as a release is.

    `huggingface_hub` keeps the files of one revision in one folder.
    """
    assert checkpoint.hub is not None
    repo, revision = checkpoint.hub
    where = f"{repo}@{revision}"
    files = {
        name: _hub_file(repo, revision, name)
        for name in (*ADAPTER_FILES, "settings.json")
    }
    found = json.loads(files["settings.json"].read_text()).get("hash")
    if found != checkpoint.hash.removeprefix(SLOTS_PREFIX):
        raise ValueError(
            f"{where} holds the checkpoint with settings hash {found}."
        )
    for name, expected in checkpoint.sha256:
        sha = hashlib.sha256(files[name].read_bytes()).hexdigest()
        if sha != expected:
            raise ValueError(
                f"{name} of {where} has SHA-256 {sha}, not {expected}."
            )
    (folder,) = {p.parent for p in files.values()}
    return folder


@cache
def _chronos(checkpoint: Checkpoint | None = None) -> Any:
    adapter = None if checkpoint is None else finetuned_adapter(checkpoint)
    from chronos import Chronos2Pipeline

    apply_threads()
    device = torch_device()
    pipeline = Chronos2Pipeline.from_pretrained(
        CHRONOS_MODEL, revision=CHRONOS_REVISION, device_map=device
    )
    if adapter is None:
        return pipeline
    # Slots, if the adapter holds them, load with it.
    return Chronos2Pipeline(
        model=series_bias.load_adapter(pipeline.model, adapter)
    )


Known = Mapping[str, np.ndarray]
"""Chronos-2's native quantiles of groups, `[V, horizon, Q]`, by
`group_key`, predicted ahead (`prefetch`)."""

BlockViews = tuple[
    Sequence[tuple[tuple[str, ...], tuple[str, ...]]], Sequence[str], bool
]
"""A block cut into views: the views (`block_views`), the block's places,
and whether its places have aggregate rows (`_owned`)."""


def group_key(
    group: Series,
    horizon: int,
    levels: Sequence[float] | np.ndarray | None = None,
) -> str:
    """Return a key of what Chronos-2 sees of a group, and the horizon.

    The rows' places are in it too: with the series bias, the same numbers
    at other places forecast differently. Quantiles at `levels` other than
    Chronos-2's own are keyed apart.
    """
    h = hashlib.sha256(str(horizon).encode())
    if levels is not None:
        h.update(f"levels={[float(x) for x in levels]!r}".encode())
    for name in ("places", "target_places"):
        h.update(f"{name}={list(group.get(name) or [])!r}".encode())
    for name in CHRONOS_KEYS:
        part = group.get(name)
        if part is None:
            continue
        h.update(name.encode())
        if isinstance(part, Mapping):
            for k in sorted(part):
                h.update(k.encode())
                h.update(np.ascontiguousarray(part[k], dtype=float).tobytes())
        else:
            h.update(np.ascontiguousarray(part, dtype=float).tobytes())
    return h.hexdigest()


def _native(
    pipeline: Any,
    groups: Sequence[Series],
    horizon: int,
    known: Known | None = None,
    levels: Sequence[float] | np.ndarray | None = None,
) -> list[np.ndarray]:
    """Return each group's quantiles, `[V, horizon, Q]`.

    At Chronos-2's own levels by default, or at `levels`, which the
    pipeline interpolates between its own (`predict_quantiles`). Groups in
    `known` are read from it; the rest are predicted together.
    """
    keys = [group_key(g, horizon, levels) for g in groups]
    at = np.asarray(
        pipeline.quantiles if levels is None else levels, dtype=float
    )
    out: list[np.ndarray | None] = [
        None if known is None else known.get(k) for k in keys
    ]
    todo = [i for i, q in enumerate(out) if q is None]
    if todo:
        got = _predict(pipeline, [groups[i] for i in todo], horizon, at)
        for i, q in zip(todo, got, strict=True):
            out[i] = q.float().cpu().numpy()
    return [np.asarray(q) for q in out]


def chronos_sampler(
    groups: Sequence[Series],
    horizon: int,
    n_samples: int,
    seed: int,
    checkpoint: Checkpoint | None = None,
    known: Known | None = None,
    coupling: Mapping[str, Any] | None = None,
    views: BlockViews | None = None,
) -> list[np.ndarray]:
    """Draw trajectories from Chronos-2's quantiles.

    Each cell follows its quantiles, and `coupling` (a
    `nowcastutils.trajectories.fit_coupling` record) joins the cells
    (`couple`); with none, every cell draws its own level. With `views`,
    the groups are a block's views: each place's quantiles are taken from
    the view that forecasts it (`_owned`) and the block is coupled once,
    so one fit joins every place. Runs on a CUDA or Apple GPU when torch
    sees one.

    Parameters
    ----------
    groups : sequence of Series
        The groups of series.
    horizon : int
        Steps to draw.
    n_samples : int
        Trajectories.
    seed : int
        Seed for the uniform draws.
    checkpoint : Checkpoint, optional
        The fine-tuned checkpoint (`finetuned_adapter`) to load into
        Chronos-2. The pretrained model by default.
    known : mapping, optional
        Quantiles predicted ahead (`prefetch`), used where they hold a
        group.
    coupling : mapping, optional
        The coupling's fit, with its `method`; `independent` by default.
    views : BlockViews, optional
        The block's views, its places and whether it has aggregate rows,
        when `groups` are the views of one block.

    Returns
    -------
    list of numpy.ndarray
        Draws, `[n_samples, V, horizon]` per group, or the block's one
        with `views`.

    Raises
    ------
    ImportError
        If Chronos-2 is not installed (`uv sync --group chronos`).
    FileNotFoundError
        If the checkpoint's release has no such asset.
    ConnectionError
        If GitHub cannot be reached.
    ValueError
        If the release holds another checkpoint or adapter.
    """
    pipeline = _chronos(checkpoint)
    levels = np.asarray(pipeline.quantiles, dtype=float)
    fit = dict(coupling) if coupling else {"method": "independent"}
    rng = np.random.default_rng(seed)
    native = _native(pipeline, groups, horizon, known)  # [V, horizon, Q]
    if views is not None:
        parts = [np.moveaxis(q, -1, 0) for q in native]  # [Q, V, horizon]
        native = [np.moveaxis(_owned(parts, *views), 0, -1)]
    out = []
    for q in native:
        own = int(rng.integers(0, 2**31 - 1))
        out.append(couple(q, levels, fit["method"], fit, n_samples, own))
    return out


def layout_block(
    tri: HubTriangle,
    layout: str,
    places: Sequence[str],
    start: int,
    period: PeriodRows | None = None,
) -> Series:
    """Return the one block of every place's series, split at `start`.

    Parameters
    ----------
    tri : HubTriangle
        The hub target by place, step and lag.
    layout : str
        One of `LAYOUTS`.
    places : sequence of str
        Places, in the order of `tri`.
    start : int
        Steps of context, the first `start` steps of `tri`.
    period : PeriodRows, optional
        Each place's period totals on `tri`'s grid. With them each place
        has a second target row, of its totals, after the count rows; it
        has no lags.

    Returns
    -------
    Series
        The targets' context, the places (`places`), the place of each
        target row (`target_places`, indices into `places`), and for
        `ragged` each place's lag series as past and future covariates.
    """
    target = tri.final[:, :start]
    target_places = list(range(len(places)))
    if period is not None:
        target = np.concatenate([target, period.totals[:, :start]])
        target_places += target_places
    if layout == "complete":
        return {
            "target": target,
            "places": list(places),
            "target_places": target_places,
        }
    lags = tri.partial.shape[2]
    names = [f"{p}_lag_{d}" for p in places for d in range(lags)]
    cells = tri.partial.transpose(0, 2, 1).reshape(len(names), -1)
    return {
        "target": target,
        "past_covariates": dict(zip(names, cells[:, :start], strict=True)),
        "future_covariates": dict(zip(names, cells[:, start:], strict=True)),
        "places": list(places),
        "target_places": target_places,
    }


BLOCK_LIMIT_ENV: Final = "CHRONOS_BLOCK_LIMIT"
"""The environment variable of `FoundationModel.block_limit`."""


def block_views(
    places: Sequence[str], cap: int
) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    """Return the views of a block's places, at most `cap` places each.

    Each view is its places and the places it forecasts; every place is
    forecast by exactly one view. Within `cap`, the whole block. Over it,
    as training's `system_views`: the country with its provinces (or, if
    they do not fit, chunks of provinces under the country, the first
    forecasting the country) forecasts them, and each province with as
    many of its zones as fit forecasts those zones. Every place is its own
    series of the task; nothing is summed. `cap` is at least 2.
    """
    places = tuple(places)
    if len(places) <= cap:
        return [(places, places)]
    step = max(1, cap - 1)
    zones = [
        p for p in places if p.count("-") >= 2 and province_of(p) in places
    ]
    tops = [p for p in places if p not in zones]
    country, provinces = tops[0], tops[1:]
    out: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    if len(tops) <= cap:
        out.append((tuple(tops), tuple(tops)))
    else:
        for i in range(0, len(provinces), step):
            chunk = tuple(provinces[i : i + step])
            mine = (country, *chunk) if i == 0 else chunk
            out.append(((country, *chunk), mine))
    for province in provinces:
        own = [z for z in zones if province_of(z) == province]
        for i in range(0, len(own), step):
            chunk = tuple(own[i : i + step])
            out.append(((province, *chunk), chunk))
    return out


def _sub(
    tri: HubTriangle, places: Sequence[str], view: Sequence[str]
) -> HubTriangle:
    """Return `tri`'s rows of the places of `view`."""
    at = [list(places).index(p) for p in view]
    return replace(tri, final=tri.final[at], partial=tri.partial[at])


def _owned(
    parts: Sequence[np.ndarray],
    views: Sequence[tuple[tuple[str, ...], tuple[str, ...]]],
    places: Sequence[str],
    period: bool,
) -> np.ndarray:
    """Return each place's draws from the view that forecasts it.

    Rows as one block of `places`: the count rows, then (with `period`)
    the aggregate rows.
    """
    n, _, h = parts[0].shape
    k = len(places)
    out = np.zeros((n, k * (1 + period), h))
    for (view, mine), draws in zip(views, parts, strict=True):
        for p in mine:
            i, j = list(places).index(p), view.index(p)
            out[:, i] = draws[:, j]
            if period:
                out[:, k + i] = draws[:, len(view) + j]
    return out


def _zones(places: Sequence[str]) -> list[str]:
    """Return the places that are zones of a province in `places`."""
    return [p for p in places if p.count("-") >= 2 and province_of(p) in places]


def active_places(
    places: Sequence[str],
    tri: HubTriangle,
    cases: int,
    weeks: int,
    recent_weeks: int,
    source: str,
) -> set[str]:
    """Return the active zones.

    A zone is active with at least `cases` cases in the last `weeks`
    weeks and at least one in the last `recent_weeks` weeks (0: no such
    rule), as reported at the latest release, over the steps that end
    with its cut-off.
    """
    per = 7 if source == "daily" else 1
    end = tri.last + 1

    def total(n_weeks: int) -> np.ndarray:
        start = max(0, end - n_weeks * per)
        return np.nansum(tri.final[:, start:end], axis=1)

    window = total(weeks)
    lately = total(recent_weeks) if recent_weeks else np.ones(len(places))
    at = {p: i for i, p in enumerate(places)}
    return {
        z
        for z in _zones(places)
        if window[at[z]] >= cases and lately[at[z]] >= 1
    }


def active_views(
    places: Sequence[str], active: Collection[str], cap: int | None
) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    """Return the views of a block whose joint block holds active zones.

    The joint block is the country, every province and the `active`
    zones (cut into `block_views` over `cap`). Each province has a block
    of itself and all its zones (cut into chunks of zones under the
    province over `cap`, as training cuts), which forecasts its zones not
    in the joint block. Every place is forecast by exactly one view.
    """
    places = tuple(places)
    zones = _zones(places)
    rest = [z for z in zones if z not in active]
    if not rest:
        return block_views(places, cap) if cap else [(places, places)]
    main = tuple(p for p in places if p not in rest)
    out = block_views(main, cap) if cap else [(main, main)]
    step = max(1, cap - 1) if cap else len(zones)
    for province in dict.fromkeys(province_of(z) for z in zones):
        own = [z for z in zones if province_of(z) == province]
        for i in range(0, len(own), step):
            chunk = tuple(own[i : i + step])
            mine = tuple(z for z in chunk if z in rest)
            if mine:
                out.append(((province, *chunk), mine))
    return out


TWO_STEP_DELAY: Final = 40
"""The two-step model's longest delay $D$ in days, capped by the history."""


def point_nowcast(
    tri: HubTriangle, places: Sequence[str], task: ForecastTask
) -> HubTriangle:
    """Return `tri` with each place's days to the cut-off nowcast.

    Each place's daily triangle at the cut-off (`round(3 * D)` days,
    $D$ = `TWO_STEP_DELAY` capped by the history, as
    `randomwalknowcast_daily`) is filled by baselinenowcast's point
    nowcast, and those days of the place's counts are replaced by it.
    """
    from nowcastbaselines.baselinenowcast import baselinenowcast
    from nowcastbaselines.daily import daily_triangle

    history = task.daily_history
    if history is None or history.empty:
        raise ModelUnavailableError("the two-step model needs daily releases")
    cutoff = as_date(str(task.cutoff)[:10])
    at = tri.times.index(cutoff.isoformat())
    have = (cutoff - as_date(str(history["onset_date"].min()))).days + 1
    delay = min(TWO_STEP_DELAY, have // 3)
    if delay < 3:
        raise ModelUnavailableError(f"{have} days of releases are too few")
    rows = min(3 * delay, at + 1)
    final = tri.final.copy()
    for i, place in enumerate(places):
        cells, mask = daily_triangle(history, place, cutoff, rows, delay)
        point = baselinenowcast(cells, mask, output_type="point")
        final[i, at - rows + 1 : at + 1] = np.asarray(point, np.float64)
    return replace(tri, final=final)


def _walk_start(task: ForecastTask) -> str:
    """Return the earliest week `randomwalk` starts its walk from.

    It is read from the delay alone, so it exists where the walk has no
    spread.
    """
    walk = RandomWalk()
    snap = walk.snapshots(task)
    return str(snap.week_ends[int(start_weeks(snap, walk.settings).min())])


def _weekly_from_totals(
    drawn: np.ndarray,
    period: PeriodRows,
    tri: HubTriangle,
    start: int,
    weeks: Sequence[str],
) -> np.ndarray:
    """Return weekly draws `[S, L, W]` read from the total rows.

    A week whose end is past the context takes its total row's draw
    there; an earlier week takes its total summed from the daily counts.
    `drawn` is `[S, 2L, horizon]`, the count rows then the total rows.
    """
    n_places = period.totals.shape[0]
    out = []
    for week in weeks:
        t = tri.times.index(week)
        if t >= start:
            out.append(drawn[:, n_places:, t - start])
        else:
            total = np.nan_to_num(period.totals[:, t])
            out.append(np.broadcast_to(total, (drawn.shape[0], n_places)))
    return np.stack(out, axis=2)


DAILY_COLUMNS: Final = ("location", "date", "sample", "value")
"""Columns of the daily trajectories (`FoundationModel.forecast_both`)."""


def _daily_frame(
    drawn: np.ndarray,
    tri: HubTriangle,
    start: int,
    places: Sequence[str],
    weeks: Sequence[str],
) -> pd.DataFrame:
    """Return each day of the target weeks from the daily count rows.

    A day past the context takes its count row's draw; an earlier day the
    count reported. `drawn` is
    `[S, V, horizon]`, the count rows first.
    """
    first = as_date(weeks[0]) - dt.timedelta(days=6)
    days = [t for t in tri.times if first <= as_date(t) <= as_date(weeks[-1])]
    n, n_loc = drawn.shape[0], len(places)
    cols = []
    for day in days:
        t = tri.times.index(day)
        final = np.broadcast_to(tri.final[:, t], (n, n_loc))
        cols.append(drawn[:, :n_loc, t - start] if t >= start else final)
    values = np.stack(cols, axis=2)  # [S, L, D]
    return pd.DataFrame(
        {
            "location": np.repeat(list(places), n * len(days)),
            "date": np.tile(days, n * n_loc),
            "sample": np.tile(np.repeat(np.arange(1, n + 1), len(days)), n_loc),
            "value": values.transpose(1, 0, 2).reshape(-1),
        }
    )[list(DAILY_COLUMNS)]


def _rows(*tables: pd.DataFrame | None) -> pd.DataFrame | None:
    """Stack the tables that are there, None if none is."""
    there = [t for t in tables if t is not None and not t.empty]
    return pd.concat(there, ignore_index=True) if there else None


@dataclass
class FoundationModel:
    """Chronos-2 on one source and layout of the hub target's releases.

    Attributes
    ----------
    name : str
        Model name.
    source : str
        One of `SOURCES`, or `FROM_DATA` to read it from each task
        (`for_task`).
    layout : str
        One of `LAYOUTS`.
    checkpoint : Checkpoint or None
        The fine-tuned checkpoint Chronos-2 loads, or None for the
        pretrained model.
    sample : Sampler or None
        Draws the trajectories, Chronos-2 with `checkpoint` by default.
    stream : str or None
        The task's random stream the draws are seeded from, the model's
        name by default. Models given one stream draw the same random
        numbers (`nowcasteval.marginals.forecast_quantiles`).
    known : mapping or None
        Chronos-2's quantiles of tasks predicted ahead in few batched
        passes (`prefetch`); others are predicted when forecast.
    lags : int or None
        A fixed lag count $D$; None estimates it from each task's data
        (`lag_count`).
    coupling : str
        How Chronos-2's cells are joined into trajectories, one of
        `COUPLINGS`. `online` (the default) chooses the coupling and its
        look-back window per task from the model's own forecasts of the
        same block at the task's earlier releases and the series' own
        history, and fits it (`_coupling_fit`); `independent` and
        `comonotone` need no fit and are the SI's comparisons.
    copula_fit : dict or None
        The coupling the latest forecast drew by (`_coupling_fit`'s
        record, without the Schaake windows themselves), or None before
        one; the submission writes it beside the forecast.
    collect : list or None
        Private to `prefetch`: the groups a forecast would predict are
        added to it, and the forecast stops.
    two_step : bool
        The two-step model (#538's sensitivity analysis): each place's
        daily counts to the cut-off are first replaced by baselinenowcast's
        point nowcast (`point_nowcast`), and Chronos-2 forecasts from the
        day after the cut-off on those corrected counts. The nowcast days
        are the point nowcast, with no uncertainty. Complete layout, daily
        source only.
    block_limit : int or None
        Most series in one block Chronos-2 predicts, the machine's block
        capacity; a task's block over it is cut into views
        (`block_views`), as training cuts a system over its limit. None
        (the default) reads `CHRONOS_BLOCK_LIMIT` from the environment,
        else no limit.
    active_zone_cases : int or None
        With a value, the joint block holds the country, every province
        and the active zones: at least this many cases in the last
        `active_zone_weeks` weeks and one in the last
        `active_zone_recent_weeks` (`active_places`). Every other zone
        takes its forecast from its province's block, the province and
        all its zones (`active_views`). None (the default): every zone in
        the joint block.
    active_zone_weeks : int
        The weeks to the cut-off `active_zone_cases` counts over.
    active_zone_recent_weeks : int
        The weeks to the cut-off an active zone has a case in.
    """

    name: str
    source: str
    layout: str
    checkpoint: Checkpoint | None = None
    sample: Sampler | None = field(default=None, repr=False)
    stream: str | None = field(default=None, repr=False)
    known: Known | None = field(default=None, repr=False, compare=False)
    lags: int | None = None
    coupling: str = "online"
    copula_fit: dict[str, Any] | None = field(
        default=None, repr=False, compare=False
    )
    collect: list[tuple[Series, int]] | None = field(
        default=None, repr=False, compare=False
    )
    block_limit: int | None = None
    two_step: bool = False
    active_zone_cases: int | None = None
    active_zone_weeks: int = 8
    active_zone_recent_weeks: int = 4
    reads_zones: ClassVar[bool] = True

    def __post_init__(self) -> None:
        """Check the coupling; read the block limit from the environment."""
        if self.coupling not in COUPLINGS:
            raise ValueError(
                f"unknown coupling {self.coupling!r}; one of {COUPLINGS}"
            )
        if self.two_step and (
            self.layout != "complete" or self.source not in ("daily", FROM_DATA)
        ):
            raise ValueError("two_step needs the complete layout, daily")
        if self.block_limit is None and os.environ.get(BLOCK_LIMIT_ENV):
            self.block_limit = int(os.environ[BLOCK_LIMIT_ENV])

    @property
    def period(self) -> str | None:
        """The aggregate row each place has, by source (`PERIODS`).

        The daily source's 7-day rolling sums (`rolling_rows`, as the
        daily+rolling7 training blocks), from which the hub's 7-day sums
        are read; None for the weekly source.
        """
        return PERIODS.get(self.source)

    def for_task(self, task: ForecastTask) -> FoundationModel:
        """Return the model with its source for `task`.

        With `FROM_DATA`, the daily releases when the task has them, so one
        pass gives the daily counts and the 7-day target from its 7-day
        rows, else the weekly releases. A fixed source is kept.
        """
        if self.source != FROM_DATA:
            return self
        daily = task.daily_history is not None and not task.daily_history.empty
        return replace(self, source="daily" if daily else "weekly")

    @property
    def needs_snapshots(self) -> bool:
        """Whether the model reads release snapshots: `complete` does."""
        return self.layout == "complete"

    def forecast_both(
        self, task: ForecastTask
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return the weekly and the daily trajectories of one draw.

        The weekly ones are `forecast`'s. The daily ones are the daily
        source's count rows, each day of the target weeks; empty for the
        weekly source.

        Parameters
        ----------
        task : ForecastTask
            The data and weeks to forecast. Its `target_history` and
            `zone_history` hold the weekly releases, its `daily_history`
            and `zone_daily_history` the daily ones, and its `zone_codes`
            the zones.

        Returns
        -------
        tuple of pandas.DataFrame
            The weekly trajectories, columns `location`, `target_end_date`,
            `sample`, `value`, for the country, the provinces and their
            zones, each its own series of the block, nothing summed or
            split; and the daily
            ones, columns `location`, `date`, `sample`, `value`.

        Raises
        ------
        ModelUnavailableError
            If the task has no releases of the source or fewer than two
            steps of context, or, for `complete`, no week `randomwalk`
            can start from.
        ImportError
            If Chronos-2 is not installed.
        FileNotFoundError
            If the fine-tuned checkpoint's release has no such asset.
        ConnectionError
            If GitHub cannot be reached.
        ValueError
            If the release holds another checkpoint or adapter.
        """
        run = self.for_task(task)
        if run is not self:
            out = run.forecast_both(task)
            self.copula_fit = run.copula_fit
            return out
        rows, places, weeks = self._rows(task)
        tri = self._triangle(rows, places, weeks[-1])
        start = tri.n_context
        if self.two_step:
            # Chronos-2 forecasts from the day after the cut-off, on the
            # counts baselinenowcast's point nowcast corrects.
            tri = point_nowcast(tri, places, task)
            start = tri.times.index(str(task.cutoff)[:10]) + 1
        elif self.layout == "complete":
            walk = _walk_start(task)
            if walk not in tri.times:
                raise ModelUnavailableError(
                    f"{self.name} starts from the randomwalk week {walk}, "
                    f"which the task for {task.vintage} has no "
                    f"{self.source} release of."
                )
            start = tri.times.index(walk) + 1
        if start < 2:
            raise ModelUnavailableError(
                f"{self.name} needs two steps of context, and the task for "
                f"{task.vintage} has {max(start, 0)}."
            )
        horizon = len(tri.times) - start
        seed = int(task.rng(self.stream or self.name).integers(0, 2**31 - 1))
        period = self._period(tri)
        if horizon <= 0:
            # Every step of the target weeks is complete at the latest
            # release (the target ends at least the lag count before the
            # cut-off): the counts reported, with no forecast.
            start = len(tri.times)
            drawn = np.zeros((task.n_samples, len(places), 0))
        else:
            # Zero-shot and fine-tuned models build the same block, the
            # zero-case places' lag rows included; only the weights differ.
            # A block over the block limit is cut into views.
            views = self._views(places, tri)
            groups = [
                layout_block(
                    _sub(tri, places, view),
                    self.layout,
                    view,
                    start,
                    self._period(_sub(tri, places, view)),
                )
                for view, _ in views
            ]
            whole = len(views) == 1
            sample = self.sample
            if sample is None:
                past = (
                    self._past_blocks(rows, places, tri, start, horizon)
                    if self.coupling == "online" and whole
                    else []
                )
                if self.collect is not None:
                    self.collect.extend(
                        (g, horizon) for g in [*groups, *(p[0] for p in past)]
                    )
                    raise _CollectedError
                sample = partial(
                    chronos_sampler,
                    checkpoint=self.checkpoint,
                    known=self.known,
                    views=None
                    if whole
                    else (views, places, period is not None),
                    coupling=self._coupling_fit(
                        past,
                        horizon,
                        seed,
                        self._history(tri, start),
                        task.n_samples,
                        as_date(str(task.vintage)[:10]).toordinal(),
                    ),
                )
            parts = sample(groups, horizon, task.n_samples, seed)
            # Chronos-2's sampler puts the views back together itself, so
            # it returns one part; any sampler that returns one part per
            # view has them put together here.
            draws = (
                parts[0]
                if len(parts) == 1
                else _owned(parts, views, places, period is not None)
            )
            # Counts are not negative; Chronos-2's output is otherwise kept
            # as it is, not rounded.
            drawn = np.maximum(draws, 0.0)
        if period is not None:
            values = _weekly_from_totals(drawn, period, tri, start, weeks)
        else:
            final = tri.final
            steps = np.broadcast_to(
                final, (task.n_samples, *final.shape)
            ).copy()
            steps[:, :, start:] = drawn[:, : len(places)]
            week_of = np.array(
                [_step("weekly", t).isoformat() for t in tri.times]
            )
            values = np.stack(
                [steps[:, :, week_of == w].sum(axis=2) for w in weeks],
                axis=2,
            )
        n, n_loc = task.n_samples, len(places)
        weekly_out = pd.DataFrame(
            {
                "location": np.repeat(list(places), n * len(weeks)),
                "target_end_date": np.tile(weeks, n * n_loc),
                "sample": np.tile(
                    np.repeat(np.arange(1, n + 1), len(weeks)), n_loc
                ),
                "value": values.transpose(1, 0, 2).reshape(-1),
            }
        )[list(SAMPLE_COLUMNS)]
        if self.source != "daily":
            return weekly_out, pd.DataFrame(columns=list(DAILY_COLUMNS))
        return weekly_out, _daily_frame(drawn, tri, start, places, weeks)

    def _rows(
        self, task: ForecastTask
    ) -> tuple[pd.DataFrame, tuple[str, ...], list[str]]:
        """Return the task's releases of the source, its places and weeks.

        Raises `ModelUnavailableError` with no releases of the source.
        """
        weekly = self.source == "weekly"
        base = task.target_history if weekly else task.daily_history
        zone = task.zone_history if weekly else task.zone_daily_history
        if base is None or base.empty:
            raise ModelUnavailableError(
                f"{self.name} needs the {self.source} releases, and the task "
                f"for {task.vintage} has none."
            )
        # Only the releases of the country and provinces, so a release
        # gives every place's lag cells or none.
        if zone is not None:
            zone = zone.loc[zone["vintage"].isin(set(base["vintage"]))]
        has_zones = zone is not None and task.zone_daily_history is not None
        zones = tuple(
            z
            for z in task.zone_codes
            if has_zones and province_of(z) in task.locations
        )
        rows = _rows(base, zone if zones else None)
        assert rows is not None
        places = (NATIONAL, *task.locations, *zones)
        weeks = list(task.target_end_dates)
        return rows, places, weeks

    def _triangle(
        self, rows: pd.DataFrame, places: Sequence[str], last_time: str
    ) -> HubTriangle:
        """Return the task's triangle, `lags` or `lag_count` lags wide."""
        full = hub_triangle(rows, self.source, places, last_time, all_lags=True)
        # A fixed count is still no more than the history allows.
        lags = (
            min(self.lags, max(1, full.last - 1))
            if self.lags
            else lag_count(full, self.source)
        )
        return replace(full, partial=full.partial[:, :, :lags])

    def _views(
        self, places: Sequence[str], tri: HubTriangle
    ) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
        """Return the block's views (`active_views` or `block_views`).

        With an active-zone rule, `active_views`, else `block_views` under
        `block_limit`. A place's series are its count row, its aggregate
        row (`period`) and, for `ragged`, its lag series.
        """
        per = 1 + (self.period is not None)
        if self.layout == "ragged":
            per += int(tri.partial.shape[2])
        cap = max(2, self.block_limit // per) if self.block_limit else None
        if self.active_zone_cases is not None:
            active = active_places(
                places,
                tri,
                self.active_zone_cases,
                self.active_zone_weeks,
                self.active_zone_recent_weeks,
                self.source,
            )
            return active_views(places, active, cap)
        if cap is None:
            return [(tuple(places), tuple(places))]
        return block_views(places, cap)

    def lag_count(self, task: ForecastTask) -> int:
        """Return the lag count the model gives `task`'s block."""
        if (run := self.for_task(task)) is not self:
            return run.lag_count(task)
        rows, places, weeks = self._rows(task)
        return int(self._triangle(rows, places, weeks[-1]).partial.shape[2])

    def block_sizes(self, task: ForecastTask) -> pd.DataFrame:
        """Return the size of each view the model predicts for `task`.

        Parameters
        ----------
        task : ForecastTask
            The task.

        Returns
        -------
        pandas.DataFrame
            One row per view: `view` (its index), `part` (`joint` for the
            views forecasting the country or a province, else `zones`),
            `places`,
            `forecasts` (the places it forecasts), `target_rows` (count and
            aggregate rows), `lag_series` (for `ragged`) and `series`.
        """
        if (run := self.for_task(task)) is not self:
            return run.block_sizes(task)
        rows, places, weeks = self._rows(task)
        tri = self._triangle(rows, places, weeks[-1])
        lags = int(tri.partial.shape[2]) if self.layout == "ragged" else 0
        per = 1 + (self.period is not None)
        joint = {p for p in places if p.count("-") < 2}
        out = []
        for i, (view, mine) in enumerate(self._views(places, tri)):
            target_rows = per * len(view)
            out.append(
                {
                    "view": i,
                    "part": "joint" if set(mine) & joint else "zones",
                    "places": len(view),
                    "forecasts": len(mine),
                    "target_rows": target_rows,
                    "lag_series": lags * len(view),
                    "series": target_rows + lags * len(view),
                }
            )
        return pd.DataFrame(out)

    def _period(self, tri: HubTriangle) -> PeriodRows | None:
        """Return the block's aggregate rows on `tri`, or None."""
        return rolling_rows(tri, 7) if self.period == "rolling7" else None

    def _past_blocks(
        self,
        rows: pd.DataFrame,
        places: Sequence[str],
        tri: HubTriangle,
        start: int,
        horizon: int,
    ) -> list[tuple[Series, np.ndarray, int, int]]:
        """Return the block at each earlier release, with what matured.

        Each block (at `coupling_releases`) is built as the task's is,
        from releases up to its own. It comes with its targets' values at
        the task's release, NaN where not yet final (`n_context`), its
        first target step's index in the task's steps, and its release's
        day (an ordinal date).
        """
        offset = tri.last - start
        lags = int(tri.partial.shape[2])
        now = self._values(tri)
        index = {t: i for i, t in enumerate(tri.times)}
        width = 7 if self.source == "weekly" else 1
        ahead = start + horizon - 1 - tri.last
        out = []
        for v in coupling_releases(rows["vintage"]):
            sub = rows.loc[rows["vintage"].astype(str).str[:10] <= v]
            cut = _step(self.source, str(sub["cutoff"].max()))
            end = cut + dt.timedelta(days=width * max(ahead, 0))
            # A release before any of the places reported has no block.
            if not sub["location"].isin(places).any():
                continue
            earlier = hub_triangle(
                sub, self.source, places, end.isoformat(), lags
            )
            first = earlier.last - offset
            if first < 2 or first + horizon != len(earlier.times):
                continue
            at = index.get(earlier.times[first])
            if at is None:
                continue
            steps = at + np.arange(horizon)
            ok = steps < tri.n_context
            if not ok.any():
                continue
            seen = np.full((now.shape[0], horizon), np.nan)
            seen[:, ok] = now[:, steps[ok]]
            block = layout_block(
                earlier, self.layout, places, first, self._period(earlier)
            )
            out.append((block, seen, at, as_date(v).toordinal()))
        return out

    def _values(self, tri: HubTriangle) -> np.ndarray:
        """Return the block's target rows' values at the task's release.

        The count rows (`final`) and then the aggregate rows, `[rows,
        steps]`; steps from `n_context` on are not final.
        """
        period = self._period(tri)
        if period is None:
            return tri.final
        return np.concatenate([tri.final, period.totals])

    def _history(self, tri: HubTriangle, start: int) -> np.ndarray:
        """Return the target rows' final values before the first step drawn.

        The Schaake shuffle's history (`choose_coupling`).
        """
        return self._values(tri)[:, : min(start, tri.n_context)]

    def _coupling_fit(
        self,
        past: Sequence[tuple[Series, np.ndarray, int, int]],
        horizon: int,
        seed: int,
        history: np.ndarray,
        n_samples: int,
        day: int,
    ) -> dict[str, Any]:
        """Return the coupling's fit for one task, and keep its record.

        For `online`, `choose_coupling` picks and fits the coupling from
        Chronos-2's quantiles of the earlier blocks (`past`), what matured
        of them, and the target rows' `history`, with draws seeded from the
        task's own seed. The record, without the Schaake windows, is kept
        as `copula_fit`.
        """
        if self.coupling != "online":
            fit = fit_coupling(np.empty((0, 0, 0)), self.coupling)
            self.copula_fit = fit
            return fit
        pipeline = _chronos(self.checkpoint)
        levels = np.asarray(pipeline.quantiles, dtype=float)
        native = _native(pipeline, [p[0] for p in past], horizon, self.known)
        rows = history.shape[0]
        quantiles = np.zeros((len(past), rows, horizon, len(levels)))
        for k, q in enumerate(native):
            quantiles[k] = q[:rows]
        observed = np.full((len(past), rows, horizon), np.nan)
        for k, (_, seen, _, _) in enumerate(past):
            observed[k] = seen
        starts = np.array([p[2] for p in past], dtype=int)
        fit = choose_coupling(
            quantiles,
            levels,
            observed,
            np.array([p[3] for p in past], dtype=int),
            day,
            history,
            starts,
            n_samples=n_samples,
            seed=seed,
        )
        self.copula_fit = {k: v for k, v in fit.items() if k != "windows"}
        return fit

    def forecast(self, task: ForecastTask) -> pd.DataFrame:
        """Return trajectories of every hub target in the task.

        The weekly part of `forecast_both`, which says what is raised.
        """
        return self.forecast_both(task)[0]

    def quantiles(
        self, task: ForecastTask, levels: Sequence[float]
    ) -> pd.DataFrame:
        """Return Chronos-2's own quantiles where each target is one step.

        Each week of the weekly source is one of Chronos-2's steps, so its
        quantiles are Chronos-2's own at `levels`, which its pipeline
        interpolates between its training levels (`native_sampler`).
        They pass through `forecast` as one trajectory per level; the only
        step after the draws, the floor at zero, is monotone, so
        trajectory `k` is the quantile at level `k`.
        A week of the daily source read from each place's aggregate row is
        one step of that row, so it has a native quantile too.

        Parameters
        ----------
        task : ForecastTask
            The data and weeks to forecast.
        levels : sequence of float
            Increasing quantile levels.

        Returns
        -------
        pandas.DataFrame
            Columns `location`, `target_end_date`, `quantile` and `value`.
        """
        at = np.asarray(levels, dtype=float)
        run = replace(
            self, sample=native_sampler(at, self.checkpoint, self.known)
        )
        out = run.forecast(replace(task, n_samples=len(at)))
        out["quantile"] = at[out["sample"].to_numpy() - 1]
        return out[["location", "target_end_date", "quantile", "value"]]

    def daily_quantiles(
        self, task: ForecastTask, levels: Sequence[float]
    ) -> pd.DataFrame | None:
        """Return Chronos-2's own quantiles of each day's count.

        The daily source's count rows: each day is one step, so it has a
        native quantile, from the same prediction (`known`) as the 7-day
        sums (the aggregate rows where the block has them, else the summed
        draws).

        Parameters
        ----------
        task : ForecastTask
            The data and weeks to forecast.
        levels : sequence of float
            Increasing quantile levels.

        Returns
        -------
        pandas.DataFrame or None
            Columns `location`, `date`, `quantile` and `value`, or None
            for the weekly source, which has no daily counts.
        """
        if self.for_task(task).source != "daily":
            return None
        at = np.asarray(levels, dtype=float)
        run = replace(
            self, sample=native_sampler(at, self.checkpoint, self.known)
        )
        _, out = run.forecast_both(replace(task, n_samples=len(at)))
        out["quantile"] = at[out["sample"].to_numpy() - 1]
        return out[["location", "date", "quantile", "value"]]


def native_sampler(
    levels: np.ndarray,
    checkpoint: Checkpoint | None = None,
    known: Known | None = None,
) -> Sampler:
    """Return a sampler giving Chronos-2's quantiles at `levels` as paths.

    Parameters
    ----------
    levels : numpy.ndarray
        Increasing quantile levels, `[K]`.
    checkpoint : Checkpoint, optional
        The fine-tuned checkpoint, the pretrained model by default.
    known : mapping, optional
        Quantiles predicted ahead (`prefetch`).

    Returns
    -------
    Sampler
        Gives `[K, V, horizon]` per group: path `k` is each series and
        step's quantile at `levels[k]`, as Chronos-2's pipeline gives it
        (interpolated between its own levels). It draws no random numbers.
    """

    def sample(
        groups: Sequence[Series],
        horizon: int,
        n: int,  # noqa: ARG001
        seed: int,  # noqa: ARG001
    ) -> list[np.ndarray]:
        pipeline = _chronos(checkpoint)
        out = []
        for q in _native(pipeline, groups, horizon, known, levels):
            q = np.sort(q, axis=-1)  # [V, H, K], monotone in the level
            out.append(np.moveaxis(q, -1, 0))  # [K, V, H]
        return out

    return sample


class _CollectedError(Exception):
    """Raised by `prefetch`'s probe once a task's groups are collected."""


def prefetch(
    model: FoundationModel,
    tasks: Sequence[ForecastTask],
    levels: Sequence[float] | None = None,
) -> FoundationModel:
    """Return the model with Chronos-2's quantiles of `tasks` predicted.

    Each task's groups are built as `forecast_both` builds them, then all
    groups of one horizon are predicted together, in passes of up to
    `chronos_batch()` rows (`_predict`), instead of one pass a task. The
    model's draws and own quantiles of those tasks then read them, so they
    are what forecasting each task alone gives. A model with its own
    sampler, or a task it cannot forecast, is left as it is.

    Parameters
    ----------
    model : FoundationModel
        The model.
    tasks : sequence of ForecastTask
        The tasks it will forecast, with their `n_samples` and seeds.
    levels : sequence of float, optional
        Quantile levels its own quantiles will be read at
        (`FoundationModel.quantiles`); Chronos-2's own, for draws, by
        default.

    Returns
    -------
    FoundationModel
        The model with `known` holding the tasks' quantiles.
    """
    if model.sample is not None:
        return model
    seen: list[tuple[Series, int]] = []
    # Each task's block and, for draws with the gaussian coupling, its
    # earlier releases' blocks; quantiles at `levels` have no coupling.
    probe = replace(
        model,
        collect=seen,
        coupling=model.coupling if levels is None else "independent",
    )
    for task in tasks:
        try:
            probe.forecast_both(task)
        except (_CollectedError, ModelUnavailableError):
            continue
    known = dict(model.known or {})
    pipeline = _chronos(model.checkpoint)
    for horizon in sorted({h for _, h in seen}):
        groups = [g for g, h in seen if h == horizon]
        quantiles = _native(pipeline, groups, horizon, known, levels)
        for g, q in zip(groups, quantiles, strict=True):
            known[group_key(g, horizon, levels)] = q
    return replace(model, known=known, collect=None)


# One model per method the paper compares, each reading its cadence from
# the task (`FROM_DATA`), so one run gives the daily counts and the 7-day
# hub target. `foundation` is the hub's model.
register_model(
    "foundation_zeroshot",
    partial(FoundationModel, "foundation_zeroshot", FROM_DATA, "ragged"),
    "Chronos-2 zero-shot on the hub target's releases, ragged layout, every "
    "target in one block.",
    fitted=True,
)
register_model(
    "foundation",
    partial(_finetuned, "foundation", "ragged"),
    "Chronos-2 fine-tuned on simulated outbreaks, on the hub target's "
    "releases, ragged layout.",
    fitted=True,
)
register_model(
    "foundation_zeroshot_complete",
    partial(
        FoundationModel, "foundation_zeroshot_complete", FROM_DATA, "complete"
    ),
    "Chronos-2 zero-shot on the hub target's releases, complete layout.",
    fitted=True,
)
# For checkpoints trained on complete counts (the no-delays run).
register_model(
    "foundation_complete",
    partial(_finetuned, "foundation_complete", "complete"),
    "Chronos-2 fine-tuned on complete counts, on the hub target's "
    "releases, complete layout.",
    fitted=True,
)
# The two-step sensitivity analysis (#538): baselinenowcast's point
# nowcast, then Chronos-2's complete layout, zero-shot and fine-tuned.
register_model(
    "foundation_zeroshot_twostep",
    partial(
        FoundationModel,
        "foundation_zeroshot_twostep",
        FROM_DATA,
        "complete",
        two_step=True,
    ),
    "baselinenowcast's point nowcast, then Chronos-2 on the corrected "
    "daily counts, complete layout.",
    fitted=True,
)
register_model(
    "foundation_twostep",
    partial(_finetuned, "foundation_twostep", "complete", two_step=True),
    "baselinenowcast's point nowcast, then the fine-tuned Chronos-2 on the "
    "corrected daily counts, complete layout.",
    fitted=True,
)


def realdata_pinned(path: Path | str = FINETUNED_CONFIG) -> bool:
    """Whether `path` pins the real-data checkpoint.

    Parameters
    ----------
    path : pathlib.Path or str
        A TOML file with `realdata`.

    Returns
    -------
    bool
    """
    return bool(tomllib.loads(Path(path).read_text()).get("realdata"))


def register_realdata_models() -> None:
    """Register `foundation_real`, fine-tuned on real data alone.

    The module registers them only once `realdata_pinned`, so while the
    pin is pending no store plan or backtest resolves their checkpoint.
    """
    register_model(
        "foundation_real",
        partial(_real_data, "foundation_real"),
        "Chronos-2 fine-tuned on real data (German and BDBV), on the hub "
        "target's releases, ragged layout (SI only).",
        fitted=True,
        replace=True,
    )


if Path(FINETUNED_CONFIG).exists() and realdata_pinned():
    register_realdata_models()
