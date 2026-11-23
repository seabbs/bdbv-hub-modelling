"""The runs and references a dashboard shows, from configuration.

Nothing about which runs exist is in the code. The runs are the
workflow's (`runs` in workflow/config.yaml, `flowlib.dashboard.context`),
each its training output folder, whose `train/n<N>/settings.json` name
its checkpoints. One run is `live`, the one the status follows.

References (zero-shot ragged, complete-data only, randomwalknowcast)
match score and forecast rows by a model regex and, optionally, a
checkpoint value: `dashboard.references`, then `--ref` specs
`name=MODEL_REGEX[,checkpoint=C][,label=L][,colour=C][,role=R]`, each
replacing a reference of the same name. A spec splits only at a comma
that starts a known field, so a value may hold commas not followed by a
field name (a regex such as `a{1,2}`).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from nowcasteval import labels

PALETTE = labels.RUN_SHADES
"""Colours of runs, in order (`labels.RUN_SHADES`; the context names and
colours every run as the fine-tuned model)."""

REF_FIELDS = {"label", "colour", "checkpoint", "role"}
ROLES = {"baseline", "start", "other"}
"""A reference's role on the score panels: the series scores are relative
to (not drawn), the one each run's curve starts from (dashed), or other
(dotted)."""


@dataclass(frozen=True)
class Run:
    """One training run the dashboard shows."""

    key: str
    label: str
    colour: str = ""
    folder: Path | None = None
    live: bool = False


@dataclass(frozen=True)
class Reference:
    """A reference series, matched on rows by model and checkpoint."""

    key: str
    label: str
    colour: str
    model: str
    checkpoint: str | None = None
    role: str = "other"
    dash: str = ""
    """Line style from the shared map (`solid`, `dashed`, `dotted`)."""

    def matches(self, row: Mapping[str, str]) -> bool:
        """Whether a score or forecast row is this reference's."""
        if not re.search(self.model, row.get("model", "")):
            return False
        return (
            self.checkpoint is None or row.get("checkpoint") == self.checkpoint
        )


def _split(spec: str, fields: set[str]) -> tuple[str, str, dict[str, str]]:
    """Split a spec at commas that start a known field.

    So a value may hold commas (a regex such as `a{1,2}`) as long as they
    are not followed by a field name; `,word=` with an unknown word is an
    error.
    """
    known = "|".join(sorted(fields))
    head, *rest = re.split(rf",(?=(?:{known})(?:=|,|$))", spec)
    if "=" not in head:
        raise ValueError(f"a spec is name=VALUE[,key=value...], not {spec!r}")
    name, value = (x.strip() for x in head.split("=", 1))
    stray = re.search(r",([A-Za-z_]+)=", value)
    if stray:
        raise ValueError(f"unknown field {stray.group(1)!r} in {spec!r}")
    opts = {}
    for item in rest:
        k, sep, v = item.partition("=")
        opts[k.strip()] = v.strip() if sep else "1"
    return name, value, opts


def make_run(
    key: str,
    folder: Path | str | None,
    *,
    label: str | None = None,
    colour: str = "",
    live: bool = False,
) -> Run:
    """Return a run of a training output folder."""
    return Run(
        key=key,
        label=label or key,
        colour=colour,
        folder=None if folder is None else Path(folder).expanduser(),
        live=live,
    )


def load_runs(runs: Iterable[tuple[str, Path | str]]) -> list[Run]:
    """Return a run of each (name, folder), the first live."""
    return [
        make_run(name, folder, live=i == 0, colour=PALETTE[i % len(PALETTE)])
        for i, (name, folder) in enumerate(runs)
    ]


def _role(role: str) -> str:
    if role not in ROLES:
        raise ValueError(f"role {role!r} is not one of {sorted(ROLES)}")
    return role


def make_reference(
    key: str,
    model: str,
    *,
    label: str | None = None,
    colour: str | None = None,
    checkpoint: str | None = None,
    role: str | None = None,
    dash: str | None = None,
) -> Reference:
    """Return a reference matched on rows by `model` and `checkpoint`."""
    return Reference(
        key=key,
        label=label or key,
        colour=colour or labels.NEUTRAL,
        model=model,
        checkpoint=checkpoint,
        role=_role(role or "other"),
        dash=dash or "",
    )


def parse_reference(spec: str) -> Reference:
    """Return the reference of a `name=MODEL_REGEX[,...]` spec."""
    name, model, o = _split(spec, REF_FIELDS)
    return make_reference(
        name,
        model,
        label=o.get("label"),
        colour=o.get("colour"),
        checkpoint=o.get("checkpoint"),
        role=o.get("role"),
    )


def load_references(
    config: Iterable[Mapping], cli: Sequence[str]
) -> list[Reference]:
    """Return the references of config then `--ref`, later ones replacing."""
    out: dict[str, Reference] = {}
    for e in config:
        out[e["key"]] = make_reference(
            e["key"],
            e["model"],
            label=e.get("label"),
            colour=e.get("colour"),
            checkpoint=e.get("checkpoint"),
            role=e.get("role"),
            dash=e.get("dash"),
        )
    for s in cli:
        r = parse_reference(s)
        out[r.key] = r
    return list(out.values())


HASH = re.compile(r"[0-9a-f]{6,}")


def split_label(name: str) -> tuple[str, str]:
    """Return a checkpoint name's hash and the prefix before it.

    `chronos-slots-abc123`, `slots-abc123` and `abc123` give
    (`abc123`, `chronos-slots-`), (`abc123`, `slots-`) and (`abc123`, ``).
    """
    m = HASH.search(name)
    if not m or m.end() != len(name):
        return name, ""
    return m.group(0), name[: m.start()]


INTERIM = "interim"
"""A run's folder of checkpoints exported mid-segment (`s<step>`)."""


def checkpoint_settings(folder: Path) -> list[Path]:
    """Return a run's saved checkpoints' settings, interim ones too."""
    train = Path(folder) / "train"
    return [
        *sorted(train.glob("n*/settings.json")),
        *sorted(train.glob(f"{INTERIM}/s*/settings.json")),
    ]


def folder_hashes(folder: Path | None) -> set[str]:
    """Return the checkpoint hashes a training output folder has saved.

    An interim checkpoint's hash (`interim-s<step>-<hash>`) counts by its
    own hash (`split_label`).
    """
    if folder is None:
        return set()
    out = set()
    for s in checkpoint_settings(Path(folder)):
        try:
            key = json.loads(s.read_text()).get("hash")
        except (OSError, ValueError):
            continue
        if key:
            out.add(split_label(str(key))[0])
    return out


class Classifier:
    """Which run or reference a checkpoint or row belongs to."""

    def __init__(
        self,
        runs: Sequence[Run],
        references: Sequence[Reference] = (),
        excluded: Iterable[str] = (),
    ) -> None:
        self.runs = list(runs)
        self.references = list(references)
        self.excluded = set(excluded)
        self._saved = {r.key: folder_hashes(r.folder) for r in self.runs}

    def run_of(self, name: str) -> str | None:
        """Return the run of a checkpoint name, or None if none claims it."""
        key, _ = split_label(name)
        if key in self.excluded:
            return None
        for r in self.runs:
            if key in self._saved[r.key]:
                return r.key
        return None

    def series(self, row: Mapping[str, str]) -> str | None:
        """Return a score or forecast row's series: a reference, else a run."""
        for ref in self.references:
            if ref.matches(row):
                return ref.key
        checkpoint = row.get("checkpoint", "")
        if not checkpoint or checkpoint == "zero-shot":
            return None
        return self.run_of(checkpoint)
