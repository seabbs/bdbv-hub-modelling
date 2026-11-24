# Copyright (c) 2026 The bdbv-hub-modelling authors. MIT licence, see LICENSE.
"""Publish a fine-tuned checkpoint to the Hugging Face Hub.

    publish-hf CHECKPOINT OUT [--repo OWNER/NAME] [--scores CSV] [--dry-run]

CHECKPOINT is a verified checkpoint folder, a run's `n<N>/` or
`interim/s<step>/`: its `settings.json`, its adapter in
`finetuned-ckpt/` and, when there, its held-out losses
(`sidecar.SIDECAR`) and 7-day check (`seven-day.json`). `build_folder`
writes what the Hub repository holds, the files a release holds
(`forecaster.ADAPTER_FILES`, `settings.json`, `forecaster.SUMS_FILE`) and
the model card (`modelcard.model_card`), so `forecaster.hub_checkpoint`
reads it as its release. `upload` uploads that folder and tags the commit
with the checkpoint's spec (`slots-<hash>`), so each published version is
a tagged revision. With `--dry-run`, or no `--repo`, only the folder is
written. The token is read from `HF_TOKEN_ENV` and never written out.
OUT records what was done.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import pandas as pd

from .forecaster import ADAPTER_FILES, SUMS_FILE, checkpoint_spec
from .modelcard import CARD_FILE, CardEvaluation, citation_bibtex, model_card
from .sidecar import SIDECAR

__all__ = [
    "HF_TOKEN_ENV",
    "SEVEN_DAY_FILE",
    "build_folder",
    "main",
    "upload",
]

HF_TOKEN_ENV: Final = "HF_TOKEN"
"""The environment variable holding the Hub token `upload` uses."""

SEVEN_DAY_FILE: Final = "seven-day.json"
"""A checkpoint's 7-day check, beside its settings (rules/evaluate.smk)."""

CITATION_FILE: Final = Path("CITATION.cff")
"""The repository's citation, which the card cites."""


def _json(path: Path) -> dict[str, Any]:
    """Return a JSON file's object, or an empty one if it is not there."""
    return json.loads(path.read_text()) if path.is_file() else {}


def build_folder(
    checkpoint: Path | str,
    out: Path | str,
    scores: pd.DataFrame | None = None,
    citation: str = "",
) -> Path:
    """Write the Hub repository's files of a checkpoint into `out`.

    Parameters
    ----------
    checkpoint : pathlib.Path or str
        The checkpoint folder: `settings.json` and `finetuned-ckpt/`.
    out : pathlib.Path or str
        The folder to write; replaced if it exists.
    scores : pandas.DataFrame, optional
        Scores relative to reference models for the card
        (`modelcard.SCORE_COLUMNS`).
    citation : str
        BibTeX for the card (`modelcard.citation_bibtex`).

    Returns
    -------
    pathlib.Path
        `out`.
    """
    checkpoint, out = Path(checkpoint), Path(out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    adapter = checkpoint / "finetuned-ckpt"
    for name in ADAPTER_FILES:
        shutil.copy2(adapter / name, out / name)
    shutil.copy2(checkpoint / "settings.json", out / "settings.json")
    sums = tuple(
        (name, hashlib.sha256((out / name).read_bytes()).hexdigest())
        for name in (*ADAPTER_FILES, "settings.json")
    )
    (out / SUMS_FILE).write_text(
        "".join(f"{sha}  {name}\n" for name, sha in sums)
    )
    settings = json.loads((out / "settings.json").read_text())
    evaluation = CardEvaluation(
        _json(checkpoint / SIDECAR),
        _json(checkpoint / SEVEN_DAY_FILE),
        scores,
    )
    card = model_card(
        settings,
        tuple(s for s in sums if s[0] in ADAPTER_FILES),
        evaluation,
        citation,
    )
    (out / CARD_FILE).write_text(card)
    return out


def upload(folder: Path | str, repo: str, tag: str, token: str) -> str:
    """Upload a built folder to a Hub repository as the revision `tag`.

    The repository is made if it is missing. The commit is tagged `tag`;
    an existing tag is an error, so a published revision never moves.

    Parameters
    ----------
    folder : pathlib.Path or str
        The folder `build_folder` wrote.
    repo : str
        The repository id, `<owner>/<name>`.
    tag : str
        The revision tag, the checkpoint's spec.
    token : str
        A Hub token with write access.

    Returns
    -------
    str
        The commit's hash.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo, repo_type="model", exist_ok=True)
    commit = api.upload_folder(
        folder_path=str(folder),
        repo_id=repo,
        repo_type="model",
        commit_message=f"Checkpoint {tag}",
    )
    api.create_tag(repo, tag=tag, revision=commit.oid, repo_type="model")
    return str(commit.oid)


def main(argv: Sequence[str] | None = None) -> int:
    """Build a checkpoint's Hub folder and, unless a dry run, upload it."""
    parser = argparse.ArgumentParser(prog="publish-hf")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("out", type=Path, help="JSON record of the run")
    parser.add_argument(
        "--folder",
        type=Path,
        default=None,
        help="Where the Hub folder is written; CHECKPOINT/huggingface by "
        "default.",
    )
    parser.add_argument("--repo", default="", help="<owner>/<name>")
    parser.add_argument(
        "--scores",
        type=Path,
        default=None,
        help="CSV of scores relative to reference models for the card.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the folder only; nothing is uploaded.",
    )
    args = parser.parse_args(argv)
    folder = args.folder or args.checkpoint / "huggingface"
    scores = pd.read_csv(args.scores) if args.scores else None
    citation = citation_bibtex(CITATION_FILE) if CITATION_FILE.is_file() else ""
    build_folder(args.checkpoint, folder, scores, citation)
    settings = json.loads((folder / "settings.json").read_text())
    tag = checkpoint_spec(settings)
    record: dict[str, Any] = {"folder": str(folder), "tag": tag}
    if args.dry_run or not args.repo:
        record["dry_run"] = True
    else:
        token = os.environ.get(HF_TOKEN_ENV, "")
        if not token:
            parser.error(f"set {HF_TOKEN_ENV} to a Hub token with write access")
        record |= {
            "dry_run": False,
            "repo": args.repo,
            "revision": upload(folder, args.repo, tag, token),
        }
    args.out.write_text(json.dumps(record) + "\n")
    print(json.dumps(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
