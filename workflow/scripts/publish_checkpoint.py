"""Publish a verified checkpoint with the trainer's own `publish`.

    python workflow/scripts/publish_checkpoint.py CHECKPOINT OUT
    python workflow/scripts/publish_checkpoint.py --held-out CHECKPOINT OUT
    python workflow/scripts/publish_checkpoint.py --main RUN CHECKPOINT OUT \
        BRANCH...

CHECKPOINT is a run's `n<N>/` folder. A checkpoint with the series bias
is released as `chronos-slots-<hash>`, else `chronos-ft-<hash>`, and
one of the real-only run (`data: real`) as `realdata-ft-<hash>`. Safe
to run again: a release whose sums match is left as it is. OUT records
the tag. With `--held-out`, an offloaded checkpoint's held-out losses
(`held-out.json`, rules/evaluate.smk), scored after its release, are
uploaded to it beside its settings, and OUT records the tag. With
`--main`, released checkpoint CHECKPOINT of the main fine-tuning run RUN
(the paper's `live_run`) becomes the one the fine-tuned models load: the
release `chronos-main` names it (`fs.promote_main`), and OUT records it.
Only a commit on one of the BRANCHes promotes (`nowcastutils.pointers`).
CHECKPOINT needs only its `settings.json`, so a checkpoint is promoted
from a folder holding its release's `settings.json` too.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from nowcastfoundation.finetune import realdata
from nowcastfoundation.finetune import stream as fs
from nowcastfoundation.forecaster import (
    checkpoint_spec,
    release_tag,
)
from nowcastfoundation.sidecar import SIDECAR


def held_out(where: Path, out: Path) -> None:
    """Upload released checkpoint `where`'s held-out losses to its release."""
    tag = json.loads((where / "published.json").read_text())["tag"]
    subprocess.run(
        ["gh", "release", "upload", tag, str(where / SIDECAR),
         "--clobber", "-R", fs.REPO],
        check=True,
    )  # fmt: skip
    out.write_text(f"{tag}\n")


def main() -> None:
    """Publish the checkpoint in `sys.argv[1]`, recording it in argv[2]."""
    if sys.argv[1] == "--held-out":
        held_out(Path(sys.argv[2]), Path(sys.argv[3]))
        return
    if sys.argv[1] == "--main":
        record = fs.promote_main(
            Path(sys.argv[3]), sys.argv[2], tuple(sys.argv[5:])
        )
        Path(sys.argv[4]).write_text(json.dumps(record) + "\n")
        return
    where, out = Path(sys.argv[1]), Path(sys.argv[2])
    settings = json.loads((where / "settings.json").read_text())
    key = settings["hash"]
    if realdata.is_real(settings):
        # The real-only run's checkpoint, `foundation_real`'s release.
        tag = realdata.publish(where)
        out.write_text(json.dumps({"tag": tag, "hash": key}) + "\n")
        return
    spec = checkpoint_spec(settings)
    fs.publish(where, spec)
    out.write_text(json.dumps({"tag": release_tag(spec), "hash": key}) + "\n")


if __name__ == "__main__":
    main()
