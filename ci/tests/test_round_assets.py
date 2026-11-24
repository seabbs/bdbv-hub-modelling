"""ci/steps/round_assets.sh: a round's run artefacts on its release."""

from __future__ import annotations

import json
import os
import subprocess
import tarfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "steps" / "round_assets.sh"

ARTEFACTS = {
    "inputs": {"inputs/context.json": json.dumps(
        {"as_of": "2026-11-15", "reference_date": "2026-11-21"})},
    "gate": {"gate/provenance.json": "{}",
             "gate/validation/bvd.log": "VALID"},
    "baseline-gate": {"gate/provenance.json": "{}"},
    "candidate-bvd": {"checks.json": "{}", "run.json": "{}",
                      "submission/model-output/seabbs-renewal/x.csv": "a\n"},
    "candidate-main": {"checks.json": "{}"},
    "fit-bvd": {"big.bin": "x"},
    "pr": {"body.md": "x"},
}  # fmt: skip


def _fake_gh(tmp_path: Path, released: bool) -> Path:
    """A `gh` serving ARTEFACTS for one run and logging uploads."""
    store = tmp_path / "artefacts"
    for name, files in ARTEFACTS.items():
        for rel, text in files.items():
            p = store / name / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gh = bin_dir / "gh"
    gh.write_text(
        "#!/usr/bin/env python3\n"
        "import json, shutil, sys\n"
        f"store = {str(store)!r}\n"
        f"log = {str(tmp_path / 'gh.log')!r}\n"
        f"released = {released!r}\n"
        "args = sys.argv[1:]\n"
        "open(log, 'a').write(' '.join(args) + '\\n')\n"
        "if args[:2] == ['api', 'repos/o/r/actions/runs/7/artifacts']:\n"
        "    import os\n"
        "    print('\\n'.join(sorted(os.listdir(store))))\n"
        "elif args[:2] == ['run', 'download']:\n"
        "    name = args[args.index('-n') + 1]\n"
        "    dest = args[args.index('-D') + 1]\n"
        "    shutil.copytree(f'{store}/{name}', dest, dirs_exist_ok=True)\n"
        "elif args[:2] == ['release', 'view']:\n"
        "    sys.exit(0 if released else 1)\n"
        "elif args[:2] == ['release', 'upload']:\n"
        "    pass\n"
    )
    gh.chmod(0o755)
    return bin_dir


def _run(tmp_path: Path, released: bool, dry: str = "false"):  # type: ignore[no-untyped-def]
    env = {
        **os.environ,
        "PATH": f"{_fake_gh(tmp_path, released)}:{os.environ['PATH']}",
        "WORK": str(tmp_path / "work"),
        "CI_DIR": str(SCRIPT.parents[1]),
        "GITHUB_REPOSITORY": "o/r",
        "DRY_RUN": dry,
    }
    return subprocess.run(
        ["bash", str(SCRIPT), "7"], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=60,
    )  # fmt: skip


def test_a_runs_candidates_and_gate_go_on_the_rounds_release(
    tmp_path: Path,
) -> None:
    run = _run(tmp_path, released=True)
    assert run.returncode == 0, run.stderr
    tar = tmp_path / "work" / "round-assets" / "round-2026-11-21-run-7.tar.gz"
    with tarfile.open(tar) as t:
        names = set(t.getnames())
    # Every candidate's files, both gates with their validation logs and
    # the round's context; never the fit directories or the PR body.
    assert "run-7/candidate-bvd/checks.json" in names
    assert "run-7/candidate-main/checks.json" in names
    assert any(n.endswith("seabbs-renewal/x.csv") for n in names)
    assert "run-7/gate/gate/validation/bvd.log" in names
    assert "run-7/baseline-gate/gate/provenance.json" in names
    assert "run-7/inputs/inputs/context.json" in names
    assert not [n for n in names if "fit-bvd" in n or "/pr/" in n]
    log = (tmp_path / "gh.log").read_text()
    assert "release upload round-2026-11-21" in log
    assert "--clobber" in log


def test_a_round_with_no_release_yet_uploads_nothing(tmp_path: Path) -> None:
    run = _run(tmp_path, released=False)
    assert run.returncode == 0, run.stderr
    assert "no release round-2026-11-21" in run.stdout
    assert "release upload" not in (tmp_path / "gh.log").read_text()


def test_a_dry_run_only_says_what_it_would_attach(tmp_path: Path) -> None:
    run = _run(tmp_path, released=True, dry="true")
    assert run.returncode == 0, run.stderr
    assert "would attach" in run.stdout
    assert "release upload" not in (tmp_path / "gh.log").read_text()
