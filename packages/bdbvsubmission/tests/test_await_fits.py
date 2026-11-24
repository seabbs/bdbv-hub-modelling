"""steps/await_fits.sh, with `gh` and the CLI standing in."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "steps" / "await_fits.sh"


def _fake_gh(tmp_path: Path, runs: list[dict[str, object]]) -> Path:
    """A `gh` that dispatches nothing and lists `runs` for `run list`."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (tmp_path / "runs.json").write_text(json.dumps(runs))
    gh = bin_dir / "gh"
    gh.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        f"runs = json.load(open({str(tmp_path / 'runs.json')!r}))\n"
        "args = sys.argv[1:]\n"
        "if args[:2] == ['run', 'list']:\n"
        "    limit = int(args[args.index('-L') + 1]) if '-L' in args else 20\n"
        "    if '-e' in args:\n"
        "        event = args[args.index('-e') + 1]\n"
        "        runs = [r for r in runs if r['event'] == event]\n"
        "    print(json.dumps(runs[:limit]))\n"
        "elif args[:2] == ['run', 'view']:\n"
        "    run = next(r for r in runs if str(r['databaseId']) == args[2])\n"
        "    print(json.dumps(run))\n"
    )
    gh.chmod(0o755)
    return bin_dir


def test_newer_pull_request_runs_do_not_end_the_wait(tmp_path: Path) -> None:
    # The Fit store run this step started is still fitting, but more than
    # ten pull request runs on the branch have finished since: the wait
    # must go on until the fits are stored, not rule them out.
    created = "2999-01-01T00:00:00Z"
    prs = [
        {
            "databaseId": i,
            "status": "completed",
            "createdAt": created,
            "event": "pull_request",
        }
        for i in range(2, 14)
    ]
    ours = {
        "databaseId": 1,
        "status": "in_progress",
        "createdAt": created,
        "event": "workflow_dispatch",
    }
    bin_dir = _fake_gh(tmp_path, [*prs, ours])
    cli = tmp_path / "fake_cli.sh"
    cli.write_text(
        'while [ $# -gt 0 ]; do [ "$1" = --out ] && out=$2; shift; done\n'
        f"n=$(cat {tmp_path}/calls 2>/dev/null || echo 0)\n"
        f"echo $((n + 1)) > {tmp_path}/calls\n"
        # Missing on the first three checks, stored from the fourth.
        'if [ "$n" -lt 3 ]; then\n'
        '  echo \'{"include": [{"model": "renewal"}]}\' > "$out"\n'
        'else echo \'{"include": []}\' > "$out"; fi\n'
    )
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "WORK": str(tmp_path / "work"),
        "CI_DIR": str(SCRIPT.parents[1]),
        "FORECAST_CLI": f"bash {cli}",
        "SKIP_FETCH": "true",
        "GITHUB_REPOSITORY": "o/r",
        "MODELS": "renewal",
        "AS_OF": "2026-11-15",
        "DISPATCH": "true",
        "FIT_REF": "next",
        "WAIT_MINUTES": "60",
        "POLL_SECONDS": "0",
    }
    run = subprocess.run(
        ["bash", str(SCRIPT)], cwd=tmp_path, env=env, capture_output=True,
        text=True, timeout=60,
    )  # fmt: skip
    assert run.returncode == 0, run.stderr
    assert "finished without" not in run.stdout
    assert "stored every candidate's fit" in run.stdout
