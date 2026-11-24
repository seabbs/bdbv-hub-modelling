"""The dashboard is served with the run, or alone, by one code path."""

import subprocess
import sys
import time
from pathlib import Path

from flowlib.dashboard import serve

ROOT = Path(__file__).parents[2]


def _script(tmp_path: Path) -> Path:
    # A stand-in server: lives until it is killed or its driver ends.
    script = tmp_path / "fake_dashboard.py"
    script.write_text(
        "import os, sys, time\n"
        "pid = int(sys.argv[sys.argv.index('--while-pid') + 1])\n"
        "while True:\n"
        "    try:\n"
        "        os.kill(pid, 0)\n"
        "    except OSError:\n"
        "        sys.exit(0)\n"
        "    time.sleep(0.1)\n"
    )
    return script


def _local(tmp_path: Path) -> dict[str, str]:
    return {"ROOT": str(tmp_path), "DASHBOARD_ROOT": str(tmp_path)}


def test_ensure_starts_one_server_and_reuses_a_live_one(tmp_path):
    driver = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"]
    )
    try:
        local, script = _local(tmp_path), _script(tmp_path)
        first = serve.ensure(local, driver.pid, script=script)
        assert serve.pidfile(local).read_text().strip() == str(first)
        assert serve.ensure(local, driver.pid, script=script) == first
    finally:
        driver.kill()
        driver.wait()
    # The server ends with the process it was started for.
    for _ in range(100):
        if not serve.alive(first):
            break
        time.sleep(0.1)
    assert not serve.alive(first)


def test_a_dead_server_is_replaced(tmp_path):
    local, script = _local(tmp_path), _script(tmp_path)
    serve.pidfile(local).parent.mkdir(parents=True)
    serve.pidfile(local).write_text("999999999\n")
    driver = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"]
    )
    try:
        got = serve.ensure(local, driver.pid, script=script)
        assert got != 999999999
        assert serve.alive(got)
    finally:
        driver.kill()
        driver.wait()


def test_the_workflow_serves_the_dashboard_of_the_runs_it_trains():
    # On by default: a fine-tuning run serves its own dashboard from its
    # onstart, the same entry the task uses, for the runs it trains.
    config = (ROOT / "workflow" / "config.yaml").read_text()
    assert "serve: true" in config
    snakefile = (ROOT / "workflow" / "Snakefile").read_text()
    assert "onstart:" in snakefile
    assert "serve.ensure(" in snakefile
    assert "runs=TRAINING.trained_runs()" in snakefile


def test_the_dashboard_shows_the_runs_it_is_given(tmp_path):
    seen = tmp_path / "argv.txt"
    script = tmp_path / "argv_dashboard.py"
    script.write_text(
        f"import sys\nopen({str(seen)!r}, 'w').write(' '.join(sys.argv[1:]))\n"
    )
    driver = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"]
    )
    try:
        serve.ensure(
            _local(tmp_path), driver.pid, script=script, runs=["runa", "runb"]
        )
        for _ in range(100):
            if seen.exists():
                break
            time.sleep(0.1)
    finally:
        driver.kill()
        driver.wait()
    assert "--run runa --run runb" in seen.read_text()


def test_the_coordinator_names_itself_as_the_driver():
    # Ticks come and go; the dashboard lives as long as the coordinator.
    text = (ROOT / "workflow" / "scripts" / "coordinate.py").read_text()
    assert "WORKFLOW_DRIVER_PID" in text
    assert serve.DRIVER_ENV == "WORKFLOW_DRIVER_PID"


def test_the_server_ends_when_its_driver_has(monkeypatch):
    import pytest

    pytest.importorskip("matplotlib")
    sys.path.insert(0, str(ROOT / "workflow" / "scripts"))
    import dashboard

    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    ended = []
    monkeypatch.setattr(dashboard.os, "_exit", ended.append)
    dashboard.end_with(gone.pid, every=0.01)
    for _ in range(100):
        if ended:
            break
        time.sleep(0.02)
    assert ended == [0]


def test_only_a_fine_tuning_run_that_asks_serves_it(monkeypatch):
    monkeypatch.delenv(serve.JOB_ENV, raising=False)
    on = {"dashboard": {"serve": True}}
    assert serve.wanted(on, ["simulate", "finetune"])
    assert not serve.wanted(on, ["score"])
    assert not serve.wanted({"dashboard": {"serve": False}}, ["finetune"])
    # One job of another machine's run (an ssh worker) serves nothing.
    monkeypatch.setenv(serve.JOB_ENV, "1")
    assert not serve.wanted(on, ["finetune"])


def test_a_worker_job_is_marked_as_one():
    from flowlib import compute

    worker = compute.Worker("w", "host", "/w", "/c")
    got = compute.job_env({"ROOT": "/r"}, worker)
    assert got[serve.JOB_ENV] == "1"


def test_an_empty_or_reused_pid_file_still_starts_a_server(tmp_path):
    local, script = _local(tmp_path), _script(tmp_path)
    serve.pidfile(local).parent.mkdir(parents=True)
    driver = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"]
    )
    try:
        serve.pidfile(local).write_text("")
        got = serve.ensure(local, driver.pid, script=script)
        assert serve.alive(got)
        # A live pid that is not the dashboard (here the driver itself).
        serve.pidfile(local).write_text(f"{driver.pid}\n")
        again = serve.ensure(local, driver.pid, script=script)
        assert again != driver.pid
    finally:
        driver.kill()
        driver.wait()


def test_the_run_never_fails_because_the_dashboard_did():
    text = (ROOT / "workflow" / "Snakefile").read_text()
    hook = text.split("onstart:", 1)[1].split("\nonerror:", 1)[0]
    assert "except Exception" in hook
