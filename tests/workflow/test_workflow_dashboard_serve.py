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


def test_the_workflow_serves_the_dashboard_when_configured():
    # One switch in the workflow config; the Snakefile's onstart calls the
    # same entry the task uses.
    config = (ROOT / "workflow" / "config.yaml").read_text()
    assert "serve: false" in config
    snakefile = (ROOT / "workflow" / "Snakefile").read_text()
    assert "onstart:" in snakefile
    assert "serve.ensure(" in snakefile


def test_the_coordinator_names_itself_as_the_driver():
    # Ticks come and go; the dashboard lives as long as the coordinator.
    text = (ROOT / "workflow" / "scripts" / "coordinate.py").read_text()
    assert "WORKFLOW_DRIVER_PID" in text
    assert serve.DRIVER_ENV == "WORKFLOW_DRIVER_PID"
