"""Every Snakemake run reports to the Panoptes monitor, on by default."""

import importlib.util
import re
from pathlib import Path

import pytest
from flowlib import compute, localconfig, monitor, schedule

ROOT = Path(__file__).parents[2]
URL = "http://127.0.0.1:8795"


def test_a_run_reports_to_the_machine_s_monitor():
    got = monitor.args({"MONITOR_URL": URL})
    assert got[:2] == ["--logger", "panoptes"]
    assert got[got.index("--logger-panoptes-address") + 1] == URL


def test_an_empty_address_switches_the_monitor_off():
    assert monitor.args({"MONITOR_URL": ""}) == []


def test_the_monitor_is_on_by_default(tmp_path):
    cfg = localconfig.read(tmp_path / "none.env", {"HOME": str(tmp_path)})
    assert cfg["MONITOR_URL"] == URL
    assert cfg["MONITOR_PORT"] == "8795"


def test_workflow_tasks_and_ticks_report_to_it():
    cfg = {"CORES": "8", "MEM_MB": "32768", "MONITOR_URL": URL}
    assert compute.args(cfg, path="")[-len(monitor.args(cfg)) :] == (
        monitor.args(cfg)
    )
    cfg |= {"SCHEDULE_TARGETS": "simulate"}
    for _, cmd, _ in schedule.plan(cfg):
        assert "panoptes" in cmd


@pytest.mark.parametrize(
    "rel",
    [
        "scripts/paper.sh",
        "tools/run_target.sh",
        "workflow/scripts/paper_run.py",
    ],
)
def test_every_other_snakemake_run_reports_to_it(rel):
    text = (ROOT / rel).read_text()
    runs = [
        line
        for line in text.splitlines()
        if re.search(r"snakemake\b", line)
        and not re.search(r"snakemake\b.*\s-n\b", line)
        and not line.lstrip().startswith("#")
    ]
    assert runs
    assert "monitor" in text


def test_the_logger_plugin_is_in_the_workflow_group():
    assert importlib.util.find_spec("snakemake_logger_plugin_panoptes")


def test_the_custom_monitor_page_is_gone():
    assert not (ROOT / "workflow" / "scripts" / "monitor.py").exists()
