"""The workflow is one rules file per step, each included once."""

import re
from pathlib import Path

ROOT = Path(__file__).parents[2]
RULES = ROOT / "workflow" / "rules"
STEPS = ("simulate", "finetune", "evaluate", "publish", "score", "report")
REPORTS = {
    "round_summaries",
    "grid_figures",
    "germany_triangle",
    "germany_dev_stats",
    "sim_report",
    "sim_report_run",
    "sim_reports",
    "german_report",
    "benchmarks",
}


def _rules(step):
    text = (RULES / f"{step}.smk").read_text()
    return set(re.findall(r"^(?:rule|checkpoint) (\w+):", text, re.M))


def test_each_step_has_its_rules_file_and_the_workflow_includes_it():
    snakefile = (ROOT / "workflow" / "Snakefile").read_text()
    included = re.findall(r'^include: "rules/(\w+)\.smk"', snakefile, re.M)
    assert included == ["common", *STEPS]
    assert sorted(p.stem for p in RULES.glob("*.smk")) == sorted(included)


def test_the_common_file_holds_helpers_and_no_rules():
    assert _rules("common") == set()


def test_reports_are_their_own_step():
    assert _rules("report") == REPORTS
    for step in ("simulate", "score"):
        assert not _rules(step) & REPORTS
