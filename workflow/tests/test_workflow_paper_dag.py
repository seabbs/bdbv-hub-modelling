"""The paper is a target of the one workflow, and its jobs all resolve."""

import re

import pytest
from test_workflow_dryrun import ROOT, dry_run

pytest.importorskip("snakemake")

PATH = re.compile(r"(?<![\w.:/])(/[\w.+-]+(?:/[\w.+@=-]+)+)")


def _job_lines(out):
    """The dry run's input, output, log and shell lines."""
    return [
        line
        for line in out.splitlines()
        if re.match(r"^\s+(input|output|log):", line)
        or line.startswith("Shell command:")
    ]


@pytest.mark.parametrize("target", ["paper", "development"])
def test_the_paper_targets_resolve_in_the_one_workflow(tmp_path, target):
    out = dry_run(tmp_path, target, quiet=False, extra=["-p"])
    assert out.returncode == 0, out.stderr + out.stdout
    for rule in (
        target,
        "publish_paper",
        "summary",
        "evidence",
        "run_progress",
        "horizon_table",
        "progress_finetuned",
    ):
        assert f"rule {rule}:" in out.stdout
    lines = _job_lines(out.stdout)
    assert lines
    for line in lines:
        # Every path is in the checkout or under the machine's ROOT.
        assert "../" not in line, line
        for path in PATH.findall(line):
            allowed = (str(ROOT), str(tmp_path), "/dev/null")
            assert path.startswith(allowed), (path, line)


@pytest.mark.parametrize("target", ["paper", "development"])
def test_the_paper_s_realism_reads_the_grid_s_rounds(tmp_path, target):
    out = dry_run(tmp_path, target, quiet=False, extra=["-p"])
    assert out.returncode == 0, out.stderr + out.stdout
    for rule in ("sim_realism", "round_summaries", "germany_dev_stats"):
        assert f"rule {rule}:" in out.stdout


def test_the_paper_s_disease_table_is_a_workflow_output(tmp_path):
    out = dry_run(tmp_path, "development", quiet=False, extra=["-p"])
    assert out.returncode == 0, out.stderr + out.stdout
    assert "rule sim_diseases:" in out.stdout
    assert "--sim-diseases" in out.stdout


def test_the_copula_scores_are_workflow_outputs(tmp_path):
    out = dry_run(tmp_path, "development", quiet=False, extra=["-p"])
    assert out.returncode == 0, out.stderr + out.stdout
    for rule in ("pairing_germany", "pairing_synthetic", "si_copula"):
        assert f"rule {rule}:" in out.stdout
    assert "--si-copula" in out.stdout
    assert "--copula-rho" in out.stdout
