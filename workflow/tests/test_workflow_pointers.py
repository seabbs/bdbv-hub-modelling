"""Each key endpoint's pointer release is promoted by its own target.

The main run's checkpoint (`chronos-main`, target `publish`) and each
stored paper set (`paper-<set>-main`, target `paper_inputs`) are named
by a promote rule that waits for the version's own release.
"""

import re

import pytest
from test_workflow_dryrun import FT, _checkpoint, dry, dry_run

pytest.importorskip("snakemake")


def _edges(tmp_path, target, config=None):
    """Return a dry run's rule-to-rule edges, by rule name."""
    out = dry_run(tmp_path, target, config=config, extra=["--rulegraph"])
    assert out.returncode == 0, out.stderr + out.stdout
    names = dict(re.findall(r'^\s*(\d+)\[label = "(\w+)"', out.stdout, re.M))
    pairs = re.findall(r"^\s*(\d+) -> (\d+)", out.stdout, re.M)
    return names, {(names[a], names[b]) for a, b in pairs}


def test_publish_promotes_the_main_run_s_checkpoint(tmp_path):
    _checkpoint(tmp_path, 250)
    jobs = dry(tmp_path, "publish", config={**FT, "live_run": "runa"})
    assert jobs["promote_main"] == 1
    _, edges = _edges(tmp_path, "publish", {**FT, "live_run": "runa"})
    assert ("publish_checkpoint", "promote_main") in edges


def test_an_ablation_run_never_promotes(tmp_path):
    _checkpoint(tmp_path, 250)
    jobs = dry(tmp_path, "publish", config={**FT, "live_run": "other"})
    assert "promote_main" not in jobs


def test_paper_inputs_promotes_each_set_it_releases(tmp_path):
    names, edges = _edges(tmp_path, "paper_inputs")
    released = {n.removeprefix("release_paper_") for n in names.values()
                if n.startswith("release_paper_")}  # fmt: skip
    assert released
    for name in released:
        assert (f"release_paper_{name}", f"promote_paper_{name}") in edges
        assert (f"promote_paper_{name}", "paper_inputs") in edges
