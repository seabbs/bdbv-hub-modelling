"""The Pages workflow deploys from main only and can read a scratch store."""

from __future__ import annotations

from pathlib import Path

import yaml

PAGES = Path(__file__).parents[1] / ".github" / "workflows" / "pages.yml"


def _workflow() -> dict:
    return yaml.safe_load(PAGES.read_text())


def test_only_main_deploys_the_site() -> None:
    condition = _workflow()["jobs"]["deploy"]["if"]
    assert "github.ref == 'refs/heads/main'" in condition


def test_a_dispatch_can_name_a_scratch_store() -> None:
    wf = _workflow()
    # PyYAML reads the key `on` as True.
    inputs = wf[True]["workflow_dispatch"]["inputs"]
    assert "store" in inputs
    prep = wf["jobs"]["prep"]
    # The job that fetches the store, whose site results Site unpacks.
    assert any("artifact.py" in s.get("run", "") for s in prep["steps"])
    assert prep["env"]["FIT_STORE"] == "${{ inputs.store || 'fit-store' }}"


def test_the_getting_started_notebook_reads_the_store_itself() -> None:
    # The site results' fit list can name a fit the fetched store has
    # since pruned, so the notebook chooses its fit from the store.
    notebook = (PAGES.parents[2] / "docs" / "getting-started.py").read_text()
    assert "store.newest_fit(" in notebook
    assert "fit-store.json" not in notebook
    assert ".json.gz" not in notebook
