"""Tests of the synthetic scenario grid (`papertables.synthetic_grid`)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from papertables import synthetic_grid

PAPER = Path(__file__).parents[1]
CONFIG = yaml.safe_load((PAPER / "workflow" / "config.yaml").read_text())
# Paths in the paper config are from the repository root.
GRID_FILE = PAPER.parent / CONFIG["synthetic"]["grid_file"]
GRID = synthetic_grid.read_grid(GRID_FILE)


def test_the_paper_reads_the_main_workflow_s_grid() -> None:
    main = yaml.safe_load(
        (PAPER.parent / "workflow" / "config.yaml").read_text()
    )
    assert (
        GRID_FILE.resolve()
        == (PAPER.parent / main["simulate"]["grid_file"]).resolve()
    )


def test_factor_shares_are_the_weights_normalised() -> None:
    shares = synthetic_grid.factor_shares(GRID)
    assert set(shares) == set(GRID["factors"])
    for levels in shares.values():
        assert abs(sum(levels.values()) - 1) < 1e-12


RELEASE_FILE = GRID_FILE.parent / "release-layer.yaml"
RELEASE = yaml.safe_load(RELEASE_FILE.read_text())


def _rows(table):
    return table[table["header"] == ""]


def test_the_grid_is_one_table_with_every_setting_once() -> None:
    # The grid was three tables and a list: two by layer, one of
    # distributions and the factors again in the model note.
    table = synthetic_grid.grid_table(GRID, RELEASE)
    shown = [s.strip("`") for s in _rows(table)["setting"]]
    grid_settings = set(GRID["factors"]) | set(GRID["distributions"])
    assert grid_settings <= set(shown)
    assert len([s for s in shown if s in grid_settings]) == len(grid_settings)
    heads = [h for h in table["header"] if h]
    assert heads == [
        *(synthetic_grid.LAYER_NAMES[k] for k in synthetic_grid.LAYERS),
        synthetic_grid.RELEASE_NAME,
    ]
    assert list(synthetic_grid.TABLE_HEADERS) == ["setting", "values", "why"]


def test_each_distribution_gives_its_reason() -> None:
    rows = _rows(synthetic_grid.grid_table(GRID, RELEASE)).set_index("setting")
    gt = rows.loc["`gt_mean`"]
    assert gt["values"].startswith("LogNormal(median 7")
    assert gt["why"] == GRID["distributions"]["gt_mean"]["why"]


def test_the_release_layer_is_the_last_group() -> None:
    table = synthetic_grid.grid_table(GRID, RELEASE)
    last = table.index[table["header"] == synthetic_grid.RELEASE_NAME][0]
    rows = table.loc[last + 1 :].set_index("setting")
    start = RELEASE["start"]
    assert f"{start['share']:.0%} of blocks" in rows.loc["`start`", "values"]
    assert "Beta(1, 3) on [7, 182] days" in rows.loc["`start`", "values"]
    assert "Normal(1, 0.05)" in rows.loc["`lag_jitter`", "values"]
    assert "`weekly` (25%)" in rows.loc["`cadences` (daily)", "values"]
    assert synthetic_grid.grid_table(GRID, None)["header"].tolist()[-1] != (
        synthetic_grid.RELEASE_NAME
    )


def test_a_factor_with_no_layer_fails() -> None:
    grid = {**GRID, "factor_layers": dict(GRID["factor_layers"])}
    grid["factor_layers"].pop("window")
    with pytest.raises(KeyError, match="window"):
        synthetic_grid.grid_table(grid, None)


def test_distributions_read_as_text() -> None:
    d = synthetic_grid.distribution
    assert d({"dist": "lognormal", "median": 7, "sdlog": 0.55}) == (
        "LogNormal(median 7, sdlog 0.55)"
    )
    assert d({"dist": "halfnormal", "scale": 0.25}) == "HalfNormal(0.25)"
    assert d({"dist": "beta", "a": 2, "b": 2, "lower": 0, "upper": 1}) == (
        "Beta(2, 2)"
    )
    assert (
        d({"dist": "beta", "a": 1, "b": 2, "lower": 0, "upper": 0.9})
        == "Beta(1, 2) on [0, 0.9]"
    )
    assert d({"dist": "gamma", "mean": 120, "sd": 60}) == (
        "Gamma(mean 120, sd 60)"
    )


def test_uneven_levels_show_their_shares() -> None:
    rows = _rows(synthetic_grid.grid_table(GRID, None)).set_index("setting")
    # window: early 5, endemic 7, established 8 of 20.
    assert "`early` (25%)" in rows.loc["`window`", "values"]
    assert "%" not in rows.loc["`gt_family`", "values"]


def test_markdown_has_header_rows_and_escapes_pipes() -> None:
    table = synthetic_grid.grid_table(GRID, RELEASE)
    md = synthetic_grid.to_markdown(table)
    lines = md.strip().splitlines()
    assert len(lines) == len(table) + 2
    assert f"| **{synthetic_grid.LAYER_NAMES['strata']}** | | |" in lines
    # Four unescaped pipes bound the three cells of each row.
    assert all(len(re.findall(r"(?<!\\)\|", line)) == 4 for line in lines[2:])


def test_the_si_includes_one_grid_table() -> None:
    from paperpipe import flow

    assert "grid" in flow.DESIGN_TABLES
    assert not {"grid-epidemic", "grid-reporting", "grid-distributions"} & set(
        flow.DESIGN_TABLES
    )
    si = (PAPER / "si.qmd").read_text()
    assert si.count("{{< include build/tables/si/grid.md >}}") == 1
    assert "grid-epidemic.md" not in si


def test_the_si_notes_name_every_level_and_the_held_out_seeds() -> None:
    config = {
        **CONFIG["synthetic"],
        "heldout_seeds": CONFIG["heldout"]["synthetic_seeds"],
        "development_seeds": CONFIG["development"]["synthetic_seeds"],
    }
    model = synthetic_grid.model_note(
        GRID, "packages/nowcastfoundation/config/finetune/grid.yaml"
    )
    # The factors are in the grid table, not repeated in the note.
    assert "##### Factors" not in model
    assert "PyRenew" in model
    assert "enw_simulate" not in model
    assert "epinowcast is pinned" not in model
    assert "packages/nowcastfoundation/config/finetune/grid.yaml" in model
    assert "finetune_simulate" not in model
    heldout = synthetic_grid.heldout_note(config)
    seeds = config["heldout_seeds"]
    assert f"{seeds[0]} to {seeds[-1]}" in heldout
    # One sentence per line.
    for note in (model, heldout):
        assert all(line.count(". ") == 0 for line in note.splitlines() if line)


def test_the_held_out_note_names_the_grid_file_and_no_machine_path() -> None:
    # It named a retired grid label and printed the simulations' folder.
    config = {
        **CONFIG["synthetic"],
        "simulations": {"paper": "/home/someone/rounds/r38/sims"},
        "heldout_seeds": CONFIG["heldout"]["synthetic_seeds"],
        "development_seeds": CONFIG["development"]["synthetic_seeds"],
    }
    note = synthetic_grid.heldout_note(config)
    assert f"`{CONFIG['synthetic']['grid_file']}`" in note
    assert "/home/" not in note
    assert "on grid `" not in note
    assert "grid" not in CONFIG["synthetic"]
