"""Our German forecasts from the fine-tuning workflow's units.

The reference unit and the paper's checkpoint unit hold every model's
German development forecasts (`nowcastgermany.checkpoint`). `units`
stacks them as the paper's development table, `release_paper_germany`
publishes it, and Site results reads that release (`stored-load`), so the
site and the paper show the same forecasts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from nowcastgermany import cli
from nowcastgermany.evaluation import load_germany_config
from nowcastgermany.store import (
    NOTE,
    UNITS,
    development_rows,
    load_units,
    read_unit,
    units_forecasts,
)

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "packages" / "nowcastgermany" / "config" / "germany.toml"
CONFIG = load_germany_config(CONFIG_PATH)
SPLIT = CONFIG.split_date
REFERENCE = ("randomwalknowcast", "foundation_zeroshot")
CHECKPOINT = ("foundation", "foundation_baselinenowcast")


def _rows(origins: tuple[str, ...], models: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for model in models:
        for origin in origins:
            for target in ("7day", "daily"):
                for h in (0, -7, 7):
                    end = pd.Timestamp(origin) + pd.Timedelta(days=h)
                    for q in (0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975):
                        rows.append(
                            {
                                "model": model,
                                "target": target,
                                "origin": origin,
                                "location": "DE",
                                "target_end_date": str(end)[:10],
                                "horizon": h,
                                "quantile": q,
                                "value": 100.0 * q,
                                "truth": 50.0,
                                "truth_vintage": str(
                                    end + pd.Timedelta(days=40)
                                )[:10],
                            }
                        )
    return pd.DataFrame(rows)


def _unit(
    root: Path, origins: tuple[str, ...], models: tuple[str, ...]
) -> Path:
    root.mkdir(parents=True)
    for origin in origins:
        _rows((origin,), models).to_csv(root / f"{origin}.csv.gz", index=False)
    (root / "written.tsv").write_text("")
    return root


def _units(tmp_path: Path, last: str = "2021-11-23") -> tuple[Path, Path]:
    dates = ("2021-11-22", last)
    reference = _unit(tmp_path / "scores" / "reference", dates, REFERENCE)
    checkpoint = _unit(tmp_path / "runa" / "n4800", dates, CHECKPOINT)
    return reference, checkpoint


def test_a_unit_reads_as_the_hub_s_7_day_nowcasts(tmp_path: Path) -> None:
    reference, _ = _units(tmp_path)
    rows = read_unit(reference)
    assert "target" not in rows
    assert set(rows["horizon"]) == {0, -7}
    assert set(rows["model"]) == set(REFERENCE)
    assert rows["origin"].nunique() == 2


def test_a_unit_with_no_dates_is_refused(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="No German unit files"):
        read_unit(tmp_path / "empty")


def test_the_units_stack_every_model_of_the_config(tmp_path: Path) -> None:
    units = _units(tmp_path)
    rows = units_forecasts(units, SPLIT, CONFIG.models)
    assert set(rows["model"]) == {*REFERENCE, *CHECKPOINT}
    assert rows["origin"].max() == "2021-11-23"


def test_a_model_the_config_does_not_list_is_left_out(tmp_path: Path) -> None:
    reference, checkpoint = _units(tmp_path)
    extra = _unit(tmp_path / "extra", ("2021-11-22",), ("not_a_model",))
    rows = units_forecasts([reference, checkpoint, extra], SPLIT, CONFIG.models)
    assert "not_a_model" not in set(rows["model"])


def test_development_rows_refuse_held_out_dates() -> None:
    frame = _rows(("2022-02-09", SPLIT), REFERENCE).drop(columns="target")
    with pytest.raises(ValueError, match="held-out"):
        development_rows(frame, SPLIT, CONFIG.models)


def test_a_unit_with_a_held_out_date_is_refused(tmp_path: Path) -> None:
    units = _units(tmp_path, last=SPLIT)
    with pytest.raises(ValueError, match="held-out"):
        units_forecasts(units, SPLIT, CONFIG.models)


def _record(tmp_path: Path, tag: str = "chronos-slots-66a7a630d729") -> Path:
    path = tmp_path / "runa" / "train" / "n4800" / "published.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"tag": tag, "hash": tag.rsplit("-", 1)[1]}))
    return path


def test_the_units_command_writes_the_development_table(
    tmp_path: Path,
) -> None:
    reference, checkpoint = _units(tmp_path)
    out = tmp_path / "results" / "germany"
    args = ["units", str(reference), str(checkpoint), "--out", str(out)]
    args += ["--config", str(CONFIG_PATH), "--record", str(_record(tmp_path))]
    assert cli.main(args) == 0
    forecasts = pd.read_csv(out / "forecasts.csv.gz")
    assert set(forecasts["model"]) == {*REFERENCE, *CHECKPOINT}
    scores = pd.read_csv(out / "scores.csv.gz")
    assert set(scores["model"]) == {*REFERENCE, *CHECKPOINT}
    assert (scores["origin"] < SPLIT).all()
    note = json.loads((out / NOTE).read_text())
    assert note == {
        "checkpoint": "chronos-slots-66a7a630d729",
        "models": sorted({*REFERENCE, *CHECKPOINT}),
    }


def test_the_units_command_runs_without_a_checkpoint(tmp_path: Path) -> None:
    reference, _ = _units(tmp_path)
    out = tmp_path / "out"
    args = ["units", str(reference), "--out", str(out)]
    assert cli.main([*args, "--config", str(CONFIG_PATH)]) == 0
    assert json.loads((out / NOTE).read_text())["checkpoint"] is None


def _published(tmp_path: Path) -> Path:
    """Unpack a paper-germany release as `paperpipe.stored` does."""
    reference, checkpoint = _units(tmp_path)
    made = tmp_path / "made"
    args = ["units", str(reference), str(checkpoint), "--out", str(made)]
    args += ["--config", str(CONFIG_PATH), "--record", str(_record(tmp_path))]
    assert cli.main(args) == 0
    folder = tmp_path / "paper-stored" / "germany"
    folder.mkdir(parents=True)
    (folder / UNITS).write_bytes((made / "forecasts.csv.gz").read_bytes())
    (folder / f"germany-{NOTE}").write_text((made / NOTE).read_text())
    manifest = {"set": "germany", "tag": "paper-germany-0123456789ab"}
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return folder


def test_the_site_loads_the_published_units(tmp_path: Path) -> None:
    folder = _published(tmp_path)
    out = tmp_path / "parts" / "germany"
    note = tmp_path / "results" / "germany" / NOTE
    args = ["stored-load", "--store", str(folder), "--out", str(out)]
    args += ["--note", str(note), "--config", str(CONFIG_PATH)]
    assert cli.main(args) == 0
    scores = pd.read_csv(out / "scores.csv.gz")
    assert set(scores["model"]) == {*REFERENCE, *CHECKPOINT}
    made = pd.read_csv(tmp_path / "made" / "scores.csv.gz")
    pd.testing.assert_frame_equal(scores, made)
    assert json.loads(note.read_text()) == {
        "release": "paper-germany-0123456789ab",
        "checkpoint": "chronos-slots-66a7a630d729",
        "models": sorted({*REFERENCE, *CHECKPOINT}),
    }


def test_the_site_writes_nothing_with_no_release(tmp_path: Path) -> None:
    out = tmp_path / "parts" / "germany"
    note = tmp_path / "results" / "germany" / NOTE
    args = ["stored-load", "--store", str(tmp_path / "none"), "--out"]
    args += [str(out), "--note", str(note), "--config", str(CONFIG_PATH)]
    assert cli.main(args) == 0
    assert not (out / "forecasts.csv.gz").exists()
    assert json.loads(note.read_text())["release"] is None


def test_a_published_table_with_a_held_out_date_is_refused(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "germany"
    folder.mkdir()
    rows = _rows(("2022-02-09", "2022-02-11"), REFERENCE)
    rows.drop(columns="target").to_csv(folder / UNITS, index=False)
    with pytest.raises(ValueError, match="held-out"):
        load_units(folder, SPLIT, CONFIG.models)
