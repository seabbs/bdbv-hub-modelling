"""Stored forecasts of the fine-tuned model on the German benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from nowcastgermany.evaluation import load_germany_config
from nowcastgermany.store import (
    STORED_MODELS,
    development_rows,
    load_stored,
    release_tag,
    write_stored,
)

ROOT = Path(__file__).resolve().parents[3]
CONFIG = load_germany_config(ROOT / "config" / "germany.toml")
SPLIT = CONFIG.split_date


DAILY = "foundation"


def _cache(origins: tuple[str, ...], model: str = DAILY) -> pd.DataFrame:
    rows = []
    for origin in origins:
        for h in (0, -7):
            end = pd.Timestamp(origin) + pd.Timedelta(days=h)
            for q in (0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975):
                rows.append(
                    {
                        "model": model,
                        "origin": origin,
                        "location": "DE",
                        "target_end_date": str(end)[:10],
                        "horizon": h,
                        "quantile": q,
                        "value": 100.0 * q,
                        "truth": 50.0,
                        "truth_vintage": str(end + pd.Timedelta(days=40))[:10],
                    }
                )
    return pd.DataFrame(rows)


def _release(tag: str, published: str, draft: bool = False) -> dict:
    return {"tag_name": tag, "published_at": published, "draft": draft}


RELEASES = [
    _release("chronos-ft-aaa", "2026-11-15T10:00:00Z"),
    _release("chronos-ft-bbb", "2026-11-16T10:00:00Z"),
    _release("realdata-ft-ccc", "2026-11-16T12:00:00Z"),
]


def test_release_tag_maps_scorer_labels() -> None:
    assert release_tag("deadbeef0005") == "chronos-ft-deadbeef0005"
    assert release_tag("slots-1234abcd") == "chronos-slots-1234abcd"
    for bad in ("reference", "", "slots-", "../x", "chronos-ft-abc"):
        with pytest.raises(ValueError, match="label"):
            release_tag(bad)


def test_development_rows_keep_the_model_before_the_split() -> None:
    frame = pd.concat(
        [
            _cache(("2021-11-22", "2022-02-09")),
            _cache(("2021-11-22",), model="foundation_zeroshot"),
        ]
    )
    out = development_rows(frame, SPLIT)
    assert set(out["model"]) == {DAILY}
    assert out["origin"].max() == "2022-02-09"
    assert (out["target_end_date"] < SPLIT).all()


def test_development_rows_refuse_held_out_dates() -> None:
    frame = _cache(("2022-02-09", SPLIT))
    with pytest.raises(ValueError, match="held-out"):
        development_rows(frame, SPLIT)


def test_write_then_load_round_trips(tmp_path: Path) -> None:
    frame = _cache(("2021-11-22", "2021-11-23"))
    store = tmp_path / "store"
    write_stored(store, "chronos-ft-bbb", frame, SPLIT, label="bbb")
    manifest = json.loads(
        (store / "chronos-ft-bbb" / "manifest.json").read_text()
    )
    assert manifest["models"] == [DAILY]
    assert manifest["first_origin"] == "2021-11-22"
    assert manifest["last_origin"] == "2021-11-23"
    assert manifest["split_date"] == SPLIT
    back = load_stored(store, "chronos-ft-bbb", SPLIT)
    pd.testing.assert_frame_equal(
        back.reset_index(drop=True),
        development_rows(frame, SPLIT).reset_index(drop=True),
        check_dtype=False,
    )


def test_load_refuses_a_store_with_held_out_dates(tmp_path: Path) -> None:
    path = tmp_path / "store" / "chronos-ft-bbb"
    path.mkdir(parents=True)
    _cache(("2022-02-09", "2022-02-11")).to_csv(
        path / "forecasts.csv.gz", index=False
    )
    with pytest.raises(ValueError, match="held-out"):
        load_stored(tmp_path / "store", "chronos-ft-bbb", SPLIT)


def test_the_daily_fine_tuned_model_is_stored() -> None:
    # Germany is daily only: no weekly model is stored.
    assert STORED_MODELS == (DAILY,)


def _pin(tmp_path: Path, spec: str) -> Path:
    path = tmp_path / "finetune.toml"
    path.write_text(f'hash = "{spec}"\nrealdata = ""\n')
    return path


def _load(tmp_path: Path, store: Path, spec: str) -> tuple[dict, Path]:
    import os

    from nowcastfoundation import forecaster as foundation
    from nowcastgermany import cli

    os.environ.pop(foundation.FINETUNED_ENV, None)
    out = tmp_path / "parts" / "germany"
    note = tmp_path / "results" / "ft-checkpoint.json"
    args = [
        "ft-load",
        "--config",
        str(ROOT / "config" / "germany.toml"),
        "--finetune",
        str(_pin(tmp_path, spec)),
        "--store",
        str(store),
        "--out",
        str(out),
        "--note",
        str(note),
    ]
    assert cli.main(args) == 0
    return json.loads(note.read_text()), out


def test_the_load_command_writes_the_pinned_checkpoint(tmp_path: Path) -> None:
    # A newer stored checkpoint does not stand in for the pin.
    store = tmp_path / "store"
    for tag in ("chronos-slots-aaaaaa", "chronos-slots-bbbbbb"):
        write_stored(store, tag, _cache(("2021-11-22",)), SPLIT, label=tag)
    meta, out = _load(tmp_path, store, "slots-aaaaaa")
    assert meta == {
        "tag": "chronos-slots-aaaaaa",
        "pinned": "chronos-slots-aaaaaa",
        "models": list(STORED_MODELS),
    }
    scores = pd.read_csv(out / "scores.csv.gz")
    assert set(scores["model"]) == {DAILY}
    assert (scores["origin"] < SPLIT).all()


def test_the_load_command_writes_nothing_when_the_pin_is_not_stored(
    tmp_path: Path,
) -> None:
    store = tmp_path / "store"
    write_stored(
        store, "chronos-slots-bbbbbb", _cache(("2021-11-22",)), SPLIT, label="b"
    )
    meta, out = _load(tmp_path, store, "slots-aaaaaa")
    assert meta["tag"] is None
    assert meta["pinned"] == "chronos-slots-aaaaaa"
    assert not (out / "forecasts.csv.gz").exists()
