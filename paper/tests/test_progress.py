"""A fine-tuning run's progress figure (paperpipe.progress)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from paperpipe.progress import HORIZON_TITLES, progress_figure


def _loss() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"kind": "training", "series": "all", "step": s, "loss": 1 / s}
            for s in range(10, 200, 10)
        ]
        + [
            {"kind": "validation", "series": "daily", "step": 100, "loss": 0.3},
            {
                "kind": "checkpoint",
                "series": "n100",
                "step": 100,
                "outbreaks": 100,
            },
        ]
    )


def _horizons() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "horizons": part,
                "study": "germany",
                "level": "national",
                "model": model,
                "checkpoint": ck,
                "outbreaks": n,
                "n": 400,
                "rel_wis": value,
            }
            for part in HORIZON_TITLES
            for model, ck, n, value in (
                ("foundation_zeroshot", "zero-shot", 0, 2.0),
                ("foundation_zeroshot", "n100", 100, 1.5),
                ("foundation_zeroshot", "n200", 200, 1.2),
                ("foundation_zeroshot_complete", "zero-shot", 0, 1.7),
            )
        ]
    )


def _seven_day() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "outbreaks": [100, 200],
            "ratio": [0.9, 1.0],
            "inside90": [0.9, 0.9],
            "rough_ratio": [1.5, 1.2],
            "n": 50,
        }
    )


def _titles(fig) -> list[str]:
    return [ax.get_title() for ax in fig.get_axes() if ax.get_visible()]


def test_the_figure_has_each_standard_panel() -> None:
    fig = progress_figure(_loss(), _horizons(), _seven_day(), None, run="runa")
    titles = _titles(fig)
    assert titles[:3] == ["Loss", "7-day check", "BDBV check, relative CRPS"]
    # One German row per part of the horizons, as the checkpoint curve.
    assert titles.count("germany, national (n = 400)") == len(HORIZON_TITLES)
    subtitles = [s._suptitle.get_text() for s in fig.subfigs[1:]]
    assert subtitles == list(HORIZON_TITLES.values())
    # Only the last German row carries the shared legend.
    assert [len(s.legends) for s in fig.subfigs[1:]] == [0, 0, 1]


def test_a_run_on_real_data_alone_is_drawn_by_its_share_of_training() -> None:
    # Its checkpoints have no outbreaks; n<N> are thousandths of its passes.
    rows = _horizons()
    tuned = rows["checkpoint"] != "zero-shot"
    rows.loc[tuned, "checkpoint"] = rows.loc[tuned, "checkpoint"].map(
        {"n100": "n500", "n200": "n1000"}
    )
    rows["outbreaks"] = 0
    fig = progress_figure(_loss(), rows, _seven_day(), None, run="realdata")
    german = [
        ax for ax in fig.get_axes() if ax.get_title().startswith("germany")
    ]
    assert german
    for ax in german:
        assert ax.get_xlabel() == "Share of the passes"
        assert [t.get_text() for t in ax.get_xticklabels()] == ["1/2", "all"]


def test_the_bdbv_panel_is_held_out_of_development_runs() -> None:
    fig = progress_figure(_loss(), _horizons(), _seven_day(), None)
    bdbv = fig.get_axes()[2]
    assert "Held out" in bdbv.texts[0].get_text()


def test_the_bdbv_panel_draws_the_fine_tuned_model_by_checkpoint() -> None:
    bdbv = pd.DataFrame(
        {
            "outbreaks": [100, 100, 200, 200],
            "model": ["foundation", "foundation_zeroshot"] * 2,
            "relative_crps": [0.9, 1.1, 0.8, 1.1],
        }
    )
    fig = progress_figure(_loss(), _horizons(), _seven_day(), bdbv)
    ax = fig.get_axes()[2]
    lines = {line.get_label(): line for line in ax.get_lines()}
    tuned = next(v for k, v in lines.items() if k.endswith("fine-tuned"))
    assert list(tuned.get_xdata()) == [100, 200]
    assert not ax.texts


def test_panels_without_rows_say_so() -> None:
    empty = pd.DataFrame(
        columns=["kind", "series", "step", "outbreaks", "loss"]
    )
    seven = _seven_day().iloc[:0]
    horizons = _horizons().iloc[:0]
    fig = progress_figure(empty, horizons, seven, None)
    texts = [t.get_text() for ax in fig.get_axes() for t in ax.texts]
    assert "No losses logged yet" in texts
    assert "No checkpoint checked yet" in texts
    assert "No German checkpoint scores yet" in texts


def test_the_cli_writes_the_figure(tmp_path: Path) -> None:
    from paperpipe.__main__ import main

    files = {
        "loss": _loss(),
        "horizons": _horizons(),
        "seven-day": _seven_day(),
    }
    for name, frame in files.items():
        frame.to_csv(tmp_path / f"{name}.csv", index=False)
    out = tmp_path / "si" / "progress-runa.png"
    main(
        [
            "progress",
            *(a for n in files for a in (f"--{n}", str(tmp_path / f"{n}.csv"))),
            "--title",
            "runa",
            "--out",
            str(out),
        ]
    )
    assert out.stat().st_size > 0
