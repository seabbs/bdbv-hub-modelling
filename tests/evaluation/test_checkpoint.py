"""Checkpoint evaluation: each published checkpoint's scores."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from nowcastgermany import checkpoint as ck


def test_german_dates_forecast_together_come_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SCORE_DATES", raising=False)
    assert ck.score_dates() == 5
    monkeypatch.setenv("SCORE_DATES", "40")
    assert ck.score_dates() == 40


def test_a_unit_records_the_precision_it_forecast_in(tmp_path, monkeypatch):
    # fp16, bf16 and fp32 rows must not pass for one another; a unit run
    # before precisions were recorded ran in fp32.
    assert ck.unit_precision(tmp_path) == "fp32"
    monkeypatch.setenv("DEVICE", "cpu")
    monkeypatch.setenv("PRECISION", "bf16")
    ck.record_precision(tmp_path)
    assert ck.unit_precision(tmp_path) == "bf16"
    assert "precision" in ck.COLUMNS


def test_each_label_runs_the_german_config_s_models(tmp_path):
    # Germany is daily only: the units run the models the German config
    # lists, the reference its zero-shot ones, a checkpoint its `_ft` ones.
    config = tmp_path / "germany.toml"
    config.write_text(
        "[nowcast]\n"
        'models = ["randomwalk", "randomwalknowcast",'
        ' "fm_chronos_daily_complete", "fm_chronos_daily_ragged",'
        ' "fm_chronos_daily_ragged_ft"]\n'
    )
    assert ck.label_models("reference", config) == (
        "randomwalknowcast",
        "fm_chronos_daily_complete",
        "fm_chronos_daily_ragged",
    )
    assert ck.label_models("abc123", config) == ("fm_chronos_daily_ragged_ft",)
    # A checkpoint trained on complete counts runs the complete `_ft` model.
    assert ck.label_models("abc123", config, "complete") == (
        "fm_chronos_daily_complete_ft",
    )


def test_a_segment_s_layout_is_read_from_its_settings(tmp_path):
    import json

    (tmp_path / "settings.json").write_text(json.dumps({"note": "x"}))
    assert ck.segment_layout(tmp_path) == "ragged"
    assert ck.segment_layout(None) == "ragged"
    (tmp_path / "settings.json").write_text(json.dumps({"layout": "complete"}))
    assert ck.segment_layout(tmp_path) == "complete"


def test_the_german_config_has_no_weekly_models():
    models = ck.german_models()
    assert models
    assert not [m for m in models if "weekly" in m]


@pytest.mark.parametrize(
    ("settings", "run"),
    [
        ({"series_bias": True, "sources": ["daily+weekly"]}, "one-model"),
        ({"series_bias": True, "sources": ["daily"]}, "attn-bias"),
        ({"series_slots": {"init": "zero"}}, "slots-zeroinit"),
        ({"series_slots": {"init": "x"}}, "slots"),
        ({}, "no-slots"),
    ],
)
def test_run_of_a_checkpoint(settings, run):
    assert ck.run_of(settings) == run


def test_outbreaks_and_grid_from_settings_or_release_text():
    assert ck.outbreaks_and_grid({"n_outbreaks": 500, "grid": "r5"}, {}) == (
        500,
        "r5",
    )
    release = {
        "name": "x",
        "body": "trained on 2,000 simulated outbreaks, round 4",
    }
    assert ck.outbreaks_and_grid({}, release) == (2000, "r4")
    assert ck.outbreaks_and_grid({}, {}) == (0, "unknown")


def _scores(models, checkpoints, levels, base=1.0):
    rows = []
    for model, chk in zip(models, checkpoints, strict=True):
        for level in levels:
            for origin in ("2026-10-01", "2026-10-08"):
                rows.append(
                    {
                        "model": model,
                        "checkpoint": chk,
                        "level": level,
                        "origin": origin,
                        "truth_vintage": "v",
                        "location": "CD",
                        "target_end_date": "w",
                        "crps": base
                        * (2 if model == "randomwalknowcast" else 1),
                        "log_crps": 1.0,
                        "coverage_50": 1.0,
                        "coverage_90": 1.0,
                        "coverage_95": 1.0,
                    }
                )
    return pd.DataFrame(rows)


def test_relative_divides_by_the_baseline_by_level():
    rows = _scores(
        ["randomwalknowcast", "fm_chronos_daily_ragged"],
        ["zero-shot", "abc"],
        ["national", "zones"],
    )
    got = ck.relative(
        ck.matched(rows, ck.GERMANY_CELL), "ebola", ("crps", "log_crps")
    )
    mine = got.loc[got["checkpoint"] == "abc"]
    assert list(mine["rel_crps"]) == [0.5, 0.5]
    assert set(got["rule"]) == {ck.RULE}
    assert list(got.columns) == ck.COLUMNS


def test_coverage_is_50_and_90_but_95_for_the_german_hub_levels():
    # The German hub's quantiles have no 0.05 or 0.95, so its rows keep
    # the 95% interval the hub scored; BDBV rows have 50% and 90%.
    assert ck.COVERAGE == (50, 90)
    assert ck.GERMANY_COVERAGE == (50, 95)
    rows = _scores(
        ["randomwalknowcast", "fm_chronos_daily_ragged"],
        ["zero-shot", "abc"],
        ["national"],
    )
    bdbv = ck.relative(rows.drop(columns="coverage_95"), "ebola", ("crps",))
    assert bdbv["coverage_90"].notna().all()
    assert bdbv["coverage_95"].isna().all()
    german = rows.drop(columns="coverage_90").assign(coverage_80=1.0)
    de = ck.relative(german, "germany", ("crps",))
    assert de["coverage_95"].notna().all()
    assert de["coverage_90"].isna().all()
    assert "coverage_80" not in de


def test_merge_keeps_other_checkpoints_rule_rows():
    old = pd.DataFrame(
        [{"study": "ebola", "rule": ck.RULE, "checkpoint": "old", "level": "x"}]
    )
    new = pd.DataFrame(
        [{"study": "ebola", "rule": ck.RULE, "checkpoint": "abc", "level": "y"}]
    )
    got = ck.merge_evidence(old, new)
    assert sorted(got["checkpoint"]) == ["abc", "old"]


def test_merge_keeps_only_the_rules_rows():
    old = pd.DataFrame(
        [
            {"study": "ebola", "rule": ck.RULE, "checkpoint": "abc"},
            {"study": "ebola", "rule": "other", "checkpoint": "old"},
            {"study": "ebola", "rule": None, "checkpoint": "old"},
        ]
    ).assign(level="x")
    new = pd.DataFrame(
        [{"study": "ebola", "rule": ck.RULE, "checkpoint": "abc", "level": "y"}]
    )
    got = ck.merge_evidence(old, new)
    assert got["level"].tolist() == ["y"]
    assert set(got["rule"]) == {ck.RULE}


def test_held_out_rows_from_settings():
    settings = {
        "held_out_loss": 1.5,
        "held_out_loss_start": 2.0,
        "held_out_loss_by_level": {"leaf": 1.2},
        "n_validation": 30,
    }
    got = ck.held_out_rows("abc", settings)
    assert list(got["level"]) == ["all", "leaf"]
    assert got["loss"].tolist() == [1.5, 1.2]
    assert list(got.columns) == ck.COLUMNS


def test_the_reported_model_is_the_live_runs_final_checkpoint():
    rows = pd.DataFrame(
        {
            "run": ["one-model", "one-model", "no-slots"],
            "grid": ["r5", "r5", "r5"],
            "outbreaks": [8400, 500, 8400],
        }
    )
    assert ck.chosen(rows).tolist() == [True, False, False]


def test_the_cli_names_its_units():
    with pytest.raises(SystemExit):
        ck.main(["nope", "cache"])


def test_release_unit_writes_settings_and_release(tmp_path, monkeypatch):
    def fake(cmd, **_kw):
        class Out:
            stdout = json.dumps({"name": "n", "body": "b"})
            returncode = 0

        if "download" in cmd:
            out = cmd[cmd.index("-O") + 1]
            Path(out).write_text(json.dumps({"grid": "r5"}))
        return Out()

    monkeypatch.setattr(ck.subprocess, "run", fake)
    ck.release_unit(tmp_path, "abc")
    assert json.loads((tmp_path / "heldout" / "abc.json").read_text()) == {
        "grid": "r5"
    }
    assert (tmp_path / "heldout" / "abc-release.json").exists()


def test_release_unit_merges_an_offloaded_release_s_held_out_losses(
    tmp_path, monkeypatch
):
    def fake(cmd, **_kw):
        class Out:
            stdout = json.dumps(
                {
                    "name": "n",
                    "body": "b",
                    "assets": [{"name": "held-out.json"}],
                }
            )
            returncode = 0

        if "download" in cmd:
            out = cmd[cmd.index("-O") + 1]
            pattern = cmd[cmd.index("-p") + 1]
            data = (
                {"grid": "r6"}
                if pattern == "settings.json"
                else {"held_out_loss": 0.2, "grid": "x"}
            )
            Path(out).write_text(json.dumps(data))
        return Out()

    monkeypatch.setattr(ck.subprocess, "run", fake)
    ck.release_unit(tmp_path, "abc")
    got = json.loads((tmp_path / "heldout" / "abc.json").read_text())
    assert got == {"grid": "r6", "held_out_loss": 0.2}
    release = json.loads(
        (tmp_path / "heldout" / "abc-release.json").read_text()
    )
    assert release == {"name": "n", "body": "b"}


def test_cached_draws_keep_the_7day_target(tmp_path):
    rows = pd.DataFrame(
        {
            "origin": ["2026-10-01"] * 2,
            "location": ["CD"] * 2,
            "target_end_date": ["2026-10-03"] * 2,
            "value": [1.0, 2.0],
            "target": ["7day", "daily"],
        }
    )
    rows.to_csv(tmp_path / "2026-10-01.csv.gz", index=False)
    got = ck._cached(tmp_path)
    assert got["value"].tolist() == [1.0]
    assert "target" not in got


def test_relative_rows_mean_the_wis_components():
    rows = _scores(
        ["randomwalknowcast", "fm_chronos_daily_ragged"],
        ["zero-shot", "abc"],
        ["national"],
    ).assign(wis=3.0, dispersion=1.0, overprediction=0.5, underprediction=1.5)
    got = ck.relative(rows, "ebola", ("crps", "log_crps"))
    row = got.loc[got["checkpoint"] == "abc"].iloc[0]
    assert row["wis"] == 3.0
    assert (
        row["dispersion"] + row["overprediction"] + row["underprediction"]
        == 3.0
    )


def test_tables_score_no_bdbv_held_out_for_the_paper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # BDBV is held out: the tables compute no BDBV rows, and rows already
    # in them stay.
    monkeypatch.setattr(
        ck, "germany_rows", lambda *_a: pd.DataFrame(columns=ck.COLUMNS)
    )
    old = pd.DataFrame(
        [{"study": "ebola", "rule": ck.RULE, "checkpoint": "old",
          "level": "national", "model": "m"}]
    )  # fmt: skip
    evidence = tmp_path / "scores.csv"
    old.reindex(columns=ck.COLUMNS).to_csv(evidence, index=False)
    ck.table(tmp_path, evidence, tmp_path / "checkpoints.csv", [])
    kept = pd.read_csv(evidence)
    assert kept["study"].tolist() == ["ebola"]


def test_a_german_unit_scores_a_run_s_own_segment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A run scores each segment from its folder, before any release, under
    # the label the rule gives it.
    from nowcastfoundation import forecaster

    seen = []
    monkeypatch.setattr(
        ck,
        "germany_unit",
        lambda cache, label, segment=None: seen.append((cache, label, segment)),
    )
    ck.main(["germany", str(tmp_path), "n100", str(tmp_path / "n100")])
    assert seen == [(tmp_path, "n100", tmp_path / "n100")]
    monkeypatch.delenv(forecaster.FINETUNED_ENV, raising=False)
    ck._pin("n100", tmp_path / "n100")
    assert forecaster.finetuned_spec() == str(tmp_path / "n100")
    ck._pin("abc")
    assert forecaster.finetuned_spec() == "abc"


def _saved(train: Path, n: int, **settings) -> Path:
    where = train / f"n{n}"
    where.mkdir(parents=True)
    (where / "settings.json").write_text(
        json.dumps({"hash": f"h{n}", "n_outbreaks": n, **settings})
    )
    return where


def test_a_run_s_table_scores_its_own_saved_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A run's checkpoints are scored from its own folder, under the run's
    # name, beside the shared zero-shot reference.
    train, scores, ref = tmp_path / "train", tmp_path / "s", tmp_path / "r"
    _saved(train, 100, held_out_loss=0.5, held_out_loss_start=0.7)
    offloaded = _saved(train, 200)
    (offloaded / "held-out.json").write_text(
        json.dumps({"held_out_loss": 0.4, "held_out_loss_start": 0.7})
    )
    (train / "n400").mkdir()  # still training: no settings yet
    (scores / "n100").mkdir(parents=True)
    seen = {}

    def germany(folders):
        seen.update(folders)
        return pd.DataFrame(
            [{"study": "germany", "checkpoint": c, "level": "national"}
             for c in ("zero-shot", "n100")]
        ).reindex(columns=ck.COLUMNS)  # fmt: skip

    monkeypatch.setattr(ck, "germany_rows", germany)
    got = ck.run_table(train, scores, ref, "runa")
    assert seen == {ck.REFERENCE: ref, "n100": scores / "n100"}
    held = got.loc[got["study"] == "held_out"].set_index("checkpoint")
    assert held.loc["n200", "loss"] == 0.4
    assert held.loc["n100", "loss"] == 0.5
    by = got.drop_duplicates("checkpoint").set_index("checkpoint")
    assert by.loc["n200", "outbreaks"] == 200
    assert by.loc["n100", "run"] == "runa"
    assert by.loc["zero-shot", "run"] == "zero-shot"


def test_the_comparison_holds_each_run_and_one_zero_shot() -> None:
    def rows(run: str) -> pd.DataFrame:
        return pd.DataFrame(
            [{"study": "germany", "level": "national", "checkpoint": c,
              "run": r, "model": "m"}
             for c, r in (("zero-shot", "zero-shot"), ("n100", run))]
        ).reindex(columns=ck.COLUMNS)  # fmt: skip

    got = ck.comparison([rows("runa"), rows("noslots")])
    assert sorted(got["run"]) == ["noslots", "runa", "zero-shot"]


def test_the_german_config_scores_four_weeks_ahead_too() -> None:
    from nowcastgermany.evaluation import load_germany_config

    config = load_germany_config("config/germany.toml")
    assert config.horizons == tuple(range(28, -29, -1))


TASKS = "tasks-0123456789ab"
"""The stubbed task store's name (`_unit_stubs`)."""


def _unit_stubs(monkeypatch: pytest.MonkeyPatch, seen: list) -> None:
    from nowcastgermany import data as dg
    from nowcastgermany import evaluation as de

    def forecasts(
        _models, _triangle, dates, *_args, horizons, targets, **_kwargs
    ):
        seen.append((list(dates), tuple(horizons), tuple(targets)))
        return pd.DataFrame(
            [
                {"origin": d, "horizon": h, "target": t}
                for d in dates
                for h in horizons
                for t in targets
            ]
        )

    monkeypatch.setattr(dg, "hub_triangle", lambda _path: pd.DataFrame())
    monkeypatch.setattr(de, "germany_forecasts", forecasts)
    monkeypatch.setattr(de, "with_truth", lambda fc, *_a: fc)
    monkeypatch.setattr(ck, "record_precision", lambda _root: None)
    monkeypatch.setattr(ck, "_free", lambda: None)
    monkeypatch.setattr(ck, "german_tasks_dir", lambda *_a: Path(TASKS))
    monkeypatch.setattr(ck, "score_dates", lambda: 100)


def test_a_german_unit_forecasts_the_config_s_horizons(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Every label, the zero-shot reference and each checkpoint, takes the
    # German config's horizons, forecast days included.
    from nowcastgermany.evaluation import load_germany_config

    seen: list = []
    _unit_stubs(monkeypatch, seen)
    ck.germany_unit(tmp_path, "reference")
    want = load_germany_config("config/germany.toml").horizons
    assert {h for _, h, _ in seen} == {want}
    out = pd.read_csv(
        tmp_path / ck.GERMANY_UNIT / "reference" / f"{seen[0][0][0]}.csv.gz"
    )
    assert out["horizon"].max() == max(want)


def test_a_german_unit_redoes_dates_without_the_forecast_days(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A file written before forecast days were scored is forecast again;
    # one that reaches the last horizon is kept.
    from nowcastgermany import evaluation as de

    config = de.load_germany_config("config/germany.toml")
    first, second, *_ = de.nowcast_dates(config, held_out=False)
    root = tmp_path / ck.GERMANY_UNIT / "n100"
    root.mkdir(parents=True)
    pd.DataFrame({"origin": [first], "horizon": [0]}).to_csv(
        root / f"{first}.csv.gz", index=False
    )
    pd.DataFrame(
        {
            "origin": second,
            "horizon": max(config.horizons),
            "target": list(config.targets),
        }
    ).to_csv(root / f"{second}.csv.gz", index=False)
    (root / ck.WRITTEN_FILE).write_text(f"{second}\t{TASKS}\n")
    seen: list = []
    _unit_stubs(monkeypatch, seen)
    ck.germany_unit(tmp_path, "n100")
    done = seen[0][0]
    assert first in done
    assert second not in done


def test_the_evidence_table_scores_the_hub_s_horizons_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Forecast days are for the page; the table stays on the hub's 0 to
    # 28 days back, as the paper scores the teams.
    from nowcastgermany import evaluation as de

    root = tmp_path / ck.GERMANY_UNIT / "reference"
    root.mkdir(parents=True)
    pd.DataFrame(
        {
            "model": "m",
            "origin": "2021-11-22",
            "target_end_date": ["2021-11-22", "2021-11-29"],
            "horizon": [0, 7],
            "value": 1.0,
            "truth": 1.0,
        }
    ).to_csv(root / "2021-11-22.csv.gz", index=False)
    seen: list = []

    def score(fc, _delay):
        seen.append(set(fc["horizon"]))
        return pd.DataFrame(columns=["model", *ck.GERMANY_CELL, "wis"])

    monkeypatch.setattr(de, "score_germany", score)
    monkeypatch.setattr(ck, "relative", lambda sc, *_a: sc)
    monkeypatch.setattr(ck, "matched", lambda sc, _cell: sc)
    ck.germany_rows({"reference": root})
    assert seen
    assert all(s == {0} for s in seen)


def test_the_german_config_gives_both_daily_views() -> None:
    # Germany is shown as daily counts and as the hub's 7-day sums.
    from nowcastgermany.evaluation import load_germany_config

    config = load_germany_config("config/germany.toml")
    assert config.targets == ("7day", "daily")


def test_a_german_unit_forecasts_the_config_s_targets_and_redoes_others(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A date file with the 7-day sums alone, to the last horizon, is
    # forecast again for its daily counts.
    from nowcastgermany import evaluation as de

    config = de.load_germany_config("config/germany.toml")
    first = de.nowcast_dates(config, held_out=False)[0]
    root = tmp_path / ck.GERMANY_UNIT / "n100"
    root.mkdir(parents=True)
    pd.DataFrame(
        {"origin": [first], "horizon": [max(config.horizons)],
         "target": ["7day"]}
    ).to_csv(root / f"{first}.csv.gz", index=False)  # fmt: skip
    seen: list = []
    _unit_stubs(monkeypatch, seen)
    ck.germany_unit(tmp_path, "n100")
    assert first in seen[0][0]
    assert {t for *_, t in seen} == {config.targets}


def _daily_unit(folder: Path, model: str, spread: float) -> None:
    # One origin's daily rows of `model` at DE: quantiles `spread` either
    # side of a truth of 100, so the WIS grows with `spread`.
    levels = (0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "model": model,
                "target": target,
                "origin": "2021-11-22",
                "location": "DE",
                "target_end_date": "2021-11-20",
                "horizon": -2,
                "quantile": q,
                "value": 100 + spread * (q - 0.5) * 4,
                "truth": 130.0,
                "truth_vintage": "2021-12-30",
            }
            for target in ("7day", "daily")
            for q in levels
        ]
    ).to_csv(folder / "2021-11-22.csv.gz", index=False)


def test_daily_rows_are_relative_to_the_config_s_daily_reference(
    tmp_path: Path,
) -> None:
    from nowcastgermany.evaluation import load_germany_config

    config = load_germany_config("config/germany.toml")
    assert config.daily_baseline == "fm_chronos_daily_ragged"
    ref, n100 = tmp_path / "reference", tmp_path / "n100"
    _daily_unit(ref, "fm_chronos_daily_ragged", 10.0)
    _daily_unit(n100, "fm_chronos_daily_ragged_ft", 40.0)
    rows = ck.germany_rows(
        {ck.REFERENCE: ref, "n100": n100},
        target="daily",
        baseline=config.daily_baseline,
    )
    assert set(rows["target"]) == {"daily"}
    one = rows.set_index("checkpoint")["rel_wis"]
    assert one["zero-shot"] == pytest.approx(1.0)
    # Wider quantiles reach the truth of 130: a lower WIS than zero-shot.
    assert one["n100"] < 1.0


def test_a_run_s_daily_table_holds_both_views_by_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The SI's daily comparison: each checkpoint's relative WIS on daily
    # counts and on the 7-day sums, by level, in the comparison layout.
    train = tmp_path / "train"
    _saved(train, 100)
    _saved(train, 200)
    scores = tmp_path / "s"
    for n in (100, 200):
        (scores / f"n{n}").mkdir(parents=True)
    seen = []

    def germany(folders, target="7day", baseline=ck.BASELINE):
        seen.append((target, baseline, sorted(folders)))
        return pd.DataFrame(
            [{"checkpoint": c, "level": "national", "target": target,
              "model": "fm_chronos_daily_ragged",
              "rel_wis": {"zero-shot": 1.0, "n100": 0.8, "n200": 0.6}[c]}
             for c in ("zero-shot", "n100", "n200")]
        ).reindex(columns=ck.COLUMNS)  # fmt: skip

    monkeypatch.setattr(ck, "germany_rows", germany)
    got = ck.daily_table(train, scores, tmp_path / "r", "runa")
    assert {(t, b) for t, b, _ in seen} == {
        ("7day", "randomwalknowcast"),
        ("daily", "fm_chronos_daily_ragged"),
    }
    assert list(got.columns) == ["set", "run", "outbreaks", "score"]
    assert sorted(got["set"].unique()) == [
        "Germany, 7-day sums, national",
        "Germany, daily counts, national",
    ]
    assert set(got["run"]) == {"runa"}
    by = got.set_index(["set", "outbreaks"])["score"]
    assert by[("Germany, daily counts, national", 200)] == 0.6


def test_a_german_unit_redoes_dates_forecast_from_other_tasks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A complete date file is kept only if it was forecast from the
    # current task store: files from other task-building code, or with no
    # record, are forecast again, and each date written is recorded.
    from nowcastgermany import evaluation as de

    config = de.load_germany_config("config/germany.toml")
    old, unrecorded, current, *_ = de.nowcast_dates(config, held_out=False)
    root = tmp_path / ck.GERMANY_UNIT / "n100"
    root.mkdir(parents=True)
    for date in (old, unrecorded, current):
        pd.DataFrame(
            {
                "origin": date,
                "horizon": max(config.horizons),
                "target": list(config.targets),
            }
        ).to_csv(root / f"{date}.csv.gz", index=False)
    (root / ck.WRITTEN_FILE).write_text(
        f"{old}\ttasks-ffffffffffff\n{current}\t{TASKS}\n"
    )
    seen: list = []
    _unit_stubs(monkeypatch, seen)
    ck.germany_unit(tmp_path, "n100")
    done = seen[0][0]
    assert old in done
    assert unrecorded in done
    assert current not in done
    rec = ck.written(root).items()
    got = {k for (d, m), k in rec if d == old and m != ck.EVERY}
    assert got == {TASKS}


def test_a_reference_unit_redoes_only_the_models_whose_code_changed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The reference's forecasts do not depend on a checkpoint, so each
    # date's rows of a model are kept until that model's code or the
    # German config changes (`model_key`); a change redoes that model's
    # rows alone and keeps the others'.
    from nowcastgermany import evaluation as de

    config = de.load_germany_config("config/germany.toml")
    dates = de.nowcast_dates(config)
    seen: list = []
    _unit_stubs(monkeypatch, seen)
    asked: list = []
    real = de.germany_forecasts

    def forecasts(models, *args, **kwargs):
        asked.append(tuple(models))
        return real(models, *args, **kwargs).assign(model=models[0])

    monkeypatch.setattr(ck, "label_models", lambda *_a, **_k: ("a", "b"))
    keys = {"a": "code-a", "b": "code-b"}
    monkeypatch.setattr(ck, "model_key", lambda m: keys[m])
    root = tmp_path / ck.GERMANY_UNIT / ck.REFERENCE
    monkeypatch.setattr(de, "germany_forecasts", forecasts)
    ck.germany_unit(tmp_path, ck.REFERENCE)
    assert asked == [("a", "b")]
    keys["b"] = "code-b2"
    asked.clear()
    ck.germany_unit(tmp_path, ck.REFERENCE)
    assert asked == [("b",)]
    one = pd.read_csv(root / f"{dates[0]}.csv.gz")
    assert set(one["model"]) == {"a", "b"}
    asked.clear()
    ck.germany_unit(tmp_path, ck.REFERENCE)
    assert asked == []


def test_a_checkpoint_unit_keys_its_dates_on_the_task_store_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A checkpoint's forecasts are its own: its dates are kept while the
    # task store is the same, whatever the models' code.
    seen: list = []
    _unit_stubs(monkeypatch, seen)
    monkeypatch.setattr(ck, "model_key", lambda _m: "changes every call")
    ck.germany_unit(tmp_path, "n100")
    seen.clear()
    ck.germany_unit(tmp_path, "n100")
    assert seen == []
