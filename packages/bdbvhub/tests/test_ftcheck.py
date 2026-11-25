"""The BDBV foundation-model check runs Chronos-2 with the active-zone rule."""

import pytest

pytest.importorskip("jax")


def test_chronos_models_take_the_active_zone_rule_and_others_do_not():
    from bdbvhub.ftcheck import check_model
    from nowcastfoundation.forecaster import active_zone_rule

    cases, weeks, recent = active_zone_rule()
    zero = check_model("foundation_zeroshot")
    assert (
        zero.active_zone_cases,
        zero.active_zone_weeks,
        zero.active_zone_recent_weeks,
    ) == (cases, weeks, recent)
    walk = check_model("randomwalknowcast")
    assert not hasattr(walk, "active_zone_cases")


def test_the_check_key_moves_with_the_active_zone_rule(tmp_path):
    import shutil
    import sys
    from pathlib import Path

    root = Path(__file__).parents[3]
    sys.path.insert(0, str(root / "workflow"))
    from flowlib import scoring

    for part in scoring.FT_CHECK_CODE:
        src = root / part
        dst = tmp_path / part
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)
    conf = (
        tmp_path / "packages" / "nowcastfoundation" / "config" / "finetune.toml"
    )
    conf.parent.mkdir(parents=True, exist_ok=True)
    conf.write_text('hash = "a"\nactive_zone_cases = 20\n')
    one = scoring.ft_check_key(tmp_path)
    conf.write_text('hash = "b"\nactive_zone_cases = 20\n')
    assert scoring.ft_check_key(tmp_path) == one
    conf.write_text('hash = "b"\nactive_zone_cases = 10\n')
    assert scoring.ft_check_key(tmp_path) != one


def _quantiles():
    import pandas as pd

    rows = []
    for model, med in (("a", 10.0), ("b", 20.0)):
        for loc in ("CD", "CD-IT"):
            for wb in (-1, 0):
                for k, week in enumerate(("2026-11-01", "2026-11-08")):
                    rows.append(
                        {
                            "model": model,
                            "target": "complete",
                            "origin": "2026-11-15",
                            "location": loc,
                            "target_end_date": week,
                            "weeks_back": wb,
                            "q5": med / 2,
                            "q25": med * 0.8,
                            "q50": med + k,
                            "q75": med * 1.2,
                            "q95": med * 2,
                            "reported": 5.0,
                            "truth": 10.0 + 2 * k,
                        }
                    )
    return pd.DataFrame(rows)


def test_calibration_gives_median_over_truth_and_reported_and_coverage():
    from bdbvhub.reporting.ftcheck_report import calibration

    got = calibration(_quantiles()).set_index(["model", "level", "weeks_back"])
    a = got.loc[("a", "national", 0)]
    # Medians 10 and 11 against truth 10 and 12, reported 5 each week.
    assert a["median_over_truth"] == pytest.approx(21 / 22)
    assert a["median_over_reported"] == pytest.approx(21 / 10)
    assert a["coverage_90"] == 1.0
    b = got.loc[("b", "province", -1)]
    # The 90% interval, 10 to 40, covers truth 10 and 12.
    assert b["coverage_90"] == 1.0
    assert b["median_over_truth"] == pytest.approx(41 / 22)
    assert set(got.index.get_level_values("level")) == {"national", "province"}


def test_roughness_compares_week_to_week_changes_with_the_truth_s():
    import numpy as np

    from bdbvhub.reporting.ftcheck_report import roughness

    got = roughness(_quantiles()).set_index(["model", "level"])
    # log1p(11) - log1p(10) for the medians, log1p(12) - log1p(10) for truth.
    med = np.log1p(11) - np.log1p(10)
    truth = np.log1p(12) - np.log1p(10)
    assert got.loc[("a", "national"), "median"] == pytest.approx(med)
    assert got.loc[("a", "national"), "truth"] == pytest.approx(truth)


def test_weeks_with_nothing_reported_have_no_ratio_to_it():
    import numpy as np

    from bdbvhub.reporting.ftcheck_report import calibration

    q = _quantiles().assign(reported=0.0)
    got = calibration(q)
    assert got["median_over_reported"].isna().all()
    assert np.isfinite(got["median_over_truth"]).all()


def test_the_page_carries_the_explorer_s_styles_and_data(tmp_path):
    # The explorer needs the docs stylesheet (legend swatches, the shaded
    # incomplete weeks) and its data in the shape it reads.
    import json
    from pathlib import Path

    import pandas as pd

    from bdbvhub.reporting.ftcheck_report import write_report
    from nowcasteval.explorer import QUANTILE_COLUMNS

    root = Path(__file__).parents[3]
    ids = {"a": "randomwalknowcast", "b": "foundation"}
    q = _quantiles().replace({"model": ids})
    target = pd.DataFrame(
        {
            "kind": "real",
            "vintage": "2026-11-22",
            "location": ["CD", "CD-IT"] * 2,
            "target_end_date": ["2026-11-01"] * 2 + ["2026-11-08"] * 2,
            "total": [10, 5, 12, 6],
            "location_name": ["DRC", "Ituri"] * 2,
        }
    )
    skill = {
        "relative_crps": [1.0, 0.9],
        "relative_log_crps": [1.0, 0.9],
        "n": [8, 8],
        "rounds": [1, 1],
    }
    tables = {
        "overall": pd.DataFrame({"model": list(ids.values()), **skill}),
        "by_weeks_back": pd.DataFrame(
            {"model": list(ids.values()), "weeks_back": [0, 0], **skill}
        ),
        "quantiles": q,
        "samples": _samples(),
        "blocks": pd.DataFrame(),
        "target": target,
    }
    out = write_report(
        tables, tmp_path / "p.html", "randomwalknowcast", "slots-abcdef", root
    )
    page = out.read_text()
    css = (root / "docs" / "stylesheets" / "extra.css").read_text()
    assert ".explorer" in css
    assert css in page
    assert "data-md-color-scheme" in page
    assert "bdbvExplorer" in page or "explorer" in page
    data = json.loads((tmp_path / "p-data" / "explorer.json").read_text())
    assert len(data["models"]) == 2
    row = data["rows"][0] if "rows" in data else data["q"][0]
    assert len(row) == 4 + len(QUANTILE_COLUMNS)


def _samples():
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(1)
    rows = []
    for model in ("randomwalknowcast", "foundation"):
        for week in ("2026-11-08", "2026-11-15", "2026-11-22"):
            for k, v in enumerate(rng.poisson(100, 200)):
                rows.append(
                    {
                        "model": model,
                        "origin": "2026-11-15",
                        "location": "CD",
                        "target_end_date": week,
                        "sample": k + 1,
                        "value": float(v),
                    }
                )
    return pd.DataFrame(rows)


def test_the_explorer_shows_each_model_s_final_count_forecast():
    # Every forecast week, complete or not, once, with the quantiles of
    # the model's own final-count draws (as the this-round page).
    import numpy as np

    from bdbvhub.reporting.ftcheck_report import explorer_quantiles
    from nowcasteval.explorer import QUANTILE_COLUMNS

    s = _samples()
    got = explorer_quantiles(s, {"randomwalknowcast": "Random walk nowcast"})
    keys = ["model", "reference_date", "location", "target_end_date"]
    assert not got.duplicated(keys).any()
    assert set(got["target_end_date"]) == {
        "2026-11-08",
        "2026-11-15",
        "2026-11-22",
    }
    one = got.loc[
        (got["model"] == "Random walk nowcast")
        & (got["target_end_date"] == "2026-11-22")
    ].iloc[0]
    own = s.loc[
        (s["model"] == "randomwalknowcast")
        & (s["target_end_date"] == "2026-11-22"),
        "value",
    ]
    assert one["q50"] == np.quantile(own, 0.5)
    assert set(QUANTILE_COLUMNS) <= set(got.columns)
    assert bool(one["ours"])
    assert one["reference_date"] == "2026-11-15"


def test_the_forecast_reads_the_backtest_config_it_is_given(monkeypatch):
    from bdbvhub import cli, ftcheck

    seen = {}

    def stop(_data, config):
        seen["config"] = config
        raise RuntimeError("stop")

    monkeypatch.setattr(ftcheck, "release_tasks", stop)
    with pytest.raises(RuntimeError, match="stop"):
        cli.main(
            [
                "ft-check",
                "forecast",
                "--model",
                "randomwalknowcast",
                "--config",
                "ablation.toml",
                "--out",
                "out",
            ]
        )
    assert seen["config"] == "ablation.toml"
