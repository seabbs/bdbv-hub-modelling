"""The BDBV foundation-model check runs Chronos-2 with the active-zone rule."""

import pytest

pytest.importorskip("jax")


def test_chronos_models_take_the_active_zone_rule_and_others_do_not():
    from bdbvhub.ftcheck import check_model
    from nowcastfoundation.forecaster import active_zone_rule

    cases, weeks, recent = active_zone_rule()
    zero = check_model("fm_chronos_daily_ragged")
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
    conf = tmp_path / "config" / "finetune.toml"
    conf.write_text('hash = "a"\nactive_zone_cases = 20\n')
    one = scoring.ft_check_key(tmp_path)
    conf.write_text('hash = "b"\nactive_zone_cases = 20\n')
    assert scoring.ft_check_key(tmp_path) == one
    conf.write_text('hash = "b"\nactive_zone_cases = 10\n')
    assert scoring.ft_check_key(tmp_path) != one


def test_release_quantiles_take_the_explorer_s_columns():
    # The release backtest names its 5% quantile `q5`; the explorer reads
    # `q05` to `q95`.
    import pandas as pd

    from bdbvhub.ftcheck import explorer_quantiles
    from nowcasteval.explorer import QUANTILE_COLUMNS

    q = pd.DataFrame(
        {
            "model": ["randomwalknowcast"],
            "target": ["complete"],
            "origin": ["2026-11-15"],
            "location": ["CD"],
            "target_end_date": ["2026-11-08"],
            "q5": [1.0],
            "q25": [2.0],
            "q50": [3.0],
            "q75": [4.0],
            "q95": [5.0],
        }
    )
    got = explorer_quantiles(q, {"randomwalknowcast": "Random walk nowcast"})
    assert list(got[list(QUANTILE_COLUMNS)].iloc[0]) == [1, 2, 3, 4, 5]
    assert got["model"].iloc[0] == "Random walk nowcast"
    assert got["reference_date"].iloc[0] == "2026-11-15"
    assert bool(got["ours"].iloc[0])


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
    from bdbvhub.ftcheck import calibration

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

    from bdbvhub.ftcheck import roughness

    got = roughness(_quantiles()).set_index(["model", "level"])
    # log1p(11) - log1p(10) for the medians, log1p(12) - log1p(10) for truth.
    med = np.log1p(11) - np.log1p(10)
    truth = np.log1p(12) - np.log1p(10)
    assert got.loc[("a", "national"), "median"] == pytest.approx(med)
    assert got.loc[("a", "national"), "truth"] == pytest.approx(truth)
