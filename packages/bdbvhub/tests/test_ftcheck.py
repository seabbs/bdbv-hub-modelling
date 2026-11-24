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
