"""The renewal model variants compared on the Ebola hub rounds."""

import pandas as pd
import pytest
import yaml

from paperpipe.evidence import write_evidence


def _scores() -> pd.DataFrame:
    """Ebola WIS rows: renewalbvd halves renewal, renewalsp doubles it.

    `renewalsp` misses one zone, so its pair is scored on the rest.
    """
    rows = []
    places = ("CD", "CD-KS", "CD-KS-BU", "CD-KS-MW")
    for model, scale in (
        ("renewal", 2.0),
        ("renewalbvd", 1.0),
        ("renewalsp", 4.0),
        ("randomwalknowcast", 3.0),
    ):
        for place in places:
            if model == "renewalsp" and place == "CD-KS-MW":
                continue
            for origin in ("2026-07-04", "2026-07-11"):
                rows.append(
                    {
                        "study": "ebola",
                        "model": model,
                        "location": place,
                        "stratum": "all",
                        "origin_date": pd.Timestamp(origin),
                        "target_date": pd.Timestamp(origin),
                        "horizon": 0,
                        "truth": 5.0,
                        "metric": "wis",
                        "score": scale,
                        "log_score": scale / 10,
                        "bias": 0.0,
                    }
                )
    return pd.DataFrame(rows)


def test_each_pair_is_compared_by_level_on_the_forecasts_both_made() -> None:
    from paperpipe.renewal import renewal_variants

    out = renewal_variants(_scores()).set_index(["comparison", "level"])
    assert set(out.index.get_level_values("level")) == {
        "national",
        "provinces",
        "zones",
    }
    assert out.loc[("bvd", "national"), "rel_score"] == pytest.approx(0.5)
    assert out.loc[("bvd", "zones"), "n"] == 4
    assert out.loc[("pooling", "zones"), "rel_score"] == pytest.approx(0.5)
    # renewalsp has one zone fewer, so the pair shares two of them.
    assert out.loc[("pooling", "zones"), "n"] == 2
    assert out.loc[("pooling", "national"), "rel_log_score"] == pytest.approx(
        0.5
    )


def test_the_renewal_step_writes_the_comparison(tmp_path) -> None:
    from paperpipe.__main__ import main

    scores = tmp_path / "ebola.csv.gz"
    _scores().to_csv(scores, index=False)
    out = tmp_path / "renewal-variants.csv"
    main(["renewal-variants", "--scores", str(scores), "--out", str(out)])
    got = pd.read_csv(out)
    assert set(got["comparison"]) == {"bvd", "pooling"}


def test_the_comparison_is_quoted_and_tabled(tmp_path) -> None:
    from paperpipe.renewal import renewal_variants

    out = write_evidence(
        {}, tmp_path / "si", tmp_path / "f", tmp_path / "e.yml"
    )
    assert out["evidence"]["renewal"]["bvd"]["national"] == (
        "(not yet available)"
    )
    assert "waits on" in (tmp_path / "si" / "renewal-variants.md").read_text()
    rows = tmp_path / "renewal-variants.csv"
    renewal_variants(_scores()).to_csv(rows, index=False)
    write_evidence(
        {"renewal_variants": rows},
        tmp_path / "si",
        tmp_path / "f",
        tmp_path / "e.yml",
    )
    doc = yaml.safe_load((tmp_path / "e.yml").read_text())["evidence"]
    assert doc["renewal"]["bvd"]["national"] == 0.5
    assert doc["renewal"]["bvd"]["national_change"] == "-50.0%"
    assert doc["renewal"]["pooling"]["zones"] == 0.5
    table = (tmp_path / "si" / "renewal-variants.md").read_text()
    assert "Zones" in table
