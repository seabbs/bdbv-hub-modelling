def test_release_quantiles_take_the_explorer_s_columns():
    # The release backtest names its 5% quantile `q5`; the explorer reads
    # `q05` to `q95`.
    import pandas as pd

    from bdbvhub.ftcheck import explorer_quantiles
    from bdbvhub.reporting.explorer import QUANTILE_COLUMNS

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
