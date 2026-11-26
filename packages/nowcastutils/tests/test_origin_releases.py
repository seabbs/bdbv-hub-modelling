"""Forecasting starts at the first release with two usable releases."""

import pandas as pd

from nowcastutils.releases import MIN_ORIGIN_RELEASES, origin_releases


def _rows(cutoffs: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {"vintage": cutoffs, "cutoff": cutoffs, "location": "CD"}
    )


GAP = ["2026-08-18", "2026-10-07", "2026-10-14", "2026-10-21"]
WEEKLY = ["2026-09-30", "2026-10-07", "2026-10-14"]


def test_origins_start_at_the_second_release() -> None:
    assert MIN_ORIGIN_RELEASES == 2
    assert origin_releases(_rows(WEEKLY), 42) == WEEKLY[1:]


def test_a_cumulative_first_release_does_not_count() -> None:
    # The first release is a dump, so the next is the only usable one.
    assert origin_releases(_rows(GAP), 42) == GAP[2:]
    # Within D of the next release it counts.
    assert origin_releases(_rows(GAP), 60) == GAP[1:]
    assert origin_releases(_rows(GAP), None) == GAP[1:]


def test_too_few_releases_give_no_origin() -> None:
    assert origin_releases(_rows(GAP[:2]), 42) == []
    assert origin_releases(_rows(WEEKLY[:1]), 42) == []
    assert origin_releases(_rows([]), 42) == []


def test_origins_follow_the_cut_offs_and_count_each_release_once() -> None:
    rows = pd.concat([_rows(WEEKLY[::-1]), _rows(WEEKLY)])
    assert origin_releases(rows, 42) == WEEKLY[1:]
    assert origin_releases(rows, 42, min_releases=3) == WEEKLY[2:]
