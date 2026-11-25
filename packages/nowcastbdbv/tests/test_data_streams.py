"""The hub files the data step fetches."""

from nowcastbdbv.data.tidy import DATA_STREAMS


def test_the_missing_onset_counts_by_report_date_are_fetched() -> None:
    assert "auxiliary-data/daily-missing-symptom-by-report-date.csv" in (
        DATA_STREAMS
    )
