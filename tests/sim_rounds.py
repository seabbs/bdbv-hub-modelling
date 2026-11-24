"""Simulated round files shared by the trainer and sim-QC tests."""


def sim_dir(tmp_path):
    """Write two seeds' counts and settings under `tmp_path`.

    Returns the simulations' ids.
    """
    import pandas as pd

    rows, settings = [], []
    for seed in (6001, 6002):
        for sim in (1, 2):
            simulation = f"{seed}-{sim:04d}"
            settings.append((simulation, 2, 1, 28))
            for draw in (1, 2):
                for place, scale in (("P01", 1), ("P02", draw + sim)):
                    for onset in pd.date_range("2026-01-03", periods=120):
                        for d in (0, 2, 6):
                            report = onset + pd.Timedelta(days=d)
                            rows.append(
                                (
                                    f"{simulation}-{draw}",
                                    place,
                                    onset.date().isoformat(),
                                    report.date().isoformat(),
                                    scale,
                                )
                            )
    counts = pd.DataFrame(
        rows,
        columns=[
            "outbreak",
            "location",
            "reference_date",
            "report_date",
            "count",
        ],
    )
    table = pd.DataFrame(
        settings, columns=["outbreak", "n_places", "n_ages", "max_delay"]
    )
    for seed in (6001, 6002):
        part = counts["outbreak"].str.startswith(str(seed))
        counts[part].to_csv(tmp_path / f"counts-{seed}.csv.gz", index=False)
        keep = table["outbreak"].str.startswith(str(seed))
        table[keep].to_csv(tmp_path / f"settings-{seed}.csv", index=False)
    return [o for o, *_ in settings]
