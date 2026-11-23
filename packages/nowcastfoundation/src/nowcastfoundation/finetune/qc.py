"""Record one round of simulations' summaries.

Summarises each simulation of `ROUND_DIR/sims` (`summaries`): cases,
strata with cases, the largest week-on-week rise in weekly cases by
reference date and the reporting delay. The grid sets no thresholds, so
no simulation is excluded.

Records the table and a one-line summary in `ROUND_DIR/qc/`.
Simulations are never deleted. The workflow's `sim_qc` rule runs it:

    uv run sim-qc ROUND_DIR
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nowcastfoundation.finetune import delay_quantile, simulations

CELL = [
    "strata",
    "n_places",
    "n_ages",
    "spread",
    "trajectory",
    "delay_family",
    "delay_form",
    "delay_change",
    "report_days",
    "shocks",
    "mixing",
    "imports",
    "seeding",
    "observation",
    "obs_strata",
    "delay_places",
    "strata_link",
]


def summaries(sim_dir: Path) -> pd.DataFrame:
    """Return one row per simulation: its cell and count summaries."""
    settings = pd.concat(
        pd.read_csv(p) for p in sorted(sim_dir.glob("settings-*.csv"))
    ).set_index("outbreak")
    rows = []
    for path in sorted(sim_dir.glob("counts-*.csv.gz")):
        counts = pd.read_csv(path)
        counts["simulation"] = simulations(counts["outbreak"])
        for simulation, c in counts.groupby("simulation"):
            ref = pd.to_datetime(c["reference_date"])
            delay = (pd.to_datetime(c["report_date"]) - ref).dt.days
            w = c["count"].to_numpy(float)
            per_draw = c.groupby("outbreak")["count"].sum()
            week = ref.dt.to_period("W")
            weekly = (
                c.assign(week=week).groupby(["outbreak", "week"])["count"].sum()
            )
            first = weekly.loc[per_draw.index[0]]
            growth = np.diff(np.log(first.to_numpy(float) + 1))
            s: Any = (
                settings.loc[simulation] if simulation in settings.index else {}
            )
            rows.append(
                {
                    "simulation": simulation,
                    **{
                        k: s.get(k) for k in CELL if k in getattr(s, "index", s)
                    },
                    "draws": per_draw.size,
                    "cases_median": float(per_draw.median()),
                    "cases_min": float(per_draw.min()),
                    "cases_max": float(per_draw.max()),
                    "strata_with_cases": c["location"].nunique(),
                    "weeks": int(first.size),
                    "peak_weekly": float(first.max()),
                    "max_weekly_log_growth": float(growth.max())
                    if growth.size
                    else np.nan,
                    "delay_mean_days": float((delay * w).sum() / w.sum()),
                    "delay_q99_days": delay_quantile(
                        c["count"].groupby(delay).sum(), 0.99
                    ),
                    "reported_within_7d": float(
                        w[delay.to_numpy() <= 7].sum() / w.sum()
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    """Check the round and record everything in its qc/ directory."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("round_dir", type=Path)
    args = parser.parse_args()
    qc = args.round_dir / "qc"
    qc.mkdir(parents=True, exist_ok=True)
    table = summaries(args.round_dir / "sims")
    table.to_csv(qc / "input.csv", index=False)
    (qc / "summary.txt").write_text(
        f"sim QC summarised {len(table)} simulations"
    )
    print((qc / "summary.txt").read_text())


if __name__ == "__main__":
    main()
