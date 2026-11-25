# bdbvsubmission

The gated weekly submission to the BDBV Modeling Hub, from fitting candidates to the hub pull request.
{ .page-subtitle }

bdbvsubmission is the package in `packages/bdbvsubmission` of the repository ([Packages](https://samabbott.co.uk/bdbv-hub-modelling/packages/)).
It checks each candidate forecast, falls back along a chain of models when one fails, and records what it submitted.
It uses the standard library only.

## Install

It has its own environment and lock, so the submission runs with exactly what it needs.
It is not in the root workspace, so `uv sync` at the root does not install it.

```sh
uv pip install "git+https://github.com/seabbs/bdbv-hub-modelling#subdirectory=packages/bdbvsubmission"
```

## Getting started

Each round has a reference date, a Saturday.
Its submission window opens six days before, at midnight in Kinshasa, and closes three days before.

```python
import datetime as dt

from bdbvsubmission.window import KINSHASA, window_for

win = window_for(dt.date(2026, 11, 21))
assert win.opens == dt.datetime(2026, 11, 15, tzinfo=KINSHASA)
assert win.closes == dt.datetime(2026, 11, 18, tzinfo=KINSHASA)
assert win.is_open(dt.datetime(2026, 11, 16, 12, tzinfo=KINSHASA))
```

A candidate's hub file holds sample trajectories.
We write a small one with two samples for one province and two weeks.
One value is negative.

```python
import csv
import tempfile
from pathlib import Path

from bdbvsubmission.hubdata import TARGET, read_samples

path = Path(tempfile.mkdtemp()) / "candidate.csv"
with path.open("w", newline="") as fh:
    out = csv.writer(fh)
    out.writerow(
        [
            "reference_date",
            "forecast_date",
            "target",
            "target_end_date",
            "location",
            "output_type",
            "output_type_id",
            "value",
        ]
    )
    for sample, values in (("1", (4, 6)), ("2", (5, -1))):
        for week, value in zip(("2026-11-14", "2026-11-21"), values):
            out.writerow(
                [
                    "2026-11-21",
                    "2026-11-17",
                    TARGET,
                    week,
                    "CD-NK",
                    "sample",
                    sample,
                    value,
                ]
            )
samples = read_samples(path)
assert samples.locations == ["CD-NK"]
assert samples.week_values("CD-NK", "2026-11-21") == [6.0, -1.0]
```

Each check returns `pass`, `warn`, `fail` or `skip`.
The negative value fails `check_values`, and a failure blocks the candidate.

```python
from bdbvsubmission.check_forecast import check_samples, check_values
from bdbvsubmission.common import blocks, worst

checks = [check_values(samples), check_samples(samples, n=2)]
assert [c.status for c in checks] == ["fail", "pass"]
assert checks[0].locations == ["CD-NK"]
assert blocks(worst(c.status for c in checks))
```

The failure names one province.
The gate then keeps the candidate and takes that province from the next candidate in the chain (`bdbvsubmission.location_fallback`).

Next: [How to](how-to.md).
