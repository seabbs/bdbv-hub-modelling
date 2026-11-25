# nowcastgermany

The German COVID-19 hospitalisation nowcasting hub as a case study: its reporting triangle as forecast tasks, the benchmark and its page.
{ .page-subtitle }

nowcastgermany is the package in `packages/nowcastgermany` of the repository ([Packages](https://samabbott.co.uk/bdbv-hub-modelling/packages/)).

## Install

```sh
uv pip install "git+https://github.com/seabbs/bdbv-hub-modelling#subdirectory=packages/nowcastgermany"
```

## Getting started

The hub's reporting triangle gives each Meldedatum's count as first reported (`value_0d`) and the change reported on each later day.
We make a small one for two states, Germany as their sum, and two of Germany's age groups.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(1)
days = pd.date_range("2021-09-01", "2021-11-28").date.astype(str)
changes = ["value_0d", "value_1d", "value_2d"]
rows = []
for state, size in (("DE-BY", 40), ("DE-BE", 20)):
    trend = size * np.exp(np.cumsum(rng.normal(0, 0.1, len(days))))
    for day, mean in zip(days, trend):
        counts = rng.poisson((mean / 2, mean / 4, mean / 4))
        rows.append((day, state, "00+", *counts, 0))
wide = pd.DataFrame(
    rows, columns=["date", "location", "age_group", *changes, "value_>80d"]
)
germany = wide.groupby("date", as_index=False).sum(numeric_only=True)
germany = germany.assign(location="DE", age_group="00+")
ages = [
    germany.assign(age_group=age, **{c: germany[c] // 2 for c in changes})
    for age in ("35-59", "60-79")
]
wide = pd.concat([wide, germany, *ages], ignore_index=True)[wide.columns]
wide.to_csv("COVID-19_hospitalizations.csv", index=False)
```

`hub_triangle` reads it as one row per Meldedatum, place and report date, with the count reported by then.

```python
from nowcastgermany.data import hub_triangle

triangle = hub_triangle("COVID-19_hospitalizations.csv")
assert list(triangle.columns) == [
    "reference_date",
    "location",
    "age_group",
    "confirm",
    "report_date",
]
```

`nowcast_tasks` makes the two tasks of a nowcast date: Germany with its states, and Germany with its age groups.
Their target dates are the ends of the hub's 7-day sums, from the nowcast date to 28 days before it.

```python
from nowcastgermany.data import nowcast_tasks

states, age_groups = nowcast_tasks(
    triangle, "2021-11-21", n_samples=100, seed=1
)
assert states.locations == ("CD-BE", "CD-BY")
assert age_groups.locations == ("CD-A3", "CD-A4")
assert len(states.target_end_dates) == 29
```

Next: [How to](how-to.md).
