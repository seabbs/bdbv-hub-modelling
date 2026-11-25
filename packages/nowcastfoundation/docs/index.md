# nowcastfoundation

Chronos-2 nowcasts and forecasts from reporting triangles, the simulation grid and fine-tuning on simulated outbreaks.
{ .page-subtitle }

nowcastfoundation is the package in `packages/nowcastfoundation` of the repository ([Packages](https://samabbott.co.uk/bdbv-hub-modelling/packages/)).

## Install

```sh
uv pip install "git+https://github.com/seabbs/bdbv-hub-modelling#subdirectory=packages/nowcastfoundation"
```

Forecasting with Chronos-2 needs torch, the `chronos` dependency group (`uv sync --group chronos` in a clone).
The examples on this site do not.

## Getting started

Chronos-2 sees the hub target as a reporting triangle: each place's count at each step, as each later release reported it.
We make daily releases for three weeks of one place, `A`.
A day's count starts at 1 and reaches 10 three days after it.

```python
import pandas as pd

rows = []
for vintage in pd.date_range("2026-03-01", "2026-03-21"):
    for day in pd.date_range("2026-02-01", vintage):
        age = (vintage - day).days
        observed = 10 if age >= 3 else 3 * age + 1
        rows.append(
            (
                vintage.date().isoformat(),
                "A",
                day.date().isoformat(),
                observed,
                0,
            )
        )
releases = pd.DataFrame(
    rows, columns=["vintage", "location", "onset_date", "observed", "imputed"]
)
releases["cutoff"] = releases["vintage"]
```

`hub_triangle` lays the releases out by place, day and release lag.
`final` holds the latest release's counts and `partial` the count at each lag, missing where no release has reported it yet.

```python
from nowcastfoundation.forecaster import hub_triangle

tri = hub_triangle(releases, "daily", ["A"], "2026-03-21", max_lag=7)
assert tri.final.shape == (1, 49)
assert tri.partial.shape == (1, 49, 7)
# The last day has only been reported at lag 0.
assert tri.partial[0, -1, 0] == 1
assert tri.partial[0, -3, :3].tolist() == [1, 4, 7]
```

`lag_count` reads how many lags a forecast needs from the data's own reporting delay.
It takes the triangle with every lag of its release history.

```python
from nowcastfoundation.forecaster import lag_count

full = hub_triangle(releases, "daily", ["A"], "2026-03-21", all_lags=True)
assert lag_count(full, "daily") == 4
```

The forecasters in `nowcastfoundation.forecaster` give these triangles to Chronos-2, zero-shot or with a fine-tuned checkpoint.

Next: [How to](how-to.md).
