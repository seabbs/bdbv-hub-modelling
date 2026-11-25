# nowcastbdbv

The 2026 Bundibugyo virus disease (BDBV) outbreak in the DRC as a case study: the hub's data streams, health zones and onset-to-report delays.
{ .page-subtitle }

nowcastbdbv is the package in `packages/nowcastbdbv` of the repository ([Packages](https://samabbott.co.uk/bdbv-hub-modelling/packages/)).

## Install

```sh
uv pip install "git+https://github.com/seabbs/bdbv-hub-modelling#subdirectory=packages/nowcastbdbv"
```

## Getting started

The hub publishes weekly confirmed cases by onset week, one block per release (`vintage`).
`tidy_target` turns its files into a target table with the cases split into those with a known onset (`observed`) and an imputed one (`imputed`).
`load_target` reads that table from `data/` after `uv run bdbvhub data`.
Here we make a small target table by hand for one province and two releases a week apart.

```python
import pandas as pd

weeks = pd.date_range("2026-09-05", periods=4, freq="7D").date.astype(str)
rows = []
for vintage, totals in (
    ("2026-09-26", (6, 9, 12, 5)),
    ("2026-10-03", (6, 10, 15, 11)),
):
    for week, total in zip(weeks, totals):
        imputed = total // 3
        rows.append(
            ("real", vintage, vintage, "CD-NK", week, total - imputed, imputed, total)
        )
target = pd.DataFrame(
    rows,
    columns=[
        "kind", "vintage", "cutoff", "location", "target_end_date",
        "observed", "imputed", "total",
    ],
)
```

`vintages` lists the releases in a table, and `imputed_share` gives the share of each week's cases with an imputed onset.

```python
from nowcastbdbv.data.tidy import imputed_share, vintages

assert vintages(target) == ["2026-09-26", "2026-10-03"]
share = imputed_share(target)
assert share["imputed_share"].between(0, 1).all()
```

Next: [How to](how-to.md).
