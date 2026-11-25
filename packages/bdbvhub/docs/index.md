# bdbvhub

Our side of the BDBV Modeling Hub: the model registry, the fit store, the weekly hub file and its checks, and the hub pages of the docs site.
{ .page-subtitle }

bdbvhub is the package in `packages/bdbvhub` of the repository ([Packages](https://samabbott.co.uk/bdbv-hub-modelling/packages/)).
It sits at the top of the package stack and imports every other `nowcast*` package.
Its `bdbvhub` command is what the CI workflows call.

## Install

```sh
uv pip install "git+https://github.com/seabbs/bdbv-hub-modelling#subdirectory=packages/bdbvhub"
```

From a clone of the repository, `uv sync --package bdbvhub` installs it with the packages it needs.

## Getting started

A model returns sample trajectories: one row per province, week and sample.
We make three samples of two weeks for two provinces.

```python
import pandas as pd

samples = pd.DataFrame(
    [
        (location, week, sample, value + sample)
        for location, value in (("CD-KN", 4), ("CD-NK", 10))
        for week in ("2026-03-14", "2026-03-21")
        for sample in (1, 2, 3)
    ],
    columns=["location", "target_end_date", "sample", "value"],
)
```

`to_hub_table` lays them out as the hub's submission table.
It checks that every place has the same sample ids and weeks, and adds the country (`CD`) as the sum of its provinces, sample by sample.

```python
from bdbvhub.submission.hub import HUB_COLUMNS, to_hub_table

table = to_hub_table(
    samples, "2026-03-14", n_samples=3, forecast_date="2026-03-14"
)
assert tuple(table.columns) == HUB_COLUMNS
national = table.query("location == 'CD'")
assert national["value"].tolist() == [16, 16, 18, 18, 20, 20]
```

A hub model id joins our team abbreviation and the model's, and the file goes under `model-output/<model id>/`.

```python
import tempfile

from bdbvhub.submission.hub import model_id, write_submission

mid = model_id("rwnowcast")
assert mid == "seabbs-rwnowcast"
out = tempfile.mkdtemp()
path = write_submission(table, out, mid)
assert path.name == "2026-03-14-seabbs-rwnowcast.csv"
```

Next: [How to](how-to.md).
