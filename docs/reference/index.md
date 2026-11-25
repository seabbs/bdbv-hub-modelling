# API reference

The Python API of `bdbvhub`, grouped by task. See [Setup](../setup.md) to install it and [Getting started](../getting-started.ipynb) for a worked example.
{ .page-subtitle }

| Task | Pages | Start with |
|---|---|---|
| Data | [nowcastbdbv](../nowcastbdbv/reference/), [nowcastgermany](../nowcastgermany/reference/) | `load_target`, `load_daily`, `load_delays`, `delays_for_vintage` |
| Models | [nowcastbaselines](../nowcastbaselines/reference/), [nowcastfoundation](../nowcastfoundation/reference/), [nowcastrenewal](../nowcastrenewal/reference/), [bdbvhub](../bdbvhub/reference/) | `get_model`, `list_models`, `ForecastTask`, `ForecastModel.forecast` |
| Evaluation | [nowcasteval](../nowcasteval/reference/), [bdbvhub](../bdbvhub/reference/) | `run_forecasts`, `score_forecasts`, `release_backtest`; the BDBV tasks are `nowcastbdbv.backtest.make_tasks` |
| Reporting | [nowcasteval](../nowcasteval/reference/), [bdbvhub](../bdbvhub/reference/) | `plot_location_grid`, `plot_forecasts`, `to_hub_table`, `write_submission` |
| CLI | [bdbvhub](../bdbvhub/reference/) | `bdbvhub --help` |

The task, the registry, releases and snapshots come from the [nowcastutils](../nowcastutils/reference/) package.
Scores, backtests and score figures are in [nowcasteval](../nowcasteval/reference/), and the random walk baselines in [nowcastbaselines](../nowcastbaselines/reference/).

## A forecast in five steps

Run from the repository root after `bdbvhub data` has written the tidy tables to `data/`.

```python
import dataclasses

from nowcastbdbv.backtest import make_tasks
from nowcastbdbv.data.tidy import (
    delays_for_vintage,
    load_daily,
    load_delays,
    load_target,
)
from nowcasteval import backtest as bt
from nowcasteval.nowcast import release_backtest
from bdbvhub.models import get_model
from nowcasteval.plots import plot_location_grid

# 1. Load the data.
target, daily, delays = load_target(), load_daily(), load_delays()
real = target.loc[target["kind"] == "real"]

# 2. Build the task a model sees at one release.
config = dataclasses.replace(
    bt.load_config("packages/nowcastbdbv/config/backtest.toml"), min_horizon=-6
)
vintage = "2026-10-17"
task = make_tasks(
    real.loc[real["vintage"] == vintage],
    delays_for_vintage(delays, vintage)[0],
    config,
    history=real,
    daily=daily,
)[0]

# 3. Forecast with models from the registry.
models = [get_model("randomwalk"), get_model("randomwalknowcast")]
forecasts = bt.run_forecasts(models, [task])

# 4. Score against later releases.
scores = release_backtest(
    [task], daily, target, models=(), stored=forecasts
).scores

# 5. Plot.
fig = plot_location_grid(forecasts, real, ["CD"])
```
