# nowcastbaselines

The random walk, the random walk nowcast and a JAX port of baselinenowcast.
Use it for a simple, well-tested forecast of weekly counts that are revised as later releases report more cases.

In this project they are the baselines every other model is scored against.
The random walk nowcast is submitted to the hub each week as `seabbs-rwnowcast`.
The random walks are also the last fallbacks of the renewal track ([`policy.toml`](../bdbvsubmission/policy.toml)).
The German benchmark and the paper's studies score relative to `randomwalknowcast`.

## Start here

- Read [`randomwalk`](src/nowcastbaselines/randomwalk.py) (`RandomWalk`, `RandomWalkNowcast`).
- The [docs site](https://samabbott.co.uk/bdbv-hub-modelling/nowcastbaselines/) has a worked example, how-to recipes and the API reference.

## Modules

- [`randomwalk`](src/nowcastbaselines/randomwalk.py): the random walk forecaster and its nowcasting variant.
- [`baselinenowcast`](src/nowcastbaselines/baselinenowcast/): a JAX port of the baselinenowcast R package, with the chain ladder on release snapshots, in `delays`, `triangle`, `aggregate`, `uncertainty` and `nowcast`.
- [`walk`](src/nowcastbaselines/walk.py): random walks on the log scale.
- [`pairing`](src/nowcastbaselines/pairing.py): the walk models' forecasts under each pairing of their draws, for the pairing sensitivity.
- [`pooling`](src/nowcastbaselines/pooling.py), [`introductions`](src/nowcastbaselines/introductions.py), [`copula`](src/nowcastbaselines/copula.py), [`zones`](src/nowcastbaselines/zones.py): pooling over a hierarchy, introductions into silent places, a copula pairing places, and zone-level forecasts.

## Next

- [How it is used in the workflow](https://samabbott.co.uk/bdbv-hub-modelling/nowcastbaselines/workflow/).
- The baselines in the hub submission: [Baseline models](https://samabbott.co.uk/bdbv-hub-modelling/methods/baselines/).

## Install

From a clone of the repository:

```sh
uv sync --package nowcastbaselines             # this package and the ones it needs
```

## Example

```python
import nowcastbaselines.randomwalk  # registers the walks
from nowcastutils.base import get_model

model = get_model("randomwalknowcast")
forecasts = model.forecast(task)  # a nowcastutils.base.ForecastTask
```

More recipes: [how-to](https://samabbott.co.uk/bdbv-hub-modelling/nowcastbaselines/how-to/).

## Tasks

`task nowcastbaselines:test` runs its tests (`task --list-all` lists the rest).

## Citation

Cite it with [CITATION.cff](CITATION.cff); the repository as a whole has [its own](../../CITATION.cff).

```bibtex
@software{abbott_nowcastbaselines,
  author = {Abbott, Sam},
  title = {{nowcastbaselines: baseline nowcasts and forecasts: random walks on the log scale, a JAX port of baselinenowcast, introductions, and a copula pairing places}},
  year = {2026},
  url = {https://github.com/seabbs/bdbv-hub-modelling/tree/main/packages/nowcastbaselines}
}
```
