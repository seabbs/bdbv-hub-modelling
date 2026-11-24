# nowcastutils

The pieces the other nowcasting packages build on: releases and their cut-offs, daily snapshots, location hierarchies, the forecast task and sample trajectories.
It is for anyone writing a model or a scorer that should see the same releases and places as the rest of the repository.

## Start here

- Read [`base`](src/nowcastutils/base.py) first. Every model takes its `ForecastTask`, returns a sample table, and is found by name in its registry.
- The [docs site](https://samabbott.co.uk/bdbv-hub-modelling/nowcastutils/) has a worked example, how-to recipes and the API reference.

## Modules

- [`base`](src/nowcastutils/base.py): the forecast task, the sample table and the model registry.
- [`releases`](src/nowcastutils/releases.py): release cut-offs, cut-off lags and release pairs.
- [`snapshots`](src/nowcastutils/snapshots.py): daily release snapshots as arrays on the onset clock.
- [`hierarchy`](src/nowcastutils/hierarchy.py): location hierarchies read from ISO 3166-2 style codes.
- [`dates`](src/nowcastutils/dates.py): epidemiological weeks and dates.
- [`forecast`](src/nowcastutils/forecast.py): a model's forecast record.
- [`trajectories`](src/nowcastutils/trajectories/): coherent sample trajectories from quantile forecasts.
- [`files`](src/nowcastutils/files.py), [`web`](src/nowcastutils/web.py): table writing and downloads.
- [`devices`](src/nowcastutils/devices.py): compute device settings.

## Next

- [How it is used in the workflow](https://samabbott.co.uk/bdbv-hub-modelling/nowcastutils/workflow/).
- The packages built on it: [nowcasteval](../nowcasteval/README.md) and [nowcastbaselines](../nowcastbaselines/README.md).

## Install

From a clone of the repository:

```sh
uv sync --package nowcastutils             # this package and the ones it needs
```

## Example

```python
from nowcastutils.hierarchy import depth, parent

parent("A-1")  # "A"
depth("A-1-07")  # 2
```

More recipes: [how-to](https://samabbott.co.uk/bdbv-hub-modelling/nowcastutils/how-to/).

## Tasks

`task nowcastutils:test` runs its tests; `task --list-all` lists every task, `task nowcastutils:<task>` from the root or `task <task>` in this folder.

## Citation

Cite it with [CITATION.cff](CITATION.cff); the repository as a whole has [its own](../../CITATION.cff).

```bibtex
@software{abbott_nowcastutils,
  author = {Abbott, Sam},
  title = {{nowcastutils: shared pieces for nowcasting and forecasting: release dates, snapshot arrays, location hierarchies, the forecast task and model registry, and quantile-to-trajectory coupling}},
  year = {2026},
  url = {https://github.com/seabbs/bdbv-hub-modelling/tree/main/packages/nowcastutils}
}
```
