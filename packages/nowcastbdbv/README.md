# nowcastbdbv

The 2026 Bundibugyo virus disease (BDBV) outbreak in the DRC as a case study.
It downloads, tidies and loads the BDBV Modeling Hub's data streams, its health zones, and the onset-to-report delays.

## Start here

- To read how the hub's data are loaded, start with [`data.tidy`](src/nowcastbdbv/data/tidy.py).

## Modules

- [`data.tidy`](src/nowcastbdbv/data/tidy.py): download the hub's data streams, and tidy and load them.
- [`data.delays`](src/nowcastbdbv/data/delays.py): empirical onset-to-report delays.
- [`data.zones`](src/nowcastbdbv/data/zones.py): the hub's health-zone codes and weekly zone counts.
- [`data.zone_map`](src/nowcastbdbv/data/zone_map.py): the health-zone map.
- [`backtest`](src/nowcastbdbv/backtest.py): the backtest's tasks from the hub's releases (hub delay and zones), and the development and leave-one-province-out runs.
- [`zones`](src/nowcastbdbv/zones.py): health-zone forecasts split from province forecasts, and the zone backtest and its scores.
- [`data-raw/`](data-raw): how the package's data files are built.

It imports the generic packages: `nowcastutils`, `nowcasteval` and `nowcastbaselines`, and `nowcastrenewal` for the BVD download helper.
More of the BDBV study moves here from `bdbvhub` (BVD inputs, reports), so that the generic packages import no outbreak.

## Install

From a clone of the repository:

```sh
uv sync --package nowcastbdbv             # this package and the ones it needs
```

## Example

```python
from nowcastbdbv.data.tidy import load_target

target = load_target()  # data/ after `uv run bdbvhub data`
```

The data: [Data](https://samabbott.co.uk/bdbv-hub-modelling/data/).

## Tasks

`task nowcastbdbv:test` runs its tests; `task --list-all` lists every task, `task nowcastbdbv:<task>` from the root or `task <task>` in this folder.

## Citation

Cite it with [CITATION.cff](CITATION.cff); the repository as a whole has [its own](../../CITATION.cff).

```bibtex
@software{abbott_nowcastbdbv,
  author = {Abbott, Sam},
  title = {{nowcastbdbv: the BDBV outbreak as a case study: the hub's data streams, health zones and onset-to-report delays}},
  year = {2026},
  url = {https://github.com/seabbs/bdbv-hub-modelling/tree/main/packages/nowcastbdbv}
}
```
