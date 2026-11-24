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
- [`zones`](src/nowcastbdbv/zones.py): health-zone forecasts split from province forecasts, and the zone backtest and its scores.
- [`data-raw/`](data-raw): how the package's data files are built.

It imports the generic packages: `nowcastutils`, `nowcasteval` and `nowcastbaselines`, and `nowcastrenewal` for the BVD download helper.
More of the BDBV study moves here from `bdbvhub` (BVD inputs, reports), so that the generic packages import no outbreak.

```sh
uv add nowcastbdbv
```
