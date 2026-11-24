# bdbvhub

Nowcasts and forecasts for the [BDBV Modeling Hub](https://github.com/InsightNet-US/BDBV-Modeling-Hub), built on the `nowcast*` packages.
It reads the hub's data (`nowcastbdbv`), fits and stores the models, writes and checks the weekly hub file, and builds the docs site's hub pages.

## Start here

- The `bdbvhub` command: `uv run bdbvhub --help` lists every step, from `data` (download and tidy) to `nowcast` (write a hub file).
- To see which models exist, read [`models`](src/bdbvhub/models.py); importing it registers every model.

## Modules

- The hub's data streams, delays and health zones are in [`nowcastbdbv.data`](../nowcastbdbv/src/nowcastbdbv/data/).
- [`models`](src/bdbvhub/models.py): the model registry.
- [`store`](src/bdbvhub/store/): the fit store, its archive and the backtest digest.
- [`pairing`](src/bdbvhub/pairing.py): the walk models' pairing sensitivity on the BDBV backtest.
- [`compare`](src/bdbvhub/compare.py): comparison with the hub's forecasts. Health-zone forecasts are in [`nowcastbdbv.zones`](../nowcastbdbv/src/nowcastbdbv/zones.py).
- [`submission`](src/bdbvhub/submission/): the hub file writer and checks, model metadata, the rounds and the CI fit configurations.
- [`reporting`](src/bdbvhub/reporting/): the docs pages, tables and figures of the hub data.
- [`cli`](src/bdbvhub/cli.py): the command.

## Next

- How a week's submission runs: [Submission workflow](https://samabbott.co.uk/bdbv-hub-modelling/submission/workflow/).
- The command's reference: [CLI](https://samabbott.co.uk/bdbv-hub-modelling/development/cli/).
- Adding a model: [Adding a model](https://samabbott.co.uk/bdbv-hub-modelling/development/adding-a-model/).

```sh
uv add bdbvhub
```
