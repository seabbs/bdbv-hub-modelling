# bdbvhub

Nowcasts and forecasts for the [BDBV Modeling Hub](https://github.com/InsightNet-US/BDBV-Modeling-Hub), built on the `nowcast*` packages.
It reads the hub's data, fits and stores the models, writes and checks the weekly hub file, and builds the docs site's hub pages.

## Start here

- The `bdbvhub` command: `uv run bdbvhub --help` lists every step, from `data` (download and tidy) to `nowcast` (write a hub file). It is also installed as `bdbvhubrenewal`, the name CI calls.
- To see which models exist, read [`models`](src/bdbvhub/models.py); importing it registers every model.

## Modules

- [`data`](src/bdbvhub/data/): download, tidy and load the hub's data streams, its delays and health zones.
- [`models`](src/bdbvhub/models.py): the model registry.
- [`store`](src/bdbvhub/store/): the fit store, its archive and the backtest digest.
- [`compare`](src/bdbvhub/compare.py), [`zones`](src/bdbvhub/zones.py): comparison with the hub's forecasts, and health-zone forecasts.
- [`submission`](src/bdbvhub/submission/): the hub file writer and checks, model metadata, the rounds and the CI fit configurations.
- [`reporting`](src/bdbvhub/reporting/): the docs pages, tables and explorer.
- [`cli`](src/bdbvhub/cli.py): the command.

## Next

- How a week's submission runs: [Submission workflow](https://samabbott.co.uk/bdbv-hub-modelling/submission/workflow/).
- The command's reference: [CLI](https://samabbott.co.uk/bdbv-hub-modelling/development/cli/).
- Adding a model: [Adding a model](https://samabbott.co.uk/bdbv-hub-modelling/development/adding-a-model/).

```sh
uv add bdbvhub
```
