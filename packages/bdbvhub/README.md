# bdbvhub

Our side of the [BDBV Modeling Hub](https://github.com/InsightNet-US/BDBV-Modeling-Hub), built on the `nowcast*` packages.
It fits our models to the hub's data, keeps the fits in a store, writes and checks the weekly hub file, and builds the hub pages of the docs site.

In this project it is the `bdbvhub` command the CI workflows call.
Fit store fits and stores every fitted model (`bdbvhub fit-store`).
Site runs the backtests and builds the docs pages (`bdbvhub backtest`, `bdbvhub report`).
Fit candidates writes the week's hub file (`bdbvhub nowcast`) for Submit to the BDBV hub.
It sits at the top of the package stack and imports every other `nowcast*` package.

## Start here

- The `bdbvhub` command: `uv run bdbvhub --help` lists every step, from `data` (download and tidy) to `nowcast` (write a hub file).
- To see which models exist, read [`models`](src/bdbvhub/models.py); importing it registers every model.

## Modules

- The hub's data streams, delays and health zones are in [`nowcastbdbv.data`](../nowcastbdbv/src/nowcastbdbv/data/).
- [`models`](src/bdbvhub/models.py): the model registry.
- [`store`](src/bdbvhub/store/): the fit store, its archive and the backtest digest.
- [`pairing`](src/bdbvhub/pairing.py): the walk models' pairing sensitivity on the BDBV backtest.
- [`compare`](src/bdbvhub/compare.py): comparison with the hub's forecasts. Health-zone forecasts are in [`nowcastbdbv.zones`](../nowcastbdbv/src/nowcastbdbv/zones.py).
- [`ftcheck`](src/bdbvhub/ftcheck.py): the Chronos-2 models, zero-shot and fine-tuned, on every hub release, scored.
- [`submission`](src/bdbvhub/submission/): the hub file writer and checks, model metadata, the rounds and the CI fit configurations.
- [`reporting`](src/bdbvhub/reporting/): the docs pages, tables and figures of the hub data.
- [`cli`](src/bdbvhub/cli.py): the command.

## Next

- How a week's submission runs: [Submission workflow](https://samabbott.co.uk/bdbv-hub-modelling/submission/workflow/).
- The command's reference: [CLI](https://samabbott.co.uk/bdbv-hub-modelling/development/cli/).
- Adding a model: [Adding a model](https://samabbott.co.uk/bdbv-hub-modelling/development/adding-a-model/).

## Install

From a clone of the repository:

```sh
uv sync --package bdbvhub             # this package and the ones it needs
```

## Example

```sh
uv run bdbvhub data        # download and tidy the hub's data
task bdbvhub:nowcast       # forecast from the latest release, a checked hub file
uv run bdbvhub --help      # every step
```

How a week's submission runs: [Submission workflow](https://samabbott.co.uk/bdbv-hub-modelling/submission/workflow/).

## Tasks

`task bdbvhub:test` runs its tests (`task --list-all` lists the rest).

## Citation

Cite it with [CITATION.cff](CITATION.cff); the repository as a whole has [its own](../../CITATION.cff).

```bibtex
@software{abbott_bdbvhub,
  author = {Abbott, Sam},
  title = {{bdbvhub: nowcasts and forecasts for the BDBV Modeling Hub: hub data, the model registry, the fit store, submission and reports}},
  year = {2026},
  url = {https://github.com/seabbs/bdbv-hub-modelling/tree/main/packages/bdbvhub}
}
```
