# nowcasteval

Scoring of sample and quantile nowcasts and forecasts, and backtests that compare models with a baseline.
It is for anyone scoring forecasts of counts that are revised as later releases report more cases.

## Start here

- To score forecasts, read [`scoring`](src/nowcasteval/scoring.py) (`score_forecasts`).
- To score a model over past releases, read [`backtest`](src/nowcasteval/backtest.py).
- The [docs site](https://samabbott.co.uk/bdbv-hub-modelling/nowcasteval/) has a worked example, how-to recipes and the API reference.

## Modules

- [`scoring`](src/nowcasteval/scoring.py): sample CRPS, WIS and its components, and the energy score, by level.
- [`backtest`](src/nowcasteval/backtest.py): forecasts over a run of releases, scored against later data.
- [`nowcast`](src/nowcasteval/nowcast.py): nowcast tasks and their truth.
- [`coupling`](src/nowcasteval/coupling.py): trajectories coupled across places and scored on what was reported.
- [`marginals`](src/nowcasteval/marginals.py), [`ppc`](src/nowcasteval/ppc.py), [`checks`](src/nowcasteval/checks.py): marginal summaries, predictive checks and model checks.
- [`pairing`](src/nowcasteval/pairing.py): scores of the walk models' draws under each pairing.
- [`labels`](src/nowcasteval/labels.py), [`plots`](src/nowcasteval/plots.py), [`tables`](src/nowcasteval/tables.py), [`scorechart`](src/nowcasteval/scorechart.py), [`scoresummary`](src/nowcasteval/scoresummary.py), [`nowcast_plots`](src/nowcasteval/nowcast_plots.py), [`pairing_plots`](src/nowcasteval/pairing_plots.py), [`explorer`](src/nowcasteval/explorer.py): display names, figures, tables of scores and the forecast explorer's data.

## Next

- [How it is used in the workflow](https://samabbott.co.uk/bdbv-hub-modelling/nowcasteval/workflow/).
- The scores in use: [Evaluation](https://samabbott.co.uk/bdbv-hub-modelling/methods/evaluation/) and [Backtest](https://samabbott.co.uk/bdbv-hub-modelling/backtest/).

```sh
uv add nowcasteval
```
