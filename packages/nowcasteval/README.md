# nowcasteval

Scores for sample and quantile nowcasts and forecasts, and backtests that compare models with a baseline.
Use it to score forecasts of counts that are revised as later releases report more cases.

In this project it scores every model the same way.
`bdbvhub` uses it for the BDBV backtest and the score pages of the docs site, and `nowcastgermany` for the German benchmark.
The workflow scores fine-tuned checkpoints with it (`workflow/flowlib/`), and the paper builds its figures from its scores.

## Start here

- To score forecasts, read [`scoring`](src/nowcasteval/scoring.py) (`score_forecasts`).
- To score a model over past releases, read [`backtest`](src/nowcasteval/backtest.py).
- The [docs site](https://samabbott.co.uk/bdbv-hub-modelling/nowcasteval/) has a worked example, how-to recipes and the API reference.

## Modules

- [`scoring`](src/nowcasteval/scoring.py): sample CRPS, WIS and its components, and the energy score, by level.
- [`backtest`](src/nowcasteval/backtest.py): forecasts over a run of releases, scored against later data.
- [`score_history`](src/nowcasteval/score_history.py): backtest scores over time, one point per deploy of the site.
- [`nowcast`](src/nowcasteval/nowcast.py): nowcast tasks and their truth.
- [`coupling`](src/nowcasteval/coupling.py): trajectories coupled across places and scored on what was reported.
- [`marginals`](src/nowcasteval/marginals.py), [`ppc`](src/nowcasteval/ppc.py), [`checks`](src/nowcasteval/checks.py): marginal summaries, predictive checks and model checks.
- [`pairing`](src/nowcasteval/pairing.py): scores of the walk models' draws under each pairing.
- [`labels`](src/nowcasteval/labels.py), [`plots`](src/nowcasteval/plots.py), [`tables`](src/nowcasteval/tables.py), [`scorechart`](src/nowcasteval/scorechart.py), [`scoresummary`](src/nowcasteval/scoresummary.py), [`nowcast_plots`](src/nowcasteval/nowcast_plots.py), [`pairing_plots`](src/nowcasteval/pairing_plots.py), [`explorer`](src/nowcasteval/explorer.py): display names, figures, tables of scores and the forecast explorer's data.

## Next

- [How it is used in the workflow](https://samabbott.co.uk/bdbv-hub-modelling/nowcasteval/workflow/).
- The scores in use: [Evaluation](https://samabbott.co.uk/bdbv-hub-modelling/methods/evaluation/) and [Backtest](https://samabbott.co.uk/bdbv-hub-modelling/backtest/).

## Install

From a clone of the repository:

```sh
uv sync --package nowcasteval             # this package and the ones it needs
```

## Example

```python
from nowcasteval.scoring import relative_skill, score_forecasts

scores = score_forecasts(forecasts, truth)  # sample forecasts against truth
relative_skill(scores, baseline="randomwalk")
```

A runnable version: [how-to](https://samabbott.co.uk/bdbv-hub-modelling/nowcasteval/how-to/).

## Tasks

`task nowcasteval:test` runs its tests (`task --list-all` lists the rest).

## Citation

Cite it with [CITATION.cff](CITATION.cff); the repository as a whole has [its own](../../CITATION.cff).

```bibtex
@software{abbott_nowcasteval,
  author = {Abbott, Sam},
  title = {{nowcasteval: evaluation of nowcasts and forecasts: proper scores by level, backtests over releases, trajectory pairing, model checks and score figures}},
  year = {2026},
  url = {https://github.com/seabbs/bdbv-hub-modelling/tree/main/packages/nowcasteval}
}
```
