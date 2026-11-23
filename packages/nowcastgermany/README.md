# nowcastgermany

The German COVID-19 hospitalisation nowcasting hub as a retrospective case study.
It turns the hub's reporting triangles into forecast tasks, scores our models beside the hub's teams, and stores the results for the docs and the paper.

## Start here

- To run the benchmark: `uv run bdbvhub germany-fetch`, then `uv run bdbvhub germany-backtest --help`.
- To read how a task is built and scored, start with [`data`](src/nowcastgermany/data.py), then [`evaluation`](src/nowcastgermany/evaluation.py).

## Modules

- [`data`](src/nowcastgermany/data.py): the hub's reporting triangles and forecasts as tasks.
- [`evaluation`](src/nowcastgermany/evaluation.py): the benchmark, our models scored beside the hub's teams.
- [`store`](src/nowcastgermany/store.py): the benchmark's stored forecasts and scores.
- [`finetune_blocks`](src/nowcastgermany/finetune_blocks.py): German real-data training blocks, for the real-data ablation.
- [`reporting`](src/nowcastgermany/reporting.py): the benchmark's docs page.

It imports `bdbvhub` for the model registry, the download helper, and the site's explorer and model list.

## Next

- The results: [German nowcast benchmark](https://samabbott.co.uk/bdbv-hub-modelling/germany/).
- Fine-tuned checkpoints are scored on it in the workflow's [score](../../workflow/README.md#4-score-rulesscoresmk) step.

```sh
uv add nowcastgermany
```
