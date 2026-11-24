# nowcastgermany

The German COVID-19 hospitalisation nowcasting hub as a retrospective case study.
It turns the hub's reporting triangles into forecast tasks, scores our models beside the hub's teams, and stores the results for the docs and the paper.

## Start here

- To run the benchmark: `uv run nowcastgermany fetch`, then `uv run nowcastgermany backtest --help`.
- To read how a task is built and scored, start with [`data`](src/nowcastgermany/data.py), then [`evaluation`](src/nowcastgermany/evaluation.py).

## Modules

- [`cli`](src/nowcastgermany/cli.py): the `nowcastgermany` command (fetch, backtest, teams, asof, paper, ft-load, ft-tag, unit-publish, pairing-sensitivity).
- [`data`](src/nowcastgermany/data.py): the hub's reporting triangles and forecasts as tasks.
- [`evaluation`](src/nowcastgermany/evaluation.py): the benchmark, our models scored beside the hub's teams.
- [`store`](src/nowcastgermany/store.py): the fine-tuned model's German forecasts, packed from the German unit as an asset of its checkpoint's release.
- [`checkpoint`](src/nowcastgermany/checkpoint.py): scoring each published fine-tuned checkpoint on the benchmark (and its held-out loss) against the references.
- [`pairing`](src/nowcastgermany/pairing.py): the walk models' pairing sensitivity on the benchmark.
- [`finetune_blocks`](src/nowcastgermany/finetune_blocks.py): German real-data training blocks, for the real-data ablation.
- [`reporting`](src/nowcastgermany/reporting.py): the benchmark's docs page.

It imports only the generic packages: the registry and download helper from `nowcastutils`, its models from `nowcastbaselines` and `nowcastfoundation`, and the explorer and scores from `nowcasteval`. The site's model list comes from its caller.

## Next

- The results: [German nowcast benchmark](https://samabbott.co.uk/bdbv-hub-modelling/germany/).
- Fine-tuned checkpoints are scored on it in the workflow's [score](../../workflow/README.md#4-score-rulesscoresmk) step.

```sh
uv add nowcastgermany
```
