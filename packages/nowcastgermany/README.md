# nowcastgermany

The German COVID-19 hospitalisation nowcasting hub as a retrospective case study.
It turns the hub's reporting triangles into forecast tasks and scores models beside the hub's teams.

In this project it is the real-data benchmark for the baselines and the fine-tuned model.
The workflow scores each fine-tuned checkpoint on it and cuts German training blocks for the real-data ablation.
The paper's German study runs its commands, and `bdbvhub` builds the docs site's benchmark page from it.

## Start here

- To run the benchmark: `uv run nowcastgermany fetch`, then `uv run nowcastgermany backtest --help`.
- To read how a task is built and scored, start with [`data`](src/nowcastgermany/data.py), then [`evaluation`](src/nowcastgermany/evaluation.py).

## Modules

- [`cli`](src/nowcastgermany/cli.py): the `nowcastgermany` command (fetch, backtest, teams, asof, paper, units, stored-load, pairing-sensitivity).
- [`data`](src/nowcastgermany/data.py): the hub's reporting triangles and forecasts as tasks.
- [`evaluation`](src/nowcastgermany/evaluation.py): the benchmark, our models scored beside the hub's teams.
- [`store`](src/nowcastgermany/store.py): our models' German development forecasts, stacked from the fine-tuning workflow's units and read back from the paper's `paper-germany` release.
- [`checkpoint`](src/nowcastgermany/checkpoint.py): scoring each published fine-tuned checkpoint on the benchmark (and its held-out loss) against the references.
- [`pairing`](src/nowcastgermany/pairing.py): the walk models' pairing sensitivity on the benchmark.
- [`finetune_blocks`](src/nowcastgermany/finetune_blocks.py): German real-data training blocks, for the real-data ablation.
- [`reporting`](src/nowcastgermany/reporting.py): the benchmark's docs page.

It imports only the generic packages: the registry and download helper from `nowcastutils`, its models from `nowcastbaselines` and `nowcastfoundation`, and the explorer and scores from `nowcasteval`. The site's model list comes from its caller.

## Next

- The results: [German nowcast benchmark](https://samabbott.co.uk/bdbv-hub-modelling/germany/).
- Fine-tuned checkpoints are scored on it in the workflow's [score](../../workflow/README.md#4-score-rulesscoresmk) step.

## Install

From a clone of the repository:

```sh
uv sync --package nowcastgermany             # this package and the ones it needs
```

## Example

```sh
uv run nowcastgermany fetch      # the hub's reporting triangles
task nowcastgermany:benchmark    # run and score the benchmark into results/germany/
```

The results: [German nowcast benchmark](https://samabbott.co.uk/bdbv-hub-modelling/germany/).

## Tasks

`task nowcastgermany:test` runs its tests (`task --list-all` lists the rest).

## Citation

Cite it with [CITATION.cff](CITATION.cff); the repository as a whole has [its own](../../CITATION.cff).

```bibtex
@software{abbott_nowcastgermany,
  author = {Abbott, Sam},
  title = {{nowcastgermany: the German hospitalisation nowcasting hub as a case study: its data as tasks, the benchmark, its store and its page}},
  year = {2026},
  url = {https://github.com/seabbs/bdbv-hub-modelling/tree/main/packages/nowcastgermany}
}
```
