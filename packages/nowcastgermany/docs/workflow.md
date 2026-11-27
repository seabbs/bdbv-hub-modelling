# How it is used in the workflow

It is the real-data benchmark for the baselines and the fine-tuned model.
The `nowcastgermany` command fetches the hub's data (`fetch`), runs the benchmark (`backtest`) and reads the hub teams' forecasts (`teams`).

In the [workflow](https://github.com/seabbs/bdbv-hub-modelling/blob/main/workflow/README.md), `germany_triangle` fetches the triangle.
`score_germany` and `score_run_germany` score each fine-tuned checkpoint on it with `nowcastgermany.checkpoint`.
`real_training_blocks` cuts the German training blocks for the real-data ablation with `nowcastgermany.finetune_blocks`.

The [paper workflow](https://github.com/seabbs/bdbv-hub-modelling/blob/main/paper/workflow/Snakefile) runs the German study in `germany_fetch`, `germany_development` and `germany_heldout`.
`germany_development` stacks the reference unit and the final checkpoint's unit as our development forecasts (`nowcastgermany units`), with no backtest of its own.
`release_paper_germany` publishes them in the `paper-germany-<hash>` release.

The [Site results workflow](https://github.com/seabbs/bdbv-hub-modelling/blob/main/.github/workflows/site-results.yml) loads our forecasts from the `paper-germany` release the paper reads (`stored-load`), scores the hub teams (`teams`, `asof`, `paper`) and builds the [German nowcast benchmark](https://samabbott.co.uk/bdbv-hub-modelling/germany/) page.
