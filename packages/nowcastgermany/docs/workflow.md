# How it is used in the workflow

It is the real-data benchmark for the baselines and the fine-tuned model.
The `nowcastgermany` command fetches the hub's data (`fetch`), runs the benchmark (`backtest`) and reads the hub teams' forecasts (`teams`).

In the [workflow](https://github.com/seabbs/bdbv-hub-modelling/blob/main/workflow/README.md), `germany_triangle` fetches the triangle.
`score_germany` and `score_run_germany` score each fine-tuned checkpoint on it with `nowcastgermany.checkpoint`.
`real_training_blocks` cuts the German training blocks for the real-data ablation with `nowcastgermany.finetune_blocks`.
`publish_germany_unit` attaches a checkpoint's German forecasts to its release (`nowcastgermany unit-publish`).

The [paper workflow](https://github.com/seabbs/bdbv-hub-modelling/blob/main/paper/workflow/Snakefile) runs the German study in `germany_fetch`, `germany_development` and `germany_heldout`.

The [Site workflow](https://github.com/seabbs/bdbv-hub-modelling/blob/main/.github/workflows/pages.yml) loads the pinned checkpoint's stored forecasts (`ft-tag`, `ft-load`), scores the hub teams (`teams`, `asof`, `paper`) and builds the [German nowcast benchmark](https://samabbott.co.uk/bdbv-hub-modelling/germany/) page.
