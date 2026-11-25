# How it is used in the workflow

The [workflow](https://github.com/seabbs/bdbv-hub-modelling/blob/main/workflow/README.md) runs this package in four steps.
Simulate (`workflow/rules/simulate.smk`) runs `python -m nowcastfoundation.simulator` for each batch of simulated outbreaks.
Fine-tune (`workflow/rules/finetune.smk`) trains Chronos-2 on them with the streaming trainer in `nowcastfoundation.finetune.stream`.
Publish (`workflow/rules/publish.smk`) checks each checkpoint and releases it on GitHub, and on the Hugging Face Hub through `nowcastfoundation.hubpublish`.
Evaluate (`workflow/rules/evaluate.smk`) scores each checkpoint on held-out simulations.

The hub models `foundation_zeroshot` and `foundation` are registered in `nowcastfoundation.forecaster`.
`bdbvhub` fits them in the Fit store workflow (`.github/workflows/fit-store.yml`) and submits `seabbs-foundation` when `hub.toml` says so.
`packages/nowcastfoundation/config/finetune.toml` says which checkpoint they load.
The German benchmark and the paper score the same checkpoints.
