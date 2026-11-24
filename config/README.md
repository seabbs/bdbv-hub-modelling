# config

Machine settings, and the fine-tuning settings that checkpoints record by path.

- [`local.env.example`](local.env.example): every machine setting, with its default. Copy it to `config/local.env` (not committed) and set what differs on your machine; `workflow/flowlib/localconfig.py` checks every `local.env` against this list.
- [`finetune.toml`](finetune.toml) and [`finetune/`](finetune) (the simulation grid and the release layer): the fine-tuning run's settings. They stay here because the trainer, whose code is hashed into every checkpoint, reads `config/finetune/grid.yaml` by that path, and each checkpoint records it.

The other settings live beside the code that reads them:

- the hub submission: [`packages/bdbvhub/config/`](../packages/bdbvhub/config) (`hub.toml`, `hub-comparison.toml`);
- the BDBV backtest: [`packages/nowcastbdbv/config/backtest.toml`](../packages/nowcastbdbv/config/backtest.toml);
- the German benchmark: [`packages/nowcastgermany/config/germany.toml`](../packages/nowcastgermany/config/germany.toml);
- the docs site: [`docs/site.toml`](../docs/site.toml).

Back to the [repository README](../README.md).
