# config

This machine's settings: paths, hosts, cores and devices for the workflow, the paper run and the Taskfile.

In this project `workflow/flowlib/localconfig.py` reads `config/local.env`.
The Taskfile, the Snakemake workflow and the shell scripts call it, so every one of them sees the same values.

## Where to look


- [`local.env.example`](local.env.example): every machine setting, with its default. Copy it to `config/local.env` (not committed) and set what differs on your machine; `workflow/flowlib/localconfig.py` checks every `local.env` against this list.

The other settings live beside the code that reads them:

- fine-tuning (the run's settings, the simulation grid and the release layer): [`packages/nowcastfoundation/config/`](../packages/nowcastfoundation/config);
- the hub submission: [`packages/bdbvhub/config/`](../packages/bdbvhub/config) (`hub.toml`, `hub-comparison.toml`);
- the BDBV backtest: [`packages/nowcastbdbv/config/backtest.toml`](../packages/nowcastbdbv/config/backtest.toml);
- the German benchmark: [`packages/nowcastgermany/config/germany.toml`](../packages/nowcastgermany/config/germany.toml);
- the docs site: [`docs/site.toml`](../docs/site.toml).

Back to the [repository README](../README.md).
