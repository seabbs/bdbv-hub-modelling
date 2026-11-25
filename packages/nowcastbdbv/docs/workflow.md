# How it is used in the workflow

`bdbvhub` reads the hub's data through this package.
`uv run bdbvhub data` downloads and tidies the hub's streams into `data/`, and the fit store, the backtest and the submission load them from there.
The backtest's tasks come from `nowcastbdbv.backtest` with the settings in [`config/backtest.toml`](https://github.com/seabbs/bdbv-hub-modelling/blob/main/packages/nowcastbdbv/config/backtest.toml).
The Fit store workflow ([`fit-store.yml`](https://github.com/seabbs/bdbv-hub-modelling/blob/main/.github/workflows/fit-store.yml)) refits when the package's data code or that config changes.
The workflow's round reports load the target with `load_target`, and its scoring and ablation rules read the same backtest config.
The paper's Ebola study uses the same release backtest.
