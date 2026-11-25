# How it is used in the workflow

The CI workflows and the Snakemake rules run the package through the `bdbvhub` command.

- [Fit store](https://github.com/seabbs/bdbv-hub-modelling/blob/main/.github/workflows/fit-store.yml) downloads the hub's data (`bdbvhub data`) fits and stores every model (`bdbvhub fit-store`) and runs the random walk backtests (`bdbvhub backtest`).
- [Fit candidates](https://github.com/seabbs/bdbv-hub-modelling/blob/main/.github/workflows/fit-candidates.yml) unpacks each candidate's stored fit (`bdbvhub fit-store unpack`) and writes its hub file (`bdbvhub nowcast`).
- [Submit to the BDBV hub](https://github.com/seabbs/bdbv-hub-modelling/blob/main/.github/workflows/submit.yml) reaches the rest of the repository only through the `bdbvhub` command.
- [Site](https://github.com/seabbs/bdbv-hub-modelling/blob/main/.github/workflows/pages.yml) loads the stored fits, compares them with the hub's forecasts and builds the hub pages of the docs site (`bdbvhub fit-store load`, `bdbvhub compare-hub`, `bdbvhub report`).

In the [workflow](https://github.com/seabbs/bdbv-hub-modelling/tree/main/workflow), the `hub_data`, `hub_comparison`, `rt_check`, `this_round`, `ft_check_forecast` and `ft_check_page` rules call it, as does `ablation_forecast`.
The paper's workflow reads the hub's releases with `bdbvhub data`.

How a week's submission runs is on the main site: [Submission workflow](https://samabbott.co.uk/bdbv-hub-modelling/submission/workflow/).
