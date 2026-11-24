# data

Data fetched from outside the repository.
Nothing here is committed but this file: each command below rebuilds its part, and the tests read the frozen copy in `tests/fixtures/data/`.

## What is fetched where

- The hub's tidy tables (`target.csv`, `daily.csv`, `delays.csv.gz`, `health_zones.csv.gz`, `daily_zones.csv.gz`, `locations.csv`, `national-coherence.csv`, `sources.json`) and its task configuration (`hub-config/`): `task nowcastbdbv:data` (`bdbvhub data`), from every hub release, with the raw files in `raw/`.
- `bvd/`: the BVDOutbreakSize releases each backtest vintage may use (for `renewalbvd`): `task nowcastbdbv:data-bvd` (`bdbvhub bvd`).
- `hub-submissions/`: every team's weekly-round forecasts and metadata (CC0-1.0 or CC-BY-4.0), including the hub's copy of ours; other licences go to `raw/hub-submissions/`: `bdbvhub compare-hub --fetch`.
- `raw/submission-ledger/`: each round's submitted model per hub id, from the submission-ledger branch: `bdbvhub compare-hub --fetch`.
- `raw/germany/`: the German hospitalisation benchmark at the config's commits: `nowcastgermany fetch`.

The workflows fetch what they need before use, so a run on a new machine starts from an empty `data/`.
