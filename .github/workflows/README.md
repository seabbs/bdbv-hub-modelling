# Workflows

Each workflow, when it runs and what it writes.

| Workflow | File | When it runs | What it does and writes |
| --- | --- | --- | --- |
| Tests | `tests.yml` | Every push to `main` or `next`, every pull request, nightly, on dispatch | Every test: the repo checks, a job per package the change reaches, the submission scripts (`ci/`), the workflow and paper tests, and the slow tiers (model fits and backtests, the torch tests, the live R checks) nightly, on dispatch, with the `full-tests` label, or when a pull request reaches their code. Writes nothing but coverage (on `main`) and test durations. |
| Site | `pages.yml` | Pushes to `main` that touch the site's inputs, every 6 hours, pull requests (as a preview), on dispatch | Builds the docs site and the paper from the stored fits and forecasts, and publishes them to `gh-pages` (a pull request's preview under `pr-preview/`). |
| Fit store | `fit-store.yml` | Pushes to `main` that touch fit code, every 6 hours, on dispatch; a pull request only lists what it would fit | Fits each fitted model once per origin and stores the fits and the backtest forecasts on the `fit-store` branch, then asks Site to rebuild. |
| Fit candidates | `fit-candidates.yml` | Called by Submit to the BDBV hub | Forecasts from and checks each fitted candidate of the submission's fallback chain. |
| Submit to the BDBV hub | `submit.yml` | On the round's schedule and on dispatch | Checks the week's forecast and opens or updates the pull request to the hub; records each round on the `submission-ledger` branch and its release. |
| Round assets | `round-assets.yml` | On dispatch | Attaches Submit runs' files (candidates, checks, provenance, inputs) to their rounds' releases. |
| Clean up after a PR | `pr-closed.yml` | When a pull request closes or merges | Cancels its runs, deletes its caches and main's superseded ones, and removes its site preview. |
