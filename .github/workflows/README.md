# Workflows

The GitHub Actions workflows that test the code, build the docs site, keep the fit store and submit to the BDBV Modeling Hub.

In this project they are what runs without a person: every pull request's checks, the site every 6 hours, and the weekly hub submission.
They call the same Taskfile tasks and package commands as a local run, with shared setup in `.github/actions/` and helper scripts in `.github/scripts/`.

Each workflow, when it runs and what it writes:

| Workflow | File | When it runs | What it does and writes |
| --- | --- | --- | --- |
| Tests | `tests.yml` | Every push to `main` or `next`, every pull request, nightly, on dispatch | Every test: the repo checks, a job per package the change reaches, the submission scripts (`packages/bdbvsubmission/`), the workflow and paper tests, and the slow tiers (model fits and backtests, the torch tests, the live R checks) nightly, on dispatch, with the `full-tests` label, or when a pull request reaches their code. Writes nothing but coverage (on `main`) and test durations. |
| Site | `pages.yml` | Pushes to `main` that touch the site's inputs, every 6 hours, pull requests (as a preview), on dispatch | Builds the docs site and the paper from the stored fits and forecasts, and publishes them to `gh-pages` (a pull request's preview under `pr-preview/`). |
| Package docs | `package-docs.yml` | Pushes to `main` and pull requests that touch a package, the template or its hooks; on dispatch | Builds each changed package's docs site (`packages/<package>/mkdocs.yml`) and, from `main`, publishes it to `gh-pages` at `<package>/`, which the Site deploy keeps. |
| Fit store | `fit-store.yml` | Pushes to `main` that touch fit code, every 6 hours, on dispatch; a pull request only lists what it would fit | Fits each fitted model once per origin and stores the fits and the backtest forecasts on the `fit-store` branch, then asks Site to rebuild. |
| Fit candidates | `fit-candidates.yml` | Called by Submit to the BDBV hub | Forecasts from and checks each fitted candidate of the submission's fallback chain. |
| Submit to the BDBV hub | `submit.yml` | On the round's schedule and on dispatch | Checks the week's forecast and opens or updates the pull request to the hub; records each round on the `submission-ledger` branch and its release. |
| Round assets | `round-assets.yml` | On dispatch | Attaches Submit runs' files (candidates, checks, provenance, inputs) to their rounds' releases. |
| Clean up after a PR | `pr-closed.yml` | When a pull request closes or merges | Cancels its runs, deletes its caches and main's superseded ones, and removes its site preview. |
