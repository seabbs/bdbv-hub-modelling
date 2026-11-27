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
| Fit store | `fit-store.yml` | Pushes to `main` that touch fit code, every 6 hours, on dispatch; a pull request only lists what it would fit | Fits each fitted model once per origin and stores the fits and the scored backtests as the Actions artifacts `fit-store` and `fit-store-backtests`, starting Site results after each store. |
| Site results | `site-results.yml` | Started by Fit store after it stores backtests or fits; on dispatch | Makes the site's results from what the fit store holds now (fits not yet stored are shown as pending), stores them as the artifact `fit-store-site`, then asks Site to rebuild. |
| Fit candidates | `fit-candidates.yml` | Called by Submit to the BDBV hub | Forecasts from and checks each fitted candidate of the submission's fallback chain. |
| Submit to the BDBV hub | `submit.yml` | On the round's schedule and on dispatch | Checks the week's forecast and opens or updates the pull request to the hub; records each round on the `submission-ledger` branch and its release. |
| Round assets | `round-assets.yml` | On dispatch | Attaches Submit runs' files (candidates, checks, provenance, inputs) to their rounds' releases. |
| Fit store seed | `fit-store-seed.yml` | On dispatch, from `main` | Uploads the store the old `fit-store` git branch holds as the store's artifacts, once, so nothing is refitted. |
| Clean up after a PR | `pr-closed.yml` | When a pull request closes or merges | Cancels its runs, deletes its caches and main's superseded ones, and removes its site preview. |

## Where the fit store lives

The fit store is kept as GitHub Actions artifacts, not on a branch, so it adds nothing to the repository's history (`packages/bdbvhub/src/bdbvhub/store/artifact.py`).

| Artifact | Written by | What it holds |
| --- | --- | --- |
| `fit-store` | Fit store's `store-*` jobs | Every stored fit, `<model>/<origin>/<key>.json.gz`, with the walks' zone runs beside them. |
| `fit-store-backtests` | Fit store's `backtests` job | The scored backtests, `latest.tar.gz`, and the commit that made them, `latest.json`. |
| `fit-store-site` | Site results | The site's results, `latest.tar.gz`, with their commit and input hash, `latest.json`. |

A reader takes the newest unexpired artifact of each name made by a run on `main` in this repository.
Each store job fetches the newest `fit-store`, adds its fits and uploads the whole store again, and the store jobs of a run go one after another.
Each artifact is kept 90 days, and every store renews it.
Nothing fails when one has expired or is not yet made.
An empty store is fitted again by the next Fit store run, missing backtests leave their sections out, and Site builds nothing while the site results are missing: on `main` it starts Site results, which rebuilds the site when they are made.

A dispatch from another branch can name a scratch store (`store`, a name starting `fit-store-scratch`), read and written under that name from any branch and kept 7 days.
Pull requests read the main store and write nothing.

What is kept for good is each round's release (`round-<reference date>`): the Pages build attaches every model's results, forecasts and data (`results-release attach`), and Submit attaches the run's pinned inputs, every candidate's hub file and checks, the fit directory each candidate forecast from and the gates' provenance (`packages/bdbvsubmission/steps/round_assets.sh`).

Until the first store artifact exists, readers read the old `fit-store` branch, and `fit-store-seed.yml` uploads its store as the artifacts.
