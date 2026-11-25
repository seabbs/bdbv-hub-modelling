# bdbvsubmission

The gated weekly submission to the BDBV Modeling Hub: what runs when the hub publishes new data, from fitting candidates to opening the hub pull request.
It checks each candidate forecast, falls back along a chain of models when one fails, and records what it submitted.
It uses the standard library only and has its own environment and lock (`pyproject.toml`, `uv.lock`), so the submission runs with exactly what it needs.

In this project the Submit to the BDBV hub workflow (`.github/workflows/submit.yml`) runs its steps each round.
It reaches the rest of the repository only through the `bdbvhub` command, which Fit candidates (`.github/workflows/fit-candidates.yml`) calls to forecast each candidate.

## Where to look

- How the weekly run works, step by step: [INTEGRATION.md](INTEGRATION.md).
- Setting it up on a fork (secrets, branches, permissions): [SETUP.md](SETUP.md).
- What each step accepts, falls back to or refuses: [`policy.toml`](policy.toml).
- The steps (`steps/`, shell) call the Python in `src/bdbvsubmission/`; `.github/workflows/submit.yml` runs them.
- `model-metadata/`: the metadata file of each of our hub models, as the hub requires. Write and check them with `task bdbvhub:metadata` and `task bdbvhub:validate`.

## Tasks

It is not in the root workspace, so `uv sync` at the root does not install it; its tasks use its own environment.

```sh
task submission:test        # its lint, types and tests
task submission:dry         # the whole run on fixture data, every scenario
task submission:preflight   # what a live submission still lacks (needs gh as a repo admin)
```

Back to the [repository README](../../README.md).

## Citation

Cite it with [CITATION.cff](CITATION.cff); the repository as a whole has [its own](../../CITATION.cff).

```bibtex
@software{abbott_bdbvsubmission,
  author = {Abbott, Sam},
  title = {{bdbvsubmission: gated submission of forecasts to the BDBV Modeling Hub}},
  year = {2026},
  url = {https://github.com/seabbs/bdbv-hub-modelling/tree/main/packages/bdbvsubmission}
}
```
