# ci

The hub submission's own CI: what runs each week when the hub publishes new data, from fitting candidates to opening the hub pull request.
It is its own uv project (`pyproject.toml`, `uv.lock`) so the submission runs with exactly what it needs.

- How the weekly run works, step by step: [INTEGRATION.md](INTEGRATION.md).
- Setting it up on a fork (secrets, branches, permissions): [SETUP.md](SETUP.md).
- What each step accepts, falls back to or refuses: [`policy.toml`](policy.toml).
- The steps (`steps/`, shell) call the Python in `src/submission_ci/`; `.github/workflows/submit.yml` runs them.

```sh
task ci:test        # its lint, types and tests
task ci:dry         # the whole run on fixture data, every scenario
task ci:preflight   # what a live submission still lacks (needs gh as a repo admin)
```

Back to the [repository README](../README.md).
