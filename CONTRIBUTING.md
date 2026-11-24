# Contributing

Set up with `task setup`.
`task --list` shows every task.

## Checks

Run these before sending a change:

```bash
task lint             # ruff and strict mypy
task test-all         # every test, the model fit tests included
task submission:test      # the submission workflow tests
task dry:submit       # the submission workflow dry runs
task docs             # the site, built in strict mode
```

`task test` leaves out the model fit tests (marked `slow`).
The CI runs them weekly and on demand.
Any change to `packages/nowcastrenewal/`, the simulator or the renewal forecaster needs `task test-all` to pass first.

## Tasks

Each Taskfile lives beside its code and runs in its own folder: `packages/<package>/Taskfile.yml` for each package (namespace: the package name, for example `task bdbvhub:nowcast`), `workflow/Taskfile.yml` (`workflow`), `packages/bdbvsubmission/Taskfile.yml` (`ci`) and `paper/Taskfile.yml` (`paper`).
The root Taskfile only includes them, beside the top-level tasks (setup, test, lint, format, docs, serve, getting-started, paper), so every task is found from the repository root as `task <namespace>:<task>`, or as `task <task>` inside its folder.
The named dry runs are `task dry:tick`, `task dry:paper`, `task dry:submit` and `task dry:fit-store`.

The tasks' names before the project Taskfiles:

| Old | New |
|---|---|
| `paper:render` | `paper` |
| `data` | `nowcastbdbv:data` |
| `data-bvd` | `nowcastbdbv:data-bvd` |
| `backtest` | `bdbvhub:backtest` |
| `backtest-lopo` | `bdbvhub:backtest-lopo` |
| `backtest-zones` | `bdbvhub:backtest-zones` |
| `results` | `bdbvhub:results` |
| `score-history` | `nowcasteval:score-history` |
| `germany` | `nowcastgermany:benchmark` |
| `compare-hub` | `bdbvhub:compare-hub` |
| `plots` | `bdbvhub:plots` |
| `simulate` | `workflow:simulate` |
| `push` | `workflow:push` |
| `release` | `workflow:release` |
| `finetune` | `workflow:finetune` |
| `publish` | `workflow:publish` |
| `score` | `workflow:score` |
| `land` | `workflow:land` |
| `dashboard` | `workflow:dashboard` |
| `dashboard:static` | `workflow:dashboard-static` |
| `all` | `workflow:all` |
| `tick` | `workflow:tick` |
| `run` | `workflow:run` |
| `monitor` | `workflow:monitor` |
| `dag` | `workflow:dag` |
| `rulegraph` | `workflow:rulegraph` |
| `progress` | `workflow:progress` |
| `report` | `workflow:report` |
| `workflow:check` | `workflow:check` |
| `nowcast` | `bdbvhub:nowcast` |
| `metadata` | `bdbvhub:metadata` |
| `validate-hub` | `bdbvhub:validate` |
| `submission-preflight` | `submission:preflight` |
| `ci-test` | `submission:test` |
| `ci-dry` | `submission:dry` |
| `ci-fit` | `bdbvhub:ci-fit` |
| `fit-store` | `bdbvhub:fit-store` |
| `fit-store-plan` | `bdbvhub:fit-store-plan` |

## Commits

Write a failing test, commit it, make it pass, commit again.
Keep each commit to one change.
