# tests

Tests of the repository as a whole, which no single package holds.
They check the layout, the import contract between the packages, the docs and READMEs, the citations, the CI configuration and the fit store's branch.

In this project `task test` runs them after every package's tests (`testpaths` in `pyproject.toml`).
CI runs them in the Tests workflow's "Workflow, paper and repo tests" job.
Each package's own tests are in `packages/<name>/tests/`, the workflow's in `workflow/tests/` and the paper's in `paper/tests/`.

## Where to look

- `test_*.py`: one file per check, named for what it checks.
- [`fixtures/data/`](fixtures/data): a frozen copy of the hub's tidy tables, which the tests read in place of `data/` (the root `conftest.py` sets `BDBV_DATA_DIR`).
- [`fixtures/bvd/`](fixtures/bvd): BVDOutbreakSize releases for `nowcastrenewal`'s download tests.
- [`fixtures/germany/`](fixtures/germany/README.md): part of the German hospitalisation data, with its source.
- [`model/`](model): references the renewal model is tested against: NumPy and direct versions of its kernels, an R script of the simulator's reporting, and `r_live` to call R packages in live checks.

```sh
task test                     # every package's tests, then these
uv run pytest -q tests        # these only
```

Back to the [repository README](../README.md).
