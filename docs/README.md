# docs

The source of the [docs site](https://samabbott.co.uk/bdbv-hub-modelling/), built with MkDocs (`mkdocs.yml` at the root).
The site shows the models, their backtests, the comparison with the hub's teams, the German benchmark and the paper.

In this project the Site workflow (`.github/workflows/pages.yml`) builds it with the stored fits and the paper and publishes it to `gh-pages`.
A package without its own site gets a page here that shows its README.

## Where to look


- Pages are Markdown here; `generated/` and `figures/` are written by the Pages build (`bdbvhub report` and friends) and not committed.
- `hooks/` holds the MkDocs hooks: cache busting, the citations listed from each `CITATION.cff`, and the package pages drawn from their READMEs (`readmes.py`).
- `examples/` holds snippets the pages include.
- Each package with its own site has it under `packages/<name>/docs/`.

```sh
task docs     # build the site into _site/
task serve    # serve it while editing
```

Back to the [repository README](../README.md).
