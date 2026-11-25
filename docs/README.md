# docs

The source of the [docs site](https://samabbott.co.uk/bdbv-hub-modelling/), built with MkDocs (`mkdocs.yml` at the root).
The site shows the models, their backtests, the comparison with the hub's teams, the German benchmark and the paper.

In this project the Site workflow (`.github/workflows/pages.yml`) builds it with the stored fits and the paper and publishes it to `gh-pages`.
Each package has its own site (`packages/<name>/mkdocs.yml` and `docs/`, from `packages/mkdocs-base.yml`), which the Package docs workflow builds and publishes at `<site>/<name>/` when the package changes.
The Packages section lists every package site it finds in `packages/`.

## Where to look


- Pages are Markdown here; `generated/` and `figures/` are written by the Pages build (`bdbvhub report` and friends) and not committed.
- `hooks/` holds the MkDocs hooks: cache busting, the citations listed from each `CITATION.cff`, README contents marked into a page (`readmes.py`), the package sites found in `packages/` (`packages.py`) and each package site's links back to this one (`site_links.py`).
- `examples/` holds snippets the pages include.

```sh
task docs     # build the site into _site/
task serve    # serve it while editing
```

Back to the [repository README](../README.md).
