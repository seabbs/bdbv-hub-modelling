# docs

The source of the [docs site](https://samabbott.co.uk/bdbv-hub-modelling/), built with MkDocs (`mkdocs.yml` at the root).

- Pages are Markdown here; `generated/` and `figures/` are written by the Pages build (`bdbvhub report` and friends) and not committed.
- `hooks/` holds the MkDocs hooks: cache busting, and the citations listed from each `CITATION.cff`.
- `examples/` holds snippets the pages include.
- Each package with its own site has it under `packages/<name>/docs/`.

```sh
task docs     # build the site into _site/
task serve    # serve it while editing
```

Back to the [repository README](../README.md).
