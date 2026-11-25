# How to

The examples continue from [Getting started](index.md).

## Sum a daily triangle into weeks

`aggregate_triangle` turns a daily triangle into steps of `k` days.
Bins end on the latest cut-off and every `k` days before it.

```python
from nowcastfoundation.finetune import aggregate_triangle

weekly = aggregate_triangle(full, 7, 2)
assert weekly.times[-1] == "2026-03-21"
assert weekly.partial.shape == (1, 7, 2)
# The last week is still being reported.
assert weekly.final[0, -1] < weekly.final[0, 0] == 70
```

`rolling_totals` gives each day's sum over the `k` days to it instead.

```python
from nowcastfoundation.forecaster import rolling_totals

assert rolling_totals(tri.final, 7)[0, -1] == 52
```

## Name a checkpoint's release

Each fine-tuned checkpoint is released on GitHub under a tag built from its settings hash.
A checkpoint with the series bias has a `slots-` hash.

```python
from nowcastfoundation.forecaster import release_tag

assert release_tag("abc123") == "chronos-ft-abc123"
assert release_tag("slots-abc123") == "chronos-slots-abc123"
```

`nowcastfoundation.checkpoints.load` downloads a release and checks it against its SHA-256 sums.
`forecast_triangle` then nowcasts your own reporting triangle with it.
Both need the `chronos` group and the network, so they are not run here.

## Lay out the series bias

The series bias ties each row of a block to the other rows of its place and to its parent place.
`parents_of` finds each place's parent from its code, and the country is the parent of a place with no nearer one.

```python
from nowcastfoundation.series_bias import block_rows, parents_of

places = ["CD", "CD-NK", "CD-NK-03", "CD-SK"]
assert parents_of(places) == [-1, 0, 1, 0]
# Four target rows, then two lag series per place.
rows = block_rows([0, 1, 2, 3], 8, parents_of(places))
assert rows[0].tolist() == [0, 1, 2, 3, 0, 0, 1, 1, 2, 2, 3, 3]
assert rows[1].tolist() == [-1, 0, 1, 0, -1, -1, 0, 0, 1, 1, 0, 0]
```

## Draw a simulation's settings from a grid

The simulator draws each simulation's settings from the grid file.
A tiny grid shows the draws.
`generator` keys the random numbers by the batch's seed and the simulation's place in it.

```python
from nowcastfoundation.simulator.draws import draw_cell, draw_setting, generator

grid = {
    "factors": {"trajectory": {"growth": 2, "decline": 1}},
    "distributions": {"r0": {"dist": "lognormal", "median": 1.2, "sdlog": 0.2}},
}
rng = generator(1, 0, 3)
cell = draw_cell(grid, rng)
assert cell["trajectory"] in ("growth", "decline")
assert draw_setting(grid, "r0", rng) > 0
assert draw_setting(grid, "r0", rng, n=3).shape == (3,)
```

## Count the training blocks' combinations

The trainer counts each block's layout, levels and parents as it trains.
`shares` gives the table of them.

```python
from nowcastfoundation.finetune.combinations import count, shares

seen = {}
count(
    seen,
    [
        {"source": "daily", "mix": "system"},
        {"source": "daily", "mix": "system"},
        {"source": "weekly", "mix": "country", "parent": "loose"},
    ],
)
table = shares(seen["combinations"])
assert table["blocks"].tolist() == [2, 1]
assert table["share"].sum() == 1
```

Next: [Reference](reference.md).
