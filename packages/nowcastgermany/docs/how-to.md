# How to

The examples continue from [Getting started](index.md).

## Name the hub's places

The tasks code Germany as the country `CD`, its states as provinces and its age groups as `CD-A<i>`.
`level` gives the level each place is scored at.

```python
from nowcastgermany.data import level, place_code

assert place_code("DE") == "CD"
assert place_code("DE-BY") == "CD-BY"
assert place_code("DE", "35-59") == "CD-A3"
levels = [level(p) for p in ("DE", "DE-BY", "35-59")]
assert levels == ["national", "states", "age"]
```

## Score models on the benchmark

`germany_forecasts` runs registered models on each nowcast date and keeps their quantiles at the hub's levels.
Here the two baselines nowcast the last three 7-day sums.

```python
from nowcastgermany.evaluation import germany_forecasts

forecasts = germany_forecasts(
    ["randomwalk", "randomwalknowcast"],
    triangle,
    ["2021-11-21"],
    n_samples=100,
    seed=1,
    horizons=(0, -1, -2),
)
places = {"DE", "DE-BE", "DE-BY", "35-59", "60-79"}
assert set(forecasts["location"]) == places
```

`with_truth` adds each sum's truth, its count reported $D$ days after it ends.
`score_germany` scores the forecasts by WIS and interval coverage, as the hub did.
`relative_scores` divides each model's scores by a reference model's, on the forecasts both made.

```python
from nowcastgermany.evaluation import relative_scores, score_germany, with_truth

scores = score_germany(with_truth(forecasts, triangle, 5), 5)
assert set(scores["pool"]) == {"national", "states", "age"}
skill = relative_scores(scores, ["target"], "randomwalknowcast")
reference = skill.loc[skill["model"] == "randomwalknowcast"]
assert reference["rel_wis"].iloc[0] == 1.0
```

## Keep the held-out half out of the page

The nowcast dates are split at a split date, and the later half is held out for the paper.
`held_out_leaks` lists any held-out date the page's files name.

```python
from pathlib import Path

from nowcastgermany.reporting import held_out_leaks

Path("summary.json").write_text('{"last": "2021-12-01"}')
assert held_out_leaks([Path("summary.json")], "2022-01-01") == []
assert held_out_leaks([Path("summary.json")], "2021-11-29") != []
```

## Cut German training blocks

The real-data ablation trains only on reports before the first scored German date.
`window` gives the last report date used and the last date with truth.

```python
from nowcastgermany.finetune_blocks import window

assert window("2021-11-22", 40) == ("2021-11-21", "2021-10-12")
```

Next: [Reference](reference.md).
