# How to

The examples continue from [Getting started](index.md).

## See how recent weeks fill in

`revision_table` compares each release's counts with a later reference release.
It is the reporting triangle in weekly form.

```python
from nowcastbdbv.data.tidy import revision_table

revisions = revision_table(target)
first = revisions.loc[revisions["vintage"] == "2026-09-26"]
assert (first["fraction"] <= 1).all()
```

`newly_missing` gives the cases added between consecutive releases, split by onset status.

```python
from nowcastbdbv.data.tidy import newly_missing

added = newly_missing(target)
assert added["new_total"].item() == 42 - 32
assert added["days"].item() == 7
```

## Key a hub release

The hub labels a release by its `as_of` date and may rewrite older labels a day later.
`release_id` maps a label back to the one results and the submission ledger are keyed by.

```python
from nowcastbdbv.data.tidy import release_id

assert release_id("2026-10-25") == "2026-10-24"
assert release_id("2026-11-01") == "2026-11-01"
```

## Estimate onset-to-report delays

`empirical_pmf` turns delays in days into a daily pmf, and `delay_summary` summarises them by group.

```python
import numpy as np

from nowcastbdbv.data.delays import delay_summary, empirical_pmf

pmf = empirical_pmf([0, 1, 1, 2, 3, 5, 40], max_delay=10)
assert np.isclose(pmf.sum(), 1.0)
delays = pd.DataFrame(
    {"vintage": "2026-10-03", "location": "CD-NK", "delay": [1, 2, 2, 4, 7]}
)
assert delay_summary(delays)["median"].item() == 2
```

`hub_delay_pmf` gives the pmf of the delay the hub uses to impute missing onsets at a release.

```python
from nowcastbdbv.data.delays import hub_delay_pmf

hub = hub_delay_pmf("2026-10-24", max_delay=20)
assert np.isclose(hub.sum(), 1.0)
```

## Give places their populations and centres

Province populations and population-weighted centres ship with the package, as do the health zones' centres.

```python
from nowcastbdbv.data.tidy import province_places
from nowcastbdbv.data.zones import PLACE_COLUMNS, zone_places

provinces = province_places(["CD-NK", "CD-IT"])
assert list(provinces.columns) == list(PLACE_COLUMNS)
zones = zone_places()
assert zones["location"].str.startswith("CD-NK-").any()
```

Next: [Reference](reference.md).
