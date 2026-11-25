# How to

The examples continue from [Getting started](index.md).

## Mix infections between places

A kernel says where each place's contacts go, and a share says how much of its force of infection leaves.
`mixing_matrix` turns them into the matrix the renewal uses.
Here the kernel decays with the distance between the two places.

```python
from nowcastrenewal.components.mixing import (
    distance_kernel,
    great_circle_km,
    mixing_matrix,
)

km = great_circle_km([0.0, 0.5], [20.0, 20.5])
kernel = distance_kernel(km, decay=50.0, per="origin")
mixing = mixing_matrix(kernel, share=0.1)
# Each origin keeps 90% and sends 10% to the other place.
assert np.allclose(mixing, [[0.9, 0.1], [0.1, 0.9]])
mixed = run_renewal(start, gt, rt, population=[1e6, 1e6], mixing=mixing)
# The shrinking place now takes in infections from the growing one.
assert mixed.infections[-1, 1] > run.infections[-1, 1]
```

## Turn infections into expected onsets

`delayed_events` convolves each place's infections with a delay and scales them by the ascertainment.

```python
from nowcastrenewal.components.reference import delayed_events

infections = run.infections[len(gt) :]
onsets = delayed_events(infections, incubation, ascertainment=0.6)
assert onsets.shape == infections.shape
```

## Write a delay as hazards

The fitted models parametrise a reporting delay by its daily hazards of report.
`log_hazard_probs` turns logit hazards back into log probabilities.

```python
from nowcastrenewal.components.hazards import log_hazard_probs
from nowcastrenewal.components.pmfs import censored_logit_hazard, censored_pmf

report = Delay.from_mean_cv("gamma", 3.0, 0.8)
pmf = censored_pmf(report, 10)
log_p = log_hazard_probs(censored_logit_hazard(report, 10))
assert np.allclose(np.exp(log_p), pmf, atol=1e-6)
```

## Look up the model variants

The hub models are variants of one configuration (`ModelConfig`).
`renewalbvd` adds the BVDOutbreakSize reproduction numbers as a pseudo-observation.

```python
from nowcastrenewal.forecaster import VARIANTS, model_config

assert list(VARIANTS) == ["renewal", "renewalsp", "renewalbvd"]
assert model_config(VARIANTS["renewalbvd"]).bvd
assert not model_config(VARIANTS["renewal"]).bvd
```

The model's fixed delays come from the same pmfs.

```python
from nowcastrenewal.model.model import (
    generation_interval,
    incubation_period,
    seed_lead,
)

assert np.allclose(generation_interval(), gt)
# Days to seed a place before its first case week.
assert seed_lead(incubation_period()) == 5
```

## Choose a BVDOutbreakSize release

A forecast may use a BVDOutbreakSize release only if it existed at the forecast's cut-off.
`bvd_cutoff` gives that time, and `parse_releases` keeps the `results-<run>` releases from the GitHub release list.

```python
from nowcastrenewal.bvd import bvd_cutoff, parse_releases

cutoff = bvd_cutoff("2026-10-24")
assert cutoff.isoformat() == "2026-10-25T23:59:59+01:00"
payload = [
    {"tag_name": "results-3169", "published_at": "2026-10-27T10:00:00Z"},
    {"tag_name": "latest", "published_at": "2026-10-27T10:00:00Z"},
]
releases = parse_releases(payload)
assert list(releases["tag"]) == ["results-3169"]
```

`select_release` then picks the newest of them that qualifies.

Next: [Reference](reference.md).
