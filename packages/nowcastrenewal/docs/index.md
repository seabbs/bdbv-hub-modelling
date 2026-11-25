# nowcastrenewal

Renewal models of an outbreak, built on PyRenew: the fitted renewal models and the joint model that simulates outbreaks.
{ .page-subtitle }

nowcastrenewal is the package in `packages/nowcastrenewal` of the repository ([Packages](https://samabbott.co.uk/bdbv-hub-modelling/packages/)).

## Install

```sh
uv pip install "git+https://github.com/seabbs/bdbv-hub-modelling#subdirectory=packages/nowcastrenewal"
```

The fitted renewal model also needs PyRenew, the `model` extra.

## Getting started

The fitted models and the simulator share one set of components.
Each delay is a continuous distribution recorded on whole days, so its pmf is double interval censored.
Here we make the generation time from its mean and coefficient of variation.

```python
import numpy as np

from nowcastrenewal.components.pmfs import (
    Delay,
    delay_pmf,
    generation_time_pmf,
)

gt = generation_time_pmf(Delay.from_mean_cv("gamma", 15.3, 9.3 / 15.3), 42)
# PyRenew's convention: element 0 is a lag of one day.
assert gt.shape == (42,)
assert np.isclose(gt.sum(), 1.0)
incubation = delay_pmf(Delay.from_mean_cv("lognormal", 6.3, 3.5 / 6.3), 28)
assert incubation.shape == (29,)
```

`run_renewal` runs each place's renewal from starting infections and reproduction numbers.
We run two places for 60 days, one growing and one shrinking.

```python
from nowcastrenewal.components.renewal import run_renewal

n_days, n_places = 60, 2
start = np.full((len(gt), n_places), 5.0)
rt = np.tile([1.3, 0.8], (n_days, 1))
run = run_renewal(start, gt, rt, population=[1e6, 1e6])
# The starting infections, then one row a day.
assert run.infections.shape == (len(gt) + n_days, n_places)
assert run.infections[-1, 0] > run.infections[-1, 1]
```

Next: [How to](how-to.md).
