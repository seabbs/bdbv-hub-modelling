# nowcastrenewal

Renewal models of an outbreak, built on [PyRenew](https://github.com/CDCgov/PyRenew) (JAX, NumPyro).
It holds the renewal models fitted to the hub's data and the joint model that simulates outbreaks for fine-tuning. The two share one set of components.

## Start here

- To fit or forecast with a renewal model, read [`forecaster`](src/nowcastrenewal/forecaster.py), then [`model`](src/nowcastrenewal/model/) for the model itself.
- To see how a simulated outbreak is made, read [`simulation.outbreak`](src/nowcastrenewal/simulation/outbreak.py) (`simulate_outbreak`).

## Modules

- [`components`](src/nowcastrenewal/components/): pieces shared by the fitted models and the simulator: pmfs, the renewal, mixing, walks, noise and reference events.
- [`model`](src/nowcastrenewal/model/): the fitted renewal model, with provinces or health zones as patches.
- [`forecaster`](src/nowcastrenewal/forecaster.py): the renewal variants as forecasters, with their fitting, forecasts and diagnostics.
- [`zones`](src/nowcastrenewal/zones.py): the renewal variants fitted on health zones.
- [`bvd`](src/nowcastrenewal/bvd.py): the BVDOutbreakSize reproduction number estimates that the `bvd` variants read.
- [`latent`](src/nowcastrenewal/latent/): the simulator's latent side: R paths, imports, mixing, starting infections and infections.
- [`reporting`](src/nowcastrenewal/reporting/): the simulator's reporting side: delays, effects, the expected triangle, observation noise and revisions.
- [`simulation`](src/nowcastrenewal/simulation/): the simulator's joint model, from latent infections to reported counts.

## Next

- How the fitted models work: [Methods](https://samabbott.co.uk/bdbv-hub-modelling/renewal/methods/) and the [renewal model reference](https://samabbott.co.uk/bdbv-hub-modelling/methods/model/).
- The simulator's place in the workflow: [simulate](../../workflow/README.md#1-simulate-rulessimulatesmk), with draws from `nowcastfoundation.simulator`.

## Install

From a clone of the repository:

```sh
uv sync --package nowcastrenewal             # this package and the ones it needs
```

The renewal model itself needs PyRenew, the `model` extra: `uv sync --package nowcastrenewal --extra model`.
Its live checks against epinowcast and primarycensored need R (CI's r-live job installs it).

## Example

```python
import nowcastrenewal.zones  # registers the renewal variants
from nowcastutils.base import get_model

model = get_model("renewal")
forecasts = model.forecast(task)  # fits with NUTS, then forecasts
```

How the models work: [Methods](https://samabbott.co.uk/bdbv-hub-modelling/renewal/methods/).

## Tasks

`task nowcastrenewal:test` runs its tests (`task --list-all` lists the rest).

## Citation

Cite it with [CITATION.cff](CITATION.cff); the repository as a whole has [its own](../../CITATION.cff).

```bibtex
@software{abbott_nowcastrenewal,
  author = {Abbott, Sam},
  title = {{nowcastrenewal: renewal-model nowcasts and forecasts: the PyRenew renewal model, its forecasters and the BVDOutbreakSize inputs}},
  year = {2026},
  url = {https://github.com/seabbs/bdbv-hub-modelling/tree/main/packages/nowcastrenewal}
}
```
