# Joint-model-informed renewal

We compare renewal models fitted to the hub data on their own with renewal models informed by a larger joint model, [BVDOutbreakSize](https://github.com/epiforecasts/BVDOutbreakSize), through approximate Markov melding.
All are built on [PyRenew](https://github.com/CDCgov/PyRenew) and fit the hub's known and imputed onsets.
This section is for readers who want to know how the renewal models submitted to the hub work and where they fall short.

![Share of the last onset week reported by days since onset for each fitted renewal model, beside the random walk nowcast's.](../figures/renewal-delay.svg#only-light)
![Share of the last onset week reported by days since onset for each fitted renewal model, beside the random walk nowcast's.](../figures/renewal-delay-dark.svg#only-dark)

## Where to start

- To understand the models, start with [Methods](methods.md), which follows the data from the hub to the submission.
- To fit one, start with `bdbvhubrenewal fit` ([CLI](../development/cli.md)) or `nowcastrenewal.forecaster` in [Python](../reference/model.md).

## In this section

- [Methods](methods.md): how the models work, from data to submission, with their limitations.
- [Methods reference](../methods/model.md): the zone model, its single-patch province form, and the BVDOutbreakSize-informed model, with their equations.
- [Implementation](../methods/implementation.md): how the models are built in code, computed and sampled.

The code is in [nowcastrenewal](../packages/index.md#nowcastrenewal).

Next: [Methods](methods.md).
