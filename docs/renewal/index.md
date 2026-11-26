# Joint-model-informed renewal

We compare renewal models fitted to the hub data on their own with renewal models informed by a larger joint model, [BVDOutbreakSize](https://github.com/epiforecasts/BVDOutbreakSize), through approximate Markov melding.
All are built on [PyRenew](https://github.com/CDCgov/PyRenew) and fit the hub's known and imputed onsets.
This section is for readers who want to know how the renewal models submitted to the hub work and where they fall short.

![Share of the last onset week reported by days since onset for each fitted renewal model, beside the random walk nowcast's.](../figures/renewal-delay.svg#only-light)
![Share of the last onset week reported by days since onset for each fitted renewal model, beside the random walk nowcast's.](../figures/renewal-delay-dark.svg#only-dark)

## Where to start

- To understand the models, start with the paper's [supporting information](../paper/si.html#renewal-model), linked from [Methods](methods.md).
- To fit one, start with `bdbvhub fit` ([CLI](../development/cli.md)) or `nowcastrenewal.forecaster` in [Python](../nowcastrenewal/reference/).

## In this section

- [Methods](methods.md): what the models are, with a link to their methods in the paper's supporting information, and how they are submitted.
- [Methods reference](../methods/model.md): what the fits give the site and the hub beyond the supporting information, such as the weekly Rt target.
- [Implementation](../methods/implementation.md): how the models are built in code, computed and sampled.

The code is in [nowcastrenewal](../packages/index.md#nowcastrenewal).

Next: [Methods](methods.md).
