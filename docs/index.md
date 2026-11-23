# ![bdbv-hub-modelling](assets/logo.svg#only-light){ .title-logo width="336" height="44" } ![](assets/logo-dark.svg#only-dark){ .title-logo width="336" height="44" }

Nowcasts and forecasts of confirmed Bundibugyo virus disease cases for the BDBV Modeling Hub.
{ .page-subtitle }

This repository holds our modelling for the [BDBV Modeling Hub](https://github.com/InsightNet-US/BDBV-Modeling-Hub).
The hub collects weekly nowcasts and short-term forecasts of confirmed Bundibugyo virus disease cases in the 2026 outbreak in the Democratic Republic of the Congo.
Teams submit sample trajectories of cases by week of symptom onset, nationally, by province and by health zone.

We fine-tune the Chronos-2 forecasting model on simulated outbreaks so that it can nowcast from reporting triangles.
Our baselines in [`nowcastbaselines`](nowcastbaselines/) are `randomwalknowcast`, a JAX port of baselinenowcast ([Johnson et al. 2025](https://doi.org/10.12688/wellcomeopenres.25027.2)) with a random walk on top, and the random walk, `randomwalk`.
Both belong to one [paper](paper/main.html), with `randomwalknowcast` as its reference model.
The paper evaluates them on held-out simulations, the German COVID-19 hospitalisation nowcasting hub and the BDBV hub.

We submit two models from this work.
`seabbs-rwnowcast` is `randomwalknowcast`, submitted each week.
`seabbs-nowcastfm` is the fine-tuned Chronos-2 model, submitted once it is opted in.

Our renewal models are linked to a larger joint model, [BVDOutbreakSize](https://github.com/epiforecasts/BVDOutbreakSize).
`renewalbvd` is the zone renewal model informed by the BVDOutbreakSize Rt estimates.
Its ablations are `renewalsp`, which fits each province on its own, and `renewal`, the zone model without BVDOutbreakSize.
We submit them as `seabbs-renewal`, the first of `renewalbvd`, `renewalsp` and `renewal`, in that order, that passes its checks.

We backtest every model on past hub releases, forecasting from each vintage and scoring against what later releases reported.
The baselines and the foundation models are also run on the German nowcast benchmark.
Every model is scored the same way, against the random walk and the random walk nowcast, and compared with the hub teams' forecasts.
The [backtest](backtest.md), [comparison with hub forecasts](comparison.md) and [German nowcast benchmark](germany.md) pages show the results.

## Quick links

The three projects:

<div class="grid cards" markdown>

-   [![](assets/logos/fine-tuned-foundation/logo-mark.svg#only-light){ width="32" height="32" }![](assets/logos/fine-tuned-foundation/logo-mark-dark.svg#only-dark){ width="32" height="32" } **Fine-tuned foundation**](foundation/index.md)

    Chronos-2 fine-tuned on simulated outbreaks to nowcast from reporting triangles, with its paper.

-   [![](assets/logos/joint-model-informed-renewal/logo-mark.svg#only-light){ width="32" height="32" }![](assets/logos/joint-model-informed-renewal/logo-mark-dark.svg#only-dark){ width="32" height="32" } **Joint-model-informed renewal**](renewal/index.md)

    Renewal models fitted on their own and informed by BVDOutbreakSize.

-   [![](assets/logos/hub-submission/logo-mark.svg#only-light){ width="32" height="32" }![](assets/logos/hub-submission/logo-mark-dark.svg#only-dark){ width="32" height="32" } **Hub submission**](hub/index.md)

    The weekly submission to the BDBV Modeling Hub, its backtest and the comparison with the hub teams.

</div>

The packages they are built from:

| Group | Package | What it is |
|---|---|---|
| Shared | [![](assets/logos/nowcastutils/logo-mark.svg#only-light){ width="24" height="24" }![](assets/logos/nowcastutils/logo-mark-dark.svg#only-dark){ width="24" height="24" } **nowcastutils**](nowcastutils/) | Releases, snapshots, location hierarchies, the forecast task and sample trajectories. |
|  | [![](assets/logos/nowcasteval/logo-mark.svg#only-light){ width="24" height="24" }![](assets/logos/nowcasteval/logo-mark-dark.svg#only-dark){ width="24" height="24" } **nowcasteval**](nowcasteval/) | Scores, backtests and trajectory pairing. |
| Models | [![](assets/logos/nowcastbaselines/logo-mark.svg#only-light){ width="24" height="24" }![](assets/logos/nowcastbaselines/logo-mark-dark.svg#only-dark){ width="24" height="24" } **nowcastbaselines**](nowcastbaselines/) | The random walk and the random walk nowcast, the baselines. |
|  | [![](assets/logos/nowcastfoundation/logo-mark.svg#only-light){ width="24" height="24" }![](assets/logos/nowcastfoundation/logo-mark-dark.svg#only-dark){ width="24" height="24" } **nowcastfoundation**](packages/index.md#nowcastfoundation) | Chronos-2 forecasts, the simulation grid and fine-tuning on simulated outbreaks. |
|  | [![](assets/logos/nowcastrenewal/logo-mark.svg#only-light){ width="24" height="24" }![](assets/logos/nowcastrenewal/logo-mark-dark.svg#only-dark){ width="24" height="24" } **nowcastrenewal**](packages/index.md#nowcastrenewal) | The renewal models and the joint model that simulates outbreaks. |
| Studies | [![](assets/logos/bdbvhub/logo-mark.svg#only-light){ width="24" height="24" }![](assets/logos/bdbvhub/logo-mark-dark.svg#only-dark){ width="24" height="24" } **bdbvhub**](packages/index.md#bdbvhub) | The BDBV hub's data, the fit store, the hub file and the submission. |
|  | [![](assets/logos/nowcastgermany/logo-mark.svg#only-light){ width="24" height="24" }![](assets/logos/nowcastgermany/logo-mark-dark.svg#only-dark){ width="24" height="24" } **nowcastgermany**](packages/index.md#nowcastgermany) | The German COVID-19 hospitalisation nowcasting hub as a case study. |

## Data credit

Surveillance data: Institut National de Santé Publique (INSP) and the Incident Management Support Team (IMST), Democratic Republic of the Congo.
Processed by ACCIDDA and the Kinshasa School of Public Health, University of Kinshasa, and distributed through the [BDBV Modeling Hub](https://github.com/InsightNet-US/BDBV-Modeling-Hub).
