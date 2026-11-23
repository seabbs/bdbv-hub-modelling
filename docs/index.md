# ![bdbv-hub-modelling](assets/logo.svg#only-light){ .title-logo width="336" height="44" } ![](assets/logo-dark.svg#only-dark){ .title-logo width="336" height="44" }

Nowcasts and forecasts of confirmed Bundibugyo virus disease cases for the BDBV Modeling Hub.
{ .page-subtitle }

The [BDBV Modeling Hub](https://github.com/InsightNet-US/BDBV-Modeling-Hub) collects weekly nowcasts and short-term forecasts of confirmed Bundibugyo virus disease cases in the Democratic Republic of the Congo.

This repository holds three projects.
The fine-tuned foundation project fine-tunes the Chronos-2 forecasting model on simulated outbreaks so that it can nowcast from reporting triangles.
The [paper](paper/main.html) and its [supporting information](paper/si.html) describe it.
It is evaluated on held-out simulations, on the German COVID-19 hospitalisation nowcasting hub and on the BDBV hub.
The joint-model-informed renewal project compares renewal models fitted to the hub data on their own with renewal models informed by a larger joint model, BVDOutbreakSize.
The hub submission project submits our models to the hub each week, backtests them and compares them with the other teams.
Every model is scored against the random walk and the random walk nowcast.

We submit three models.
`seabbs-rwnowcast` is the random walk nowcast.
`seabbs-renewal` is the first renewal model in a chain that passes its checks.
The chain starts with `renewalbvd`, the zone renewal model informed by the BVDOutbreakSize Rt estimates.
Its ablations are the fallbacks, `renewalsp`, which fits each province on its own, and then `renewal`, the zone model without BVDOutbreakSize.
`seabbs-nowcastfm` is the fine-tuned Chronos-2 model, submitted once it is opted in.

To run the paper's workflow and render the paper, use `task paper`.
The [workflow README](https://github.com/seabbs/bdbv-hub-modelling/blob/main/workflow/README.md) lists every step from simulation and fine-tuning to scoring.

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
