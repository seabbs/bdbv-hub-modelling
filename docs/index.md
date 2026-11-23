# ![bdbv-hub-modelling](assets/logo.svg#only-light){ .title-logo width="336" height="44" } ![](assets/logo-dark.svg#only-dark){ .title-logo width="336" height="44" }

Nowcasts and forecasts of confirmed Bundibugyo virus disease cases for the BDBV Modeling Hub.
{ .page-subtitle }

We build models to nowcast and forecast cases for the [BDBV Modeling Hub](https://github.com/InsightNet-US/BDBV-Modeling-Hub).
The hub collects weekly forecasts of the 2026 Bundibugyo virus disease outbreak in the Democratic Republic of the Congo.
Teams forecast confirmed cases by week of symptom onset for the country, its provinces and its health zones.
Recent weeks are always incomplete because cases take time to be reported, so a nowcast estimates how many cases those weeks will finally hold.

We fine-tune a general-purpose forecasting model on simulated outbreaks so that it can nowcast and forecast cases from data that are still being reported.
To judge whether fine-tuning helps, we compare it with simpler baselines: a standard nowcasting method with a random walk for forecasts, and the random walk on its own.
We evaluate all of them on held-out simulations, on a German COVID-19 hospitalisation benchmark, and on the Ebola outbreak data.
We describe this work in a [paper](paper/main.html).

We submit two of these models to the hub each week.
One is the standard nowcasting method with its random walk.
The other is the fine-tuned model, once it is switched on for submission.

We also fit renewal models, which describe how new infections arise from earlier ones.
Our main renewal model takes its estimates of how fast the outbreak spreads from a larger model of the whole outbreak, [BVDOutbreakSize](https://github.com/epiforecasts/BVDOutbreakSize).
Two simpler versions, one without those estimates and one that fits each province on its own, show what each part adds.
Each week we submit the first of the three that passes our checks, trying the main model first.

We also submit weekly estimates of the reproduction number, Rt, from the renewal model we submit that week, by week of infection.
We do not recommend using them.
They are estimated from cases by date of symptom onset, including the onsets the hub imputes, so they are likely to react late and be out of step with transmission.
For Rt, we recommend our [BVDOutbreakSize estimates](https://epiforecasts.io/BVDOutbreakSize/stable/) instead.

We check every model by backtesting.
We rerun each model on the data as they stood at past dates and score its forecasts against what was reported later.
Every model is scored the same way, against the two random walk baselines, and compared with the other teams' forecasts.
The baselines and the fine-tuned model are also scored on the German benchmark.
The [backtest](backtest.md), [comparison with hub forecasts](comparison.md) and [German benchmark](germany.md) pages show the results.

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
|  | [![](assets/logos/nowcastbdbv/logo-mark.svg#only-light){ width="24" height="24" }![](assets/logos/nowcastbdbv/logo-mark-dark.svg#only-dark){ width="24" height="24" } **nowcastbdbv**](packages/index.md#nowcastbdbv) | The BDBV outbreak as a case study: the hub's data streams, health zones and delays. |
|  | [![](assets/logos/nowcastgermany/logo-mark.svg#only-light){ width="24" height="24" }![](assets/logos/nowcastgermany/logo-mark-dark.svg#only-dark){ width="24" height="24" } **nowcastgermany**](packages/index.md#nowcastgermany) | The German COVID-19 hospitalisation nowcasting hub as a case study. |

## Data credit

Surveillance data: Institut National de Santé Publique (INSP) and the Incident Management Support Team (IMST), Democratic Republic of the Congo.
Processed by ACCIDDA and the Kinshasa School of Public Health, University of Kinshasa, and distributed through the [BDBV Modeling Hub](https://github.com/InsightNet-US/BDBV-Modeling-Hub).
