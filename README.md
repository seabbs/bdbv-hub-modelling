<h1>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <img src="assets/logo.svg" alt="bdbv-hub-modelling" width="336" height="44">
  </picture>
</h1>

**Nowcasts and forecasts for the BDBV Modeling Hub, and the models and evaluation behind them.**

We build models to nowcast and forecast cases for the [BDBV Modeling Hub](https://github.com/InsightNet-US/BDBV-Modeling-Hub).
The hub collects weekly forecasts of the 2026 Bundibugyo virus disease outbreak in the Democratic Republic of the Congo.
Teams forecast confirmed cases by week of symptom onset for the country, its provinces and its health zones.
Recent weeks are always incomplete because cases take time to be reported, so a nowcast estimates how many cases those weeks will finally hold.

We fine-tune a general-purpose forecasting model on simulated outbreaks so that it can nowcast and forecast cases from data that are still being reported.
To judge whether fine-tuning helps, we compare it with simpler baselines: a standard nowcasting method with a random walk for forecasts, and the random walk on its own.
We evaluate all of them on held-out simulations, on a German COVID-19 hospitalisation benchmark, and on the Ebola outbreak data.
We describe this work in a paper.

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
The [backtest](https://samabbott.co.uk/bdbv-hub-modelling/backtest/), [comparison with hub forecasts](https://samabbott.co.uk/bdbv-hub-modelling/comparison/) and [German benchmark](https://samabbott.co.uk/bdbv-hub-modelling/germany/) pages show the results.

## Quick links

The three projects:

<table><tr><td align="center" valign="top" width="33%"><a href="https://samabbott.co.uk/bdbv-hub-modelling/foundation/"><picture><source media="(prefers-color-scheme: dark)" srcset="assets/logos/fine-tuned-foundation/logo-mark-dark.svg"><img src="assets/logos/fine-tuned-foundation/logo-mark.svg" alt="Fine-tuned foundation" width="48" height="48"></picture><br><b>Fine-tuned foundation</b></a><br>Chronos-2 fine-tuned on simulated outbreaks to nowcast from reporting triangles, with its paper.</td><td align="center" valign="top" width="33%"><a href="https://samabbott.co.uk/bdbv-hub-modelling/renewal/"><picture><source media="(prefers-color-scheme: dark)" srcset="assets/logos/joint-model-informed-renewal/logo-mark-dark.svg"><img src="assets/logos/joint-model-informed-renewal/logo-mark.svg" alt="Joint-model-informed renewal" width="48" height="48"></picture><br><b>Joint-model-informed renewal</b></a><br>Renewal models fitted on their own and informed by BVDOutbreakSize.</td><td align="center" valign="top" width="33%"><a href="https://samabbott.co.uk/bdbv-hub-modelling/hub/"><picture><source media="(prefers-color-scheme: dark)" srcset="assets/logos/hub-submission/logo-mark-dark.svg"><img src="assets/logos/hub-submission/logo-mark.svg" alt="Hub submission" width="48" height="48"></picture><br><b>Hub submission</b></a><br>The weekly submission to the BDBV Modeling Hub, its backtest and the comparison with the hub teams.</td></tr></table>

The packages they are built from:

<table>
<tr><th>Group</th><th>Package</th><th>What it is</th></tr>
<tr><td rowspan="2">Shared</td><td><a href="https://samabbott.co.uk/bdbv-hub-modelling/nowcastutils/"><picture><source media="(prefers-color-scheme: dark)" srcset="assets/logos/nowcastutils/logo-mark-dark.svg"><img src="assets/logos/nowcastutils/logo-mark.svg" alt="nowcastutils" width="24" height="24"></picture> <b>nowcastutils</b></a></td><td>Releases, snapshots, location hierarchies, the forecast task and sample trajectories.</td></tr>
<tr><td><a href="https://samabbott.co.uk/bdbv-hub-modelling/nowcasteval/"><picture><source media="(prefers-color-scheme: dark)" srcset="assets/logos/nowcasteval/logo-mark-dark.svg"><img src="assets/logos/nowcasteval/logo-mark.svg" alt="nowcasteval" width="24" height="24"></picture> <b>nowcasteval</b></a></td><td>Scores, backtests and trajectory pairing.</td></tr>
<tr><td rowspan="3">Models</td><td><a href="https://samabbott.co.uk/bdbv-hub-modelling/nowcastbaselines/"><picture><source media="(prefers-color-scheme: dark)" srcset="assets/logos/nowcastbaselines/logo-mark-dark.svg"><img src="assets/logos/nowcastbaselines/logo-mark.svg" alt="nowcastbaselines" width="24" height="24"></picture> <b>nowcastbaselines</b></a></td><td>The random walk and the random walk nowcast, the baselines.</td></tr>
<tr><td><a href="https://samabbott.co.uk/bdbv-hub-modelling/packages/#nowcastfoundation"><picture><source media="(prefers-color-scheme: dark)" srcset="assets/logos/nowcastfoundation/logo-mark-dark.svg"><img src="assets/logos/nowcastfoundation/logo-mark.svg" alt="nowcastfoundation" width="24" height="24"></picture> <b>nowcastfoundation</b></a></td><td>Chronos-2 forecasts, the simulation grid and fine-tuning on simulated outbreaks.</td></tr>
<tr><td><a href="https://samabbott.co.uk/bdbv-hub-modelling/packages/#nowcastrenewal"><picture><source media="(prefers-color-scheme: dark)" srcset="assets/logos/nowcastrenewal/logo-mark-dark.svg"><img src="assets/logos/nowcastrenewal/logo-mark.svg" alt="nowcastrenewal" width="24" height="24"></picture> <b>nowcastrenewal</b></a></td><td>The renewal models and the joint model that simulates outbreaks.</td></tr>
<tr><td rowspan="3">Studies</td><td><a href="https://samabbott.co.uk/bdbv-hub-modelling/packages/#bdbvhub"><picture><source media="(prefers-color-scheme: dark)" srcset="assets/logos/bdbvhub/logo-mark-dark.svg"><img src="assets/logos/bdbvhub/logo-mark.svg" alt="bdbvhub" width="24" height="24"></picture> <b>bdbvhub</b></a></td><td>The BDBV hub's data, the fit store, the hub file and the submission.</td></tr>
<tr><td><a href="https://samabbott.co.uk/bdbv-hub-modelling/packages/#nowcastbdbv"><picture><source media="(prefers-color-scheme: dark)" srcset="assets/logos/nowcastbdbv/logo-mark-dark.svg"><img src="assets/logos/nowcastbdbv/logo-mark.svg" alt="nowcastbdbv" width="24" height="24"></picture> <b>nowcastbdbv</b></a></td><td>The BDBV outbreak as a case study: the hub's data streams, health zones and delays.</td></tr>
<tr><td><a href="https://samabbott.co.uk/bdbv-hub-modelling/packages/#nowcastgermany"><picture><source media="(prefers-color-scheme: dark)" srcset="assets/logos/nowcastgermany/logo-mark-dark.svg"><img src="assets/logos/nowcastgermany/logo-mark.svg" alt="nowcastgermany" width="24" height="24"></picture> <b>nowcastgermany</b></a></td><td>The German COVID-19 hospitalisation nowcasting hub as a case study.</td></tr>
</table>

## Status

[![Docs](https://img.shields.io/badge/docs-online-blue.svg)](https://samabbott.co.uk/bdbv-hub-modelling/)
[![Tests](https://github.com/seabbs/bdbv-hub-modelling/actions/workflows/tests.yml/badge.svg)](https://github.com/seabbs/bdbv-hub-modelling/actions/workflows/tests.yml)
[![Coverage](https://codecov.io/gh/seabbs/bdbv-hub-modelling/branch/main/graph/badge.svg?flag=package)](https://codecov.io/gh/seabbs/bdbv-hub-modelling?flags%5B0%5D=package)
[![DOI](https://zenodo.org/badge/1435982169.svg)](https://doi.org/10.5281/zenodo.23942616)
[![licence: MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)

<details><summary>Status of each package and component</summary>

| Component | Tests | Lint | Coverage |
| --- | --- | --- | --- |
| [`nowcastutils`](packages/nowcastutils) | [![Tests](https://github.com/seabbs/bdbv-hub-modelling/actions/workflows/tests.yml/badge.svg)](https://github.com/seabbs/bdbv-hub-modelling/actions/workflows/tests.yml) | [![Tests](https://github.com/seabbs/bdbv-hub-modelling/actions/workflows/tests.yml/badge.svg)](https://github.com/seabbs/bdbv-hub-modelling/actions/workflows/tests.yml) | [![package coverage](https://codecov.io/gh/seabbs/bdbv-hub-modelling/branch/main/graph/badge.svg?flag=package)](https://codecov.io/gh/seabbs/bdbv-hub-modelling?flags%5B0%5D=package) |
| [`nowcasteval`](packages/nowcasteval) | [![Tests](https://github.com/seabbs/bdbv-hub-modelling/actions/workflows/tests.yml/badge.svg)](https://github.com/seabbs/bdbv-hub-modelling/actions/workflows/tests.yml) | [![Tests](https://github.com/seabbs/bdbv-hub-modelling/actions/workflows/tests.yml/badge.svg)](https://github.com/seabbs/bdbv-hub-modelling/actions/workflows/tests.yml) | [![package coverage](https://codecov.io/gh/seabbs/bdbv-hub-modelling/branch/main/graph/badge.svg?flag=package)](https://codecov.io/gh/seabbs/bdbv-hub-modelling?flags%5B0%5D=package) |
| [`nowcastbaselines`](packages/nowcastbaselines) | [![Tests](https://github.com/seabbs/bdbv-hub-modelling/actions/workflows/tests.yml/badge.svg)](https://github.com/seabbs/bdbv-hub-modelling/actions/workflows/tests.yml) | [![Tests](https://github.com/seabbs/bdbv-hub-modelling/actions/workflows/tests.yml/badge.svg)](https://github.com/seabbs/bdbv-hub-modelling/actions/workflows/tests.yml) | [![package coverage](https://codecov.io/gh/seabbs/bdbv-hub-modelling/branch/main/graph/badge.svg?flag=package)](https://codecov.io/gh/seabbs/bdbv-hub-modelling?flags%5B0%5D=package) |
| [`nowcastfoundation`](packages/nowcastfoundation) | [![Tests](https://github.com/seabbs/bdbv-hub-modelling/actions/workflows/tests.yml/badge.svg)](https://github.com/seabbs/bdbv-hub-modelling/actions/workflows/tests.yml) | [![Tests](https://github.com/seabbs/bdbv-hub-modelling/actions/workflows/tests.yml/badge.svg)](https://github.com/seabbs/bdbv-hub-modelling/actions/workflows/tests.yml) | [![package coverage](https://codecov.io/gh/seabbs/bdbv-hub-modelling/branch/main/graph/badge.svg?flag=package)](https://codecov.io/gh/seabbs/bdbv-hub-modelling?flags%5B0%5D=package) |
| [`nowcastrenewal`](packages/nowcastrenewal) | [![Tests](https://github.com/seabbs/bdbv-hub-modelling/actions/workflows/tests.yml/badge.svg)](https://github.com/seabbs/bdbv-hub-modelling/actions/workflows/tests.yml) | [![Tests](https://github.com/seabbs/bdbv-hub-modelling/actions/workflows/tests.yml/badge.svg)](https://github.com/seabbs/bdbv-hub-modelling/actions/workflows/tests.yml) | [![package coverage](https://codecov.io/gh/seabbs/bdbv-hub-modelling/branch/main/graph/badge.svg?flag=package)](https://codecov.io/gh/seabbs/bdbv-hub-modelling?flags%5B0%5D=package) |
| [`bdbvhub`](packages/bdbvhub) | [![Tests](https://github.com/seabbs/bdbv-hub-modelling/actions/workflows/tests.yml/badge.svg)](https://github.com/seabbs/bdbv-hub-modelling/actions/workflows/tests.yml) | [![Tests](https://github.com/seabbs/bdbv-hub-modelling/actions/workflows/tests.yml/badge.svg)](https://github.com/seabbs/bdbv-hub-modelling/actions/workflows/tests.yml) | [![package coverage](https://codecov.io/gh/seabbs/bdbv-hub-modelling/branch/main/graph/badge.svg?flag=package)](https://codecov.io/gh/seabbs/bdbv-hub-modelling?flags%5B0%5D=package) |
| [`nowcastbdbv`](packages/nowcastbdbv) | [![Tests](https://github.com/seabbs/bdbv-hub-modelling/actions/workflows/tests.yml/badge.svg)](https://github.com/seabbs/bdbv-hub-modelling/actions/workflows/tests.yml) | [![Tests](https://github.com/seabbs/bdbv-hub-modelling/actions/workflows/tests.yml/badge.svg)](https://github.com/seabbs/bdbv-hub-modelling/actions/workflows/tests.yml) | [![package coverage](https://codecov.io/gh/seabbs/bdbv-hub-modelling/branch/main/graph/badge.svg?flag=package)](https://codecov.io/gh/seabbs/bdbv-hub-modelling?flags%5B0%5D=package) |
| [`nowcastgermany`](packages/nowcastgermany) | [![Tests](https://github.com/seabbs/bdbv-hub-modelling/actions/workflows/tests.yml/badge.svg)](https://github.com/seabbs/bdbv-hub-modelling/actions/workflows/tests.yml) | [![Tests](https://github.com/seabbs/bdbv-hub-modelling/actions/workflows/tests.yml/badge.svg)](https://github.com/seabbs/bdbv-hub-modelling/actions/workflows/tests.yml) | [![package coverage](https://codecov.io/gh/seabbs/bdbv-hub-modelling/branch/main/graph/badge.svg?flag=package)](https://codecov.io/gh/seabbs/bdbv-hub-modelling?flags%5B0%5D=package) |
| Submission CI (`ci/`) | [![Tests](https://github.com/seabbs/bdbv-hub-modelling/actions/workflows/tests.yml/badge.svg)](https://github.com/seabbs/bdbv-hub-modelling/actions/workflows/tests.yml) | [![Tests](https://github.com/seabbs/bdbv-hub-modelling/actions/workflows/tests.yml/badge.svg)](https://github.com/seabbs/bdbv-hub-modelling/actions/workflows/tests.yml) | [![ci coverage](https://codecov.io/gh/seabbs/bdbv-hub-modelling/branch/main/graph/badge.svg?flag=ci)](https://codecov.io/gh/seabbs/bdbv-hub-modelling?flags%5B0%5D=ci) |
| Workflow (`workflow/`) | [![Tests](https://github.com/seabbs/bdbv-hub-modelling/actions/workflows/tests.yml/badge.svg)](https://github.com/seabbs/bdbv-hub-modelling/actions/workflows/tests.yml) | [![Tests](https://github.com/seabbs/bdbv-hub-modelling/actions/workflows/tests.yml/badge.svg)](https://github.com/seabbs/bdbv-hub-modelling/actions/workflows/tests.yml) | |
| Paper (`paper/`) | [![Tests](https://github.com/seabbs/bdbv-hub-modelling/actions/workflows/tests.yml/badge.svg)](https://github.com/seabbs/bdbv-hub-modelling/actions/workflows/tests.yml) | [![Tests](https://github.com/seabbs/bdbv-hub-modelling/actions/workflows/tests.yml/badge.svg)](https://github.com/seabbs/bdbv-hub-modelling/actions/workflows/tests.yml) | |
| Docs site | [![Site](https://github.com/seabbs/bdbv-hub-modelling/actions/workflows/pages.yml/badge.svg)](https://github.com/seabbs/bdbv-hub-modelling/actions/workflows/pages.yml) | | |

</details>

## Installation

Install the packages and their dependencies with [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/seabbs/bdbv-hub-modelling
cd bdbv-hub-modelling
uv sync
```

## Usage

The [Getting started](https://samabbott.co.uk/bdbv-hub-modelling/getting-started/) page forecasts with both baselines, `randomwalknowcast` and `randomwalk`, loads the stored `renewal` fit at the same origin, and plots and scores all three in Python.

## Data

The surveillance data are from the Institut National de Santé Publique (INSP) and the Incident Management Support Team (IMST), Democratic Republic of the Congo.
They are processed by the Atlantic Coast Center for Infectious Disease Dynamics and Analytics (ACCIDDA) and the Kinshasa School of Public Health, University of Kinshasa, and distributed through the [BDBV Modeling Hub](https://github.com/InsightNet-US/BDBV-Modeling-Hub).

## Contributing

Contributions are welcome.
See [CONTRIBUTING.md](CONTRIBUTING.md).

## Citing

[![DOI](https://zenodo.org/badge/1435982169.svg)](https://doi.org/10.5281/zenodo.23942616)

Abbott S (2026). bdbv-hub-modelling: renewal-model nowcasts and forecasts for the BDBV Modeling Hub. Zenodo. <https://doi.org/10.5281/zenodo.23942616>.

```bibtex
@software{abbott_bdbv_hub_modelling,
  author = {Abbott, Sam},
  title = {{bdbv-hub-modelling}: renewal-model nowcasts and forecasts for the {BDBV} Modeling Hub},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.23942616},
  url = {https://doi.org/10.5281/zenodo.23942616}
}
```

The DOI covers all versions, and each release has its own DOI on Zenodo.
The metadata are in [CITATION.cff](CITATION.cff), which GitHub shows as the "Cite this repository" button.
That entry is the repository as a whole; each subproject has its own citation, with its own authors: [bdbvhub](packages/bdbvhub/CITATION.cff), [nowcastbaselines](packages/nowcastbaselines/CITATION.cff), [nowcastbdbv](packages/nowcastbdbv/CITATION.cff), [nowcasteval](packages/nowcasteval/CITATION.cff), [nowcastfoundation](packages/nowcastfoundation/CITATION.cff), [nowcastgermany](packages/nowcastgermany/CITATION.cff), [nowcastrenewal](packages/nowcastrenewal/CITATION.cff), [nowcastutils](packages/nowcastutils/CITATION.cff) and the [paper](paper/CITATION.cff).

## Funding

This study is funded by the National Institute for Health and Care Research (NIHR) Health Protection Research Unit in Health Analytics & Modelling, a partnership between the UK Health Security Agency, Imperial College London and the London School of Hygiene & Tropical Medicine (grant code NIHR207404).
The views expressed are those of the author(s) and not necessarily those of the NIHR, UK Health Security Agency or the Department of Health and Social Care.

## Licence

The code is under the MIT licence, see [LICENSE](LICENSE).
