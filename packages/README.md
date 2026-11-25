# packages

The workspace's Python packages, each installable on its own.
Generic packages know no outbreak; the study packages hold the BDBV and German case studies.

In this project the workflow (`workflow/`), the paper (`paper/`), the docs site (`docs/`) and the CI workflows (`.github/workflows/`) all call these packages.
`bdbvsubmission` is the exception: it has its own environment and reaches the others only through the `bdbvhub` command.

Each package imports only those below it (the import-linter contract in `pyproject.toml`, checked in CI):

1. `bdbvhub`: the hub submission, the fit store and the reports.
2. `nowcastbdbv` and `nowcastgermany`: the two studies, which never import each other.
3. `nowcastfoundation`, then `nowcastrenewal`, then `nowcastbaselines`: the models.
4. `nowcasteval`, then `nowcastutils`: evaluation and the shared pieces.

Install all of them with `uv sync` at the repository root, or one with `uv sync --package <name>`.
Each folder has its README (install, an example, its tasks and its citation), its tests and its Taskfile.

<!-- packages table -->
<table>
<tr><th>Group</th><th>Package</th><th>What it is</th></tr>
<tr><td rowspan="2">Shared</td><td><a href="https://samabbott.co.uk/bdbv-hub-modelling/nowcastutils/"><picture><source media="(prefers-color-scheme: dark)" srcset="../assets/logos/nowcastutils/logo-mark-dark.svg"><img src="../assets/logos/nowcastutils/logo-mark.svg" alt="nowcastutils" width="24" height="24"></picture> <b>nowcastutils</b></a></td><td>Releases, snapshots, location hierarchies, the forecast task and sample trajectories.</td></tr>
<tr><td><a href="https://samabbott.co.uk/bdbv-hub-modelling/nowcasteval/"><picture><source media="(prefers-color-scheme: dark)" srcset="../assets/logos/nowcasteval/logo-mark-dark.svg"><img src="../assets/logos/nowcasteval/logo-mark.svg" alt="nowcasteval" width="24" height="24"></picture> <b>nowcasteval</b></a></td><td>Scores, backtests and trajectory pairing.</td></tr>
<tr><td rowspan="3">Models</td><td><a href="https://samabbott.co.uk/bdbv-hub-modelling/nowcastbaselines/"><picture><source media="(prefers-color-scheme: dark)" srcset="../assets/logos/nowcastbaselines/logo-mark-dark.svg"><img src="../assets/logos/nowcastbaselines/logo-mark.svg" alt="nowcastbaselines" width="24" height="24"></picture> <b>nowcastbaselines</b></a></td><td>The random walk and the random walk nowcast, the baselines.</td></tr>
<tr><td><a href="https://samabbott.co.uk/bdbv-hub-modelling/nowcastfoundation/"><picture><source media="(prefers-color-scheme: dark)" srcset="../assets/logos/nowcastfoundation/logo-mark-dark.svg"><img src="../assets/logos/nowcastfoundation/logo-mark.svg" alt="nowcastfoundation" width="24" height="24"></picture> <b>nowcastfoundation</b></a></td><td>Chronos-2 forecasts, the simulation grid and fine-tuning on simulated outbreaks.</td></tr>
<tr><td><a href="https://samabbott.co.uk/bdbv-hub-modelling/nowcastrenewal/"><picture><source media="(prefers-color-scheme: dark)" srcset="../assets/logos/nowcastrenewal/logo-mark-dark.svg"><img src="../assets/logos/nowcastrenewal/logo-mark.svg" alt="nowcastrenewal" width="24" height="24"></picture> <b>nowcastrenewal</b></a></td><td>The renewal models and the joint model that simulates outbreaks.</td></tr>
<tr><td rowspan="3">Studies</td><td><a href="https://samabbott.co.uk/bdbv-hub-modelling/bdbvhub/"><picture><source media="(prefers-color-scheme: dark)" srcset="../assets/logos/bdbvhub/logo-mark-dark.svg"><img src="../assets/logos/bdbvhub/logo-mark.svg" alt="bdbvhub" width="24" height="24"></picture> <b>bdbvhub</b></a></td><td>The BDBV hub's data, the fit store, the hub file and the submission.</td></tr>
<tr><td><a href="https://samabbott.co.uk/bdbv-hub-modelling/nowcastbdbv/"><picture><source media="(prefers-color-scheme: dark)" srcset="../assets/logos/nowcastbdbv/logo-mark-dark.svg"><img src="../assets/logos/nowcastbdbv/logo-mark.svg" alt="nowcastbdbv" width="24" height="24"></picture> <b>nowcastbdbv</b></a></td><td>The BDBV outbreak as a case study: the hub's data streams, health zones and delays.</td></tr>
<tr><td><a href="https://samabbott.co.uk/bdbv-hub-modelling/nowcastgermany/"><picture><source media="(prefers-color-scheme: dark)" srcset="../assets/logos/nowcastgermany/logo-mark-dark.svg"><img src="../assets/logos/nowcastgermany/logo-mark.svg" alt="nowcastgermany" width="24" height="24"></picture> <b>nowcastgermany</b></a></td><td>The German COVID-19 hospitalisation nowcasting hub as a case study.</td></tr>
</table>
<!-- /packages table -->

Back to the [repository README](../README.md).
