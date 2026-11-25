# How it is used in the workflow

The fitted renewal models are the renewal track of the hub submission: `renewalbvd`, `renewalsp` and `renewal`, tried in that order.
The [Submit to the BDBV hub](https://github.com/seabbs/bdbv-hub-modelling/blob/main/.github/workflows/submit.yml) and [Fit candidates](https://github.com/seabbs/bdbv-hub-modelling/blob/main/.github/workflows/fit-candidates.yml) workflows fit them each round through the `bdbvhub` command.
The [Fit store](https://github.com/seabbs/bdbv-hub-modelling/blob/main/.github/workflows/fit-store.yml) workflow fits them once per hub release and stores the fits, which the submission, the backtests and the site read.
A change to this package's code refits every origin.

The simulator makes the outbreaks Chronos-2 is fine-tuned on.
The workflow's [simulate rules](https://github.com/seabbs/bdbv-hub-modelling/blob/main/workflow/rules/simulate.smk) run it through `nowcastfoundation.simulator`, which calls `nowcastrenewal.simulation.outbreak`.

How the models work: [Methods](https://samabbott.co.uk/bdbv-hub-modelling/renewal/methods/).
