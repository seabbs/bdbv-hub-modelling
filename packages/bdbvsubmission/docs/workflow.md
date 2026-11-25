# How it is used in the workflow

The Submit to the BDBV hub workflow ([`submit.yml`](https://github.com/seabbs/bdbv-hub-modelling/blob/main/.github/workflows/submit.yml)) runs it each round.
Its steps are the shell scripts in [`steps/`](https://github.com/seabbs/bdbv-hub-modelling/tree/main/packages/bdbvsubmission/steps), which call the Python modules here.
`detect_new_vintage` decides whether a run should submit, and `tracks` and `policy` give the hub models and the fallback chain.
Fit candidates ([`fit-candidates.yml`](https://github.com/seabbs/bdbv-hub-modelling/blob/main/.github/workflows/fit-candidates.yml)) reads each candidate's fit from the fit store through the `bdbvhub` command.
That command is the package's only link to the rest of the repository.
`select_candidate` then walks the chain, `build_pr_body` writes the pull request and `open_or_update_pr` opens it on the hub.
`release_rounds` makes a GitHub release of each submitted round, and `submissions` writes the table of what each round submitted, which the evaluation and the paper read.
[INTEGRATION.md](https://github.com/seabbs/bdbv-hub-modelling/blob/main/packages/bdbvsubmission/INTEGRATION.md) describes the weekly run step by step, and [SETUP.md](https://github.com/seabbs/bdbv-hub-modelling/blob/main/packages/bdbvsubmission/SETUP.md) how to set it up on a fork.
