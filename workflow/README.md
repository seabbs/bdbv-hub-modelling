# The workflow

One Snakemake workflow simulates outbreaks, fine-tunes Chronos-2 on them, publishes the checkpoints, scores them and builds the paper.
The paper's rules (`paper/workflow/Snakefile`) are a module of it, with their settings in `paper/workflow/config.yaml`.
This page walks through it in order, with the command for each piece, from a whole target down to one file.

## Running it

High level, one Taskfile task per target:

| Task | Target | Does | Writes outside this machine |
|---|---|---|---|
| `task nowcast-workflow:all` | `all` | simulate, fine-tune and score | no |
| `task nowcast-foundation:simulate ROUNDS="4 5"` | `simulate` | simulation rounds, with QC and manifests | no |
| `task nowcast-foundation:release ROUNDS=4` | `release` | each batch to its `sims-r<N>` release | GitHub releases |
| `task nowcast-foundation:push ROUNDS="19 18"` | `push` | a helping host's batches, pushed to the rounds host | the rounds host |
| `task nowcast-foundation:finetune` | `finetune` | every checkpoint prefix of the run | no |
| `task nowcast-foundation:publish` | `publish` | verify and release each checkpoint | GitHub releases |
| `task nowcast-foundation:score` | `score` | score each released checkpoint | no |
| `task nowcast-foundation:land` | `land` | commit the score tables to `evidence/` and push | git push |
| `task paper` | `development` or `paper`, then `render` | the paper (section 7) | no |

Pass Snakemake options after `--`: `task nowcast-foundation:score -- -n` is a dry run, and a file path builds just that output.
Low level, any output file, for example:

```sh
uv run --group paper snakemake <ROOT>/rounds/r34/sims/counts-33400.csv.gz  # one batch
uv run --group paper snakemake <ROOT>/runs/runa/train/n800/settings.json  # one checkpoint
uv run --group paper snakemake <SCORE_CACHE>/germany-days/slots-<hash>.done    # one checkpoint's German scores
```

The settings in `workflow/config.yaml`, with any `--configfile` or `--config` on top, are checked against `workflow/schemas/config.schema.yaml` when the workflow loads, so a misspelt or unknown setting is an error before any job runs.
A run in `runs` takes any `finetune` setting and no other.

Helpers: `task nowcast-workflow:rulegraph` draws the rule graph, `task nowcast-workflow:progress` lists what exists and what would run, `task nowcast-workflow:report` writes Snakemake's HTML report, and `task nowcast-workflow:check` runs the tests and dry runs CI runs.

## Machine settings

Paths and host details are in `config/local.env`, which git ignores; `config/local.env.example` lists every key.
The file is read by `workflow/flowlib/localconfig.py`, which the Taskfile and shell scripts call (`python3 workflow/flowlib/localconfig.py KEY`).
A value may use another setting, `$ROOT` say, in any order.
A variable in the environment wins, and with no file everything runs on this machine.
The workflow uses:

- paths: `ROOT` (rounds, and the runs in `ROOT/runs/`), `SCORE_CACHE`;
- what this machine does: `TARGETS` (what `all` runs; by default `simulate finetune score`, so a CPU machine sets `simulate score` and a GPU machine `finetune`), `SIMS_FROM` (`local` or `release`), `PUBLISH` (release each checkpoint as it is trained);
- resources: `MIN_FREE_GB` and `MIN_DISK_GB` (memory and disk a tick needs free to start), `HEADROOM_CORES` and `HEADROOM_GB` (cores and memory kept free), `CORES` and `MEM_MB` (set them outright instead), `SIM_SLOTS` (batches at once, 2 by default) and `DEVICE` (`auto`: CUDA, then MPS, then the CPU);
- sharing the rounds: `SIM_SEEDS` (the seeds this machine takes: seeds, `first-last` ranges or rounds as `r<N>`; empty: all), and on a helping machine `REMOTE` and `REMOTE_ROOT` (the rounds host and its rounds folder; empty: `ROOT`);
- `HOST_NAME`.

Nothing in the workflow needs a notes file. As an optional local extra, setting both `NOTIFY` and `MAILBOX` appends a one-line note to the file `MAILBOX` when a round is ready, a checkpoint is saved or published, the checked checkpoint misses zero-shot, scores land or a run fails. Both are empty by default, a note that cannot be written is only a warning, and no rule's output or success depends on it. The optional detached-job helpers in `tools/` use the same switch.

Task reads `config/local.env` itself, not `$LOCAL_CONFIG`.

Every target runs with one profile, `workflow/profiles/default`, and this machine's cores, memory and GPU less its headroom (`workflow/flowlib/machine.py`, through `workflow/flowlib/compute.py`, which the Taskfile and the tick call; `python3 workflow/flowlib/machine.py device` prints the device). `workflow/profiles/ci` is for CI.

### Where jobs run (`COMPUTE`)

`COMPUTE=single` (the default) runs every job on this machine, as above.
`COMPUTE=ssh` makes this machine the coordinator: it holds the one Snakemake run and places each job on a worker (`workflow/profiles/ssh`, the cluster-generic executor with `workflow/scripts/remote/`).
The coordinator needs GNU rsync 3.1 or later first on `PATH` (the copies use `--ignore-missing-args`). macOS's `/usr/bin/rsync` (openrsync) does not take it, so install rsync from Homebrew there. Submit refuses a job that would be copied with an rsync that cannot, naming the binary.

- `WORKERS` lists the workers as `name=host:ROOT:CHECKOUT`: the host as ssh names it (or `local`, which may leave out ROOT and CHECKOUT for this machine's), its rounds folder and its checkout of this repository.
- `WORKER_GROUPS` places rule groups on a worker, for example `WORKER_GROUPS="train=workstation score=server"`. The groups are `simulate`, `train`, `evaluate`, `publish` and `score` (`flowlib.compute.RULE_GROUPS`); a group left out, or `any`, runs on the first worker with room. Local rules (listings, records, publishing) always run on the coordinator.
- The pool is the sum of the workers' budgets, each read from the worker's own local config at launch (`machine.py budget`); a worker that does not answer is left out, and its jobs fail at submit until it does. A GPU rule placed by its group also takes that worker's own `gpu_<worker>`, so one worker's jobs never hold another's GPU. The submit script holds a job until it fits beside the jobs already on its worker.
- A job on another root or checkout: the worker must answer over ssh at submit, or the job fails at once (no other worker takes it); its checkout must be at the coordinator's commit; the job's inputs are pushed by rsync, its worker's Snakemake (`WORKER_SNAKEMAKE`) runs it with the coordinator's run settings (paths moved to the worker's ROOT) and the worker's own machine settings (device, headroom, caches); then its log, and on success its outputs and `params.copy_back`, are pulled back before the job counts as done. A worker silent for 10 minutes fails the job, and its retry resumes from the saved state. The job's session has a terminal, so the worker ends the job when the session drops; a retry that still finds the job running there (its pid under the worker's ROOT, `.remote-jobs/`) fails rather than start a second copy, and a cancel stops it by that pid.
- The trainer's own state stays on its worker, so `train` stays on one worker.

`single`, `ssh` and `slurm` ship. Rules declare only their needs (`threads`, `mem_mb`, `gpu`), placement lives only in the local config, and any other `COMPUTE=<name>` runs with `workflow/profiles/<name>` and nothing else from `flowlib.compute`, so a cluster or cloud profile needs no code:

- SLURM with a shared filesystem: `COMPUTE=slurm` runs `workflow/profiles/slurm` (`snakemake-executor-plugin-slurm`, in the `paper` group). Every job is a Slurm job; its memory and runtime come from the profile's `default-resources` and `set-resources`, and a rule's `gpu: 1` is its GPU request (`--gpus`). The account and partition (and a GPU partition for `train_segment`, if the cluster has one) go in an untracked profile layered on it (`SNAKEMAKE_PROFILE`), as `default-resources` `slurm_account=` and `slurm_partition=` and `set-resources`; without them the plugin passes `sinfo`'s default partition and a guessed account explicitly, which override `SBATCH_*` in the environment. Before a run: `uv sync --all-groups` once, then `UV_NO_SYNC=1` (and `UV_OFFLINE=1` for offline nodes) and `HF_HOME` on the shared filesystem in the submit environment, and `DEVICE` and `forecast.device` set outright, as the login node detects no GPU. No files move: every job sees ROOT. Its dry run is tested (`tests/workflow/test_workflow_slurm.py`); a run on a cluster is not;
- a cloud batch service: add an executor plugin (`snakemake-executor-plugin-googlebatch` or `-kubernetes`) and a storage plugin (`snakemake-storage-plugin-gcs` or `-s3`), and in the profile `executor`, `default-storage-provider`, `default-storage-prefix` (the bucket), `shared-fs-usage: none` and the executor's project, region and machine settings. Files then move through the bucket, not rsync, and ROOT must be a path the storage prefix maps (see the plugin's docs). This is not tested here.

A machine that needs more can name an untracked profile of its own with `SNAKEMAKE_PROFILE` (see `config/profile.example`), never one in the repository.
All of them rerun a step only when an input is newer (`rerun-triggers: mtime`), so editing a script does not reschedule finished work.
Settings that are the same on every machine are in `workflow/config.yaml`; override them with `--config` or `--configfile`.

Two machines share only GitHub releases: simulation batches (`sims-r<N>`, one asset per batch) and checkpoints (`chronos-slots-<hash>`, `chronos-ft-<hash>`).
Each machine reruns its task as releases land, in its coordinator run (`task nowcast-workflow:run`).

## 1. Simulate (`rules/simulate.smk`)

Round r is `batches` batches of `batch` outbreaks, seeds from 30000 + 100 r, each with exactly one observation draw.
Each outbreak draws its settings independently from the grid (`simulate.grid_file`, `config/finetune/grid.yaml`), and a round is used whole for training or as the held-out round.
The simulator is `nowcastfoundation.simulator` (`simulate_batch`, run as `python -m nowcastfoundation.simulator GRID N SEED OUT`).
It draws each outbreak's design and values in Python and runs each outbreak as one numpyro model from the shared PyRenew components, its latent infections to its reported counts (`nowcastrenewal.simulation.outbreak`).

- `simulate_batch`: one batch of the simulator. Batches run by count (`sim_slots`), started by the scheduler as slots free up, with no memory claim of their own. A batch that fails is retried once. The simulator must be the committed one, and it runs with JAX on one thread.
- `batch_record`: the batch's manifest row (sha256 of each file, memory, time, grid, commit, host), keeping a row the round's manifest already holds.
- `round_manifest`: the round's `manifest.csv`, rows in seed order.
- `sim_qc`: the sim QC (`sim-qc`, `nowcastfoundation.finetune.qc`), a record of each simulation's summaries in the round's `qc/`. It sets no thresholds and nothing is excluded or deleted.
- `round_ready`: the seed list and `READY`, last.
- `simulator_benchmark`: opt-in, two passes of one batch, cold then warm, as a `simulate_batch` job runs it on a given JAX device, in the rounds' `bench/`.
- `simulate`: the rounds in `rounds`. At most `sim_slots` batches run at once (2 by default).
- `release_batch`, `release`: opt-in, each batch to its round's release.
- `push_batch`, `push`: on a helping host, its seeds (`SIM_SEEDS`), each pushed to the rounds host's round (`REMOTE`, `REMOTE_ROOT`): its files, then its manifest row (`flowlib.hosts`).
- `await_batch`: on the rounds host, a seed not in its `SIM_SEEDS`, done once the host that simulates it has pushed its manifest row; one waits at a time. Its row is kept as pushed.

Outputs are in `<ROOT>/rounds/r<N>/`.

## 2. Fine-tune (`rules/finetune.smk`)

Runs are entries of `runs` in `workflow/config.yaml`: a name and the `finetune` settings it changes (`workflow/flowlib/runs.py`).
Every run takes the same rules through the `{run}` wildcard and keeps its files in `<ROOT>/runs/<run>/`; the rounds it trains on are shared.
`finetune` trains the runs in `train_runs` (every run when unset).
A run outside `train_runs` that another target asks for starts only after their final checkpoints.

One continuous run of `finetune-stream` (`nowcastfoundation.finetune.stream`) from base Chronos-2, cut into one segment per checkpoint prefix (`finetune.prefixes`).
Segment N resumes from the state kept with the prefix before and stops once `n<N>` is saved (`--until N`).
The state carries the learning-rate schedule, the step, the batch position and torch's random state, so the segments in turn are the run that never stopped.

- `seed_list`: the training seeds in order, the rounds' in seed order or a fixed `seed_file`. A run takes the grid's rounds (`grid_rounds`): it validates on round 33, trains on 34 to 37 and tests on 38. Rounds 9 to 32 belong to earlier versions of the grid (`retired_rounds`) and are never simulated again, as a round sets its seeds and its release `sims-r<N>`. When the grid changes, its rounds join `retired_rounds` and it takes new ones. The `release` target and each upload refuse a retired round.
- `fetch_batch`: a batch into `<ROOT>/runs/<run>/sims/`, hard linked from this machine's rounds, copied from the rounds host's (`remote`, `REMOTE` and `REMOTE_ROOT`) or downloaded from its release (`release`; `sims_from`, or `SIMS_FROM` in `config/local.env`).
- The batches a machine without the rounds can have are listed afresh each time a run plans (`listed_batches` in rules/common.smk, recorded in `<ROOT>/runs/<run>/available-batches.txt`): those the rounds host's manifests list (mirrored with its QC summaries in `<ROOT>/runs/<run>/rounds/`), or those released. A segment is planned only once its batches are listed, counting the training seeds up to the first batch not yet listed, and nothing trains before the whole validation round is listed.
- `train_segment`: one segment on the GPU. Its inputs are the batches that hold its first N outbreaks, one spare, and the validation round's batches. A finished checkpoint is final and is never retrained.
- `evaluate_steps`, `evaluate_base`, `evaluate_checkpoint`: with `finetune.eval: offload` the trainer only trains, saving an adapter snapshot at step 0 and each evaluation step. `evaluate_base` scores step 0 alone (base Chronos-2, the start loss), `evaluate_steps` a segment's snapshots (`train/evals/step-<k>.json`) and `evaluate_checkpoint` the checkpoint (`n<N>/held-out.json`, beside its settings), which needs only step 0. They run on the forecasting device (`forecast` in the config, the CPU by default) beside the trainer, newest first. They use the trainer's own loss functions (`workflow/scripts/evaluate_offloaded.py`) in the `evaluate` group, so `WORKER_GROUPS="evaluate=server"` runs them off the trainer's machine. The dashboard, `checkpoint_check` and the release read the results with the trainer's (`flowlib.evaluations`).
- `seven_day_check`: each checkpoint's 7-day check on its run's held-out blocks (`n<N>/seven-day.json`, `nowcastfoundation.finetune.evaluate.seven_day_check`): the 7-day row's median against the 7-day sums of the same block's daily medians (median ratio, and the share of those sums in the row's 90% interval), and the roughness of the log 7-day median against the truth's. It runs on the forecasting device beside the trainer, for every run however it evaluates; `finetune` asks for it and the dashboard's checkpoint tables show it.
- `validation_curve`: a run's validation loss by level at the start and at each checkpoint, against simulations trained (`<run>/validation-curve.csv`, `flowlib.evaluations.validation_curve`). The paper's `evidence` rule reads the final run's.
- `block_combinations`: the share of each combination of block layout and options over a run's training, from its final checkpoint's tally (`<run>/train/block-combinations.csv`, the SI table).
- `real_training_blocks`: a run's real-data training blocks when `finetune.real_training` is set (`<run>/real-training.pkl`): the German triangle's reports before the first scored German date, cut as the German tasks lay it out (`nowcastgermany.finetune_blocks`).
- `checkpoint_check`: a report of whether checkpoint `check_at` (800) beats zero-shot on the held-out simulations. No BDBV data is scored in development: BDBV is held out for the paper. It never stops training.
- `finetune`: every prefix.

Outputs are in `<ROOT>/runs/<run>/train/n<N>/`.

## 3. Publish (`rules/publish.smk`)

- `verify_checkpoint`: the checkpoint's hash recomputes from its commit (`finetune-verify`).
- `publish_checkpoint`: opt-in, the release `chronos-slots-<hash>` or `chronos-ft-<hash>` (`workflow/scripts/publish_checkpoint.py`).
- `publish_germany_unit`: opt-in, attaches a released checkpoint's German unit (`<SCORE_CACHE>/germany-days/slots-<hash>`) to its release as `germany-forecasts.tar.gz` (`nowcastgermany unit-publish`), which the Pages build loads for the checkpoint `config/finetune.toml` pins.
- `publish`: every checkpoint.

## 4. Score (`rules/score.smk`)

- `release_list`: the repository's releases, the only rule that reads the network; `task nowcast-foundation:score` and each tick rerun it in a run of its own before scoring, as Snakemake 9 ends a run after a forced checkpoint without the jobs it unlocks. The checkpoints scored are the slots ones first, then the others, oldest first, from `published_from`, or `score.labels`.
- `score_germany`, `score_release`: the units of `nowcastgermany.checkpoint` for each checkpoint and the references, newest checkpoint first. Each caches one file per date and resumes from it.
  Units run on the forecasting device with its threads (`forecast` in the config; `FORECAST_DEVICE`/`FORECAST_THREADS` override it per machine), beside the trainer: `train_segment` takes the `gpu` resource, with the highest priority, so the next segment starts the moment the previous ends. With `forecast.device: cuda`, `forecast.gpu: exclusive` makes forecasting jobs take it too (they wait for the GPU) and `share` runs them beside the trainer on it.
  BDBV is held out for the paper, so no BDBV unit runs.
- `score_run_germany`: a run's own saved checkpoint `n<N>`, loaded from `<ROOT>/runs/<run>/train/n<N>` with no release, cached in `<ROOT>/runs/<run>/scores/`.
- `run_scores`: every trained run's saved checkpoints and report; `score` includes it.
- `run_table`: a run's saved checkpoints as table rows, German relative scores and held-out losses under the run's name (`<run>/reports/checkpoint-scores.csv`).
- `comparison`: every run's `run_table` in one table (`<ROOT>/runs/comparison.csv`), for the paper's fine-tuning comparisons.
- `daily_table`: a run's saved checkpoints on both German views, daily counts and 7-day sums, relative to the German config's references (`<run>/reports/compare-daily.csv`).
- `compare_daily`: every run's `daily_table` in one table (`<ROOT>/runs/compare-daily.csv`), the paper's `compare_daily`.
- `coupling_germany`: a checkpoint's German forecasts' coupling scores and summary (`nowcasteval.coupling`), from its German unit, in `<SCORE_CACHE>/coupling/`: each coupling against independent draws, the foundation model's online step (`online`) among them.
- `score_table`: the rows of every checkpoint in `<SCORE_CACHE>/tables/`, keeping other rules' rows of the committed tables (BDBV rows already there stay).
- `land`: opt-in, commit the tables to `evidence/` and push, only from the evidence branch (`score.branch`), never main.

## 5. Report (`rules/report.smk`)

Reports read the other steps' outputs back for people, each with one script.

- `round_summaries`: one row per outbreak of a round (`r<N>/summaries/outbreaks.csv`, `workflow/scripts/grid_summaries.py`): design, drawn values, size, dynamics, R, delays and the phase at the training cut-offs.
- `grid_figures`: the rounds in `rounds` side by side, each summary's ECDF and the shares of patterns and phases.
- `sim_report`: a round's report, `r<N>/report/sim-report.html`; `sim_report_run` one report of every round in `rounds`; `sim_reports` all of them. `sim_diseases` gives the grid's training rounds' ranges against the published values by disease (`report/sim-diseases.csv`), for the SI's disease table.
- `german_report`: the run's checkpoints against zero-shot and randomwalknowcast on the German development half (`workflow/scripts/german_report.py`), in `<ROOT>/runs/<run>/reports/german/`.
- `hub_rounds`: our last `hub_rounds.rounds` hub rounds against every hub team on the hub's latest data, `<ROOT>/hub/hub-rounds.html` (`hub_rounds_report`, `workflow/scripts/hub_rounds_report.py`), copied to `REVIEW_DIR` when set. `hub_data` tidies the hub's data, `hub_submissions_table` reads which chain model went out from the ledger branch, and `hub_comparison` scores the hub's files as submitted beside `hub_rounds.models` rerun on current code for each round (`compare-hub --models`). They rerun when `<ROOT>/hub/head.txt` changes, which `workflow/scripts/hub_watch.sh` writes when the hub's data commit moves before it builds the target. Not part of `all`.
- `ft_check`: the BDBV foundation-model check, `<ROOT>/hub/ft-check/bdbv-ft-check.html` (`bdbvhub ft-check`, `bdbvhub.ftcheck`), copied to `REVIEW_DIR` when set. `ft_check_forecast` forecasts every hub release with one model into `<ROOT>/hub/ft-check/forecasts/<model>`: `ft_check.models` once per change of the code they depend on (`flowlib.scoring.ft_check_key`), and `ft_check.finetuned` as `ft-<checkpoint>`, so a new checkpoint (`ft_check.checkpoint`, else `config/finetune.toml`'s) reruns only it. Chronos-2 models build the active-zone and by-province blocks. The page gives each model's pairwise relative skill on the CRPS and log CRPS, scaled to `ft_check.reference`, overall and by weeks back, with the forecasts and their 50% and 90% intervals. Not part of `all`.
- `rt_check`: the Rt sense check of a stored renewal fit before its `wk R_t` is submitted, `<rt_check.out_dir>/<model>-<origin>.html` (`bdbvhubrenewal rt-check`, `bdbvhub.reporting.rtcheck`): Rt by country, province and fitted zone over their cases, with BVDOutbreakSize's Rt. It reads the fit the store (`rt_check.store`) holds for the current code and is copied to `REVIEW_DIR` as `rt-check.html` when set. Not part of `all`.
- `this_round`: this week's round, `<ROOT>/hub/this-round.html` (`this_round_report`, `workflow/scripts/this_round_report.py`), copied to `REVIEW_DIR` when set: every candidate's forecasts and nowcasts against the data as of the round, the renewal candidates' wk R_t, the gate's pick and why, and the renewal fits' convergence; no scores. `this_round_inputs` downloads the latest Submit run on main's artefacts, fetches the fit store and archives every candidate's hub table for the round (`workflow/scripts/this_round_inputs.sh`). It reruns when `<ROOT>/hub/round/run.txt` or `round/store.txt` changes, which `workflow/scripts/hub_watch.sh` writes. Not part of `all`.
- `benchmarks`: opt-in, the seconds and peak memory of every forecast and score job and the seconds by model and date, as one CSV in `<SCORE_CACHE>/` (`flowlib.benchmarks`).

## 6. Dashboard (`workflow/flowlib/dashboard/`)

`task nowcast-foundation:dashboard`, or `uv run --group paper python workflow/scripts/dashboard.py`, serves a page for a running fine-tune on the machine settings' `DASHBOARD_HOST` and `DASHBOARD_PORT` (default 127.0.0.1, port 8765); it is not a Snakemake job, so it never holds the workflow's lock.
It is one process: the page is rebuilt in the background and a client that leaves is dropped, so it needs no proxy or restart loop.
`task nowcast-foundation:dashboard-static` builds the rule `dashboard`, a static render of the same page in `<ROOT>/reports/dashboard.html`.

The page has a sticky bar (section links, the live run's step, one run and reference selector every chart follows) and sections: Status (tiles), Loss (client-side plots), Attention bias, Performance on real data (scores by study, score, target and source), Nowcasts (every dataset), Recent activity (with a mailbox) and Tables. Every chart has an info marker (`page/explain.json`).

Settings:

- `workflow/config.yaml`'s `dashboard`: references (matched on rows by model and checkpoint, with a role: `baseline`, `start` or `other`), score studies and their labels, level order and defaults, the datasets (Germany: quantiles and WIS; BDBV is held out and never shown), the notes filter.
- `workflow/config.yaml`'s `runs`: the runs shown, each from `<ROOT>/runs/<name>`, the first live. `--run NAME` shows only the named runs, the first live.
- `config/local.env` (or the file `LOCAL_CONFIG` names, outside the checkout): `ROOT`, the scoring cache (`SCORE_CACHE`) the score table and each dataset's forecasts are read from, the mailbox (`MAILBOX`), and where it listens (`DASHBOARD_HOST`, `DASHBOARD_PORT`).
- `--ref`: more references, or ones replacing the configured.

```sh
uv sync --group paper                                          # once
LOCAL_CONFIG=/path/to/machine.env task nowcast-foundation:dashboard -- --run runa
# open the address it prints
```

## 7. Paper (`rules/paper.smk`, `paper/workflow/Snakefile`)

A module of this workflow (`use rule * from paper_workflow`), run from the repository root; its paths are from the root and its outputs are in `paper/results/<mode>/`.
`development` runs the development seeds and the German first half only, `paper` the held-out seeds and both German halves, and `render` renders `paper/main.qmd` and `paper/si.qmd` from the run `publish_paper` last copied to `paper/build/`.

- Data: `germany_fetch`, `germany_triangle`, `germany_dev_stats`, `ebola_hub`, `ebola_data` (the hub's releases, downloaded on the driver for `ebola_backtest`), `data`, and `synthetic_forecasts_development` and `synthetic_forecasts_paper` (the synthetic runner's forecasts of each run's seeds).
- Benchmarks: `germany_development`, `germany_heldout` (paper runs only), `ebola_backtest` and `ebola_backtest_draws`.
- Fit store history: `fit_store_history` fetches the release asset of our stored forecasts made before each round closed and checks its checksum; `publish_fit_store_history` (opt-in) builds and publishes it from the fit store branch.
- Forecasts and scores: `forecasts`, `table`, `truth`, `scores`, `joint_scores` and `real_finetune`.
- Evidence read from git (`evidence_checkpoint_scores`), gathered by `evidence`.
- Outputs: `figures_synthetic`, `figures_germany`, `figures_ebola`, `score_tables_synthetic`, `score_tables_germany`, `score_tables_ebola`, `sim_examples`, `sim_realism` (the grid's training rounds against the German development half), `pairing_germany`, `pairing_synthetic` and `si_copula` (the walks' scores with and without pairing, and their German rolling $\rho_p$, on development data), `grid_tables`, `summary` and `publish_paper`.

## Scheduled runs

The workflow updates itself as results come in. Each machine runs its own targets in one persistent coordinator run, one tick after another; Snakemake reruns only what changed. With `COMPUTE=ssh` one coordinator runs every machine's jobs.

| Where | `SCHEDULE_TARGETS` | Does, each tick |
|---|---|---|
| the machine that scores (CPU) | `score land` | lists new checkpoint releases, scores them, rebuilds the table, and commits and pushes it to the evidence branch only if it changed |
| the machine that trains (GPU) | `finetune publish` | trains the next segment once its batches are there, and releases each new checkpoint |
| a simulating machine | `simulate release` | simulates the configured rounds and releases each batch |

A tick (`task nowcast-workflow:tick`, `workflow/scripts/tick.py`) runs the pool targets (`simulate`, `push`, `finetune`, `score`, `dashboard`) as one Snakemake run at nice 19 with the default profile and the machine's budget, so a free core or the GPU takes whichever queued job fits.
When jobs compete, the trainer goes first (priority 20), then simulation (15), then checkpoint scoring (10 and below).
Before it, the lists the pool plans from (the checkpoint releases, and on a machine without the rounds the batches it can have) are refreshed in a run of their own.
A failed refresh leaves the last lists, and the pool still runs.
The outward targets (`land`, `publish`, `release`) follow in their configured order.
A failed run skips `land` and `release`, so `land` never follows a failed `score`, but `publish` still releases the checkpoints saved.
A tick's run trains only the next segment (`next_segment_only`), so it ends after one segment and the next run plans afresh.
Work that appears during a run, such as a new checkpoint or a landed round, waits for the next tick's run, so it waits at most as long as the longest running job, one segment.
Snakemake locks the worktree it runs in, so each machine has one driver worktree.
A tick is skipped when another tick is still running (a lock in `output/`) or when less than `MIN_FREE_GB` of memory or `MIN_DISK_GB` of disk is free. Its log is `output/workflow-tick.log`.

`task nowcast-workflow:run` (`workflow/scripts/coordinate.py`) is the coordinator run: it runs a tick, and as soon as one ends starts the next, after `COORDINATE_PAUSE` seconds (60), or `COORDINATE_BACKOFF` (600) after a failed tick. No timer runs it: start it once, under nohup or a service manager, and stop it with `touch output/coordinator.stop` (it ends after the running tick) or by its pid.

On CI, the site and the paper (`.github/workflows/pages.yml`) rebuild every 6 hours (`43 */6 * * *`, after the fit store's run) and whenever `evidence/` changes on main, so new checkpoint scores reach the docs and the paper without a manual step.

Nothing here needs an agent or a notes file; with `NOTIFY` set, rules still add their optional notes.

## Monitor

`task nowcast-workflow:monitor` serves [Panoptes](https://github.com/panoptes-organization/panoptes), Snakemake's web monitor: every run's jobs, their state, what is running and what has failed, in one page.
Every Snakemake run the tasks, ticks and paper scripts start reports to it through the logger plugin `snakemake-logger-plugin-panoptes` (`workflow/flowlib/monitor.py`), on by default.
Runs report to `MONITOR_URL`; another machine's runs report to the serving machine's address, and an empty `MONITOR_URL` switches reporting off.
The plugin drops events it cannot send, so a run goes on when no server is listening.
The page is served on `MONITOR_HOST` and `MONITOR_PORT`, with its database in `<ROOT>/monitor/`, all from `config/local.env`.
Panoptes has no DAG view, so the same server adds one at `/dag/` (`workflow/flowlib/monitor_server.py`, `workflow/flowlib/dag.py`).
For each `MONITOR_DAG` target (`NAME=TARGET`) it draws the rule graph and the job DAG, each job pending, running, done or failed.
A job Snakemake would not run is done; Panoptes gives the rest their state, so the colours follow the run on each visit.
The graphs are built with `MONITOR_DAG_OPTIONS` every `MONITOR_DAG_EVERY` seconds, without a lock; `task nowcast-workflow:dag -- NAME=TARGET` builds one now.
In the workflow (`rules/monitor.smk`), `monitor_dag` builds one target's graphs into `<ROOT>/monitor/dag/<name>.json` and `monitor` every `MONITOR_DAG` target's; both are opt-in.
Each tick also writes its machine's status to `<ROOT>/status/<host>.json` every `STATUS_EVERY` seconds (`workflow/flowlib/runstatus.py`).
The fine-tuning dashboard shows the science (losses, scores, nowcasts, views), not jobs.

## CI

| Target | On CI | Why |
|---|---|---|
| tests and dry runs of every target (`task nowcast-workflow:check`) | yes | fast, no network |
| the paper and its figures and tables | yes, as now (Pages) | from stored forecasts and pinned evidence |
| `score_table` from stored caches | yes, once the paper module lands | cheap |
| checkpoint forecasts, slots check | no | hours, needs the checkpoints |
| `simulate`, `finetune` | never | heavy R, GPU |
| `release`, `publish`, `land` | never | outward-facing |

`all` is the default target.
