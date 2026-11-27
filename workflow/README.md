# The workflow

One Snakemake workflow simulates outbreaks, fine-tunes Chronos-2 on them, publishes the checkpoints, scores them and builds the paper.
The paper's rules (`paper/workflow/Snakefile`) are a module of it, with their settings in `paper/workflow/config.yaml`.

In this project it makes the fine-tuned Chronos-2 checkpoints (`nowcastfoundation`) and the paper's results.
Simulation and fine-tuning run on our own CPU and GPU hosts, not on CI, and publish each round and checkpoint as a GitHub release.
The Site workflow runs the paper's targets through [`scripts/paper_run.py`](scripts/paper_run.py), and the Tests workflow runs its tests and dry runs.

Where to look:

- Entry point: `Snakefile`, run through `task workflow:<target>` (`Taskfile.yml`) with a profile from `profiles/`.
- Shared rules any study uses: `rules/` (simulate, finetune, evaluate, publish, score, monitor, report).
- Their Python: `flowlib/` (settings, scheduling, scoring units, the dashboard) and `scripts/`.
- Study modules: the paper's rules in [`paper/workflow/`](../paper/workflow/), with their own config.
- Tests: `tests/`.

This page walks through it in order, with the command for each piece, from a whole target down to one file.

## Running it

High level, one Taskfile task per target:

| Task | Target | Does | Writes outside this machine |
|---|---|---|---|
| `task workflow:all` | `all` | simulate, fine-tune and score | no |
| `task workflow:simulate ROUNDS="4 5"` | `simulate` | simulation rounds, with QC and manifests | no |
| `task workflow:release ROUNDS=4` | `release` | each batch to its `sims-r<N>` release | GitHub releases |
| `task workflow:push ROUNDS="19 18"` | `push` | a helping host's batches, pushed to the rounds host | the rounds host |
| `task workflow:finetune` | `finetune` | every checkpoint prefix of the run | no |
| `task workflow:publish` | `publish` | verify and release each checkpoint | GitHub releases |
| `task workflow:score` | `score` | score each released checkpoint | no |
| `task workflow:land` | `land` | commit the score tables to `evidence/` and push | git push |
| `task paper` | `development` or `paper`, then `render` | the paper (section 7) | no |

Pass Snakemake options after `--`: `task workflow:score -- -n` is a dry run, and a file path builds just that output.
Low level, any output file, for example:

```sh
uv run --group paper snakemake <ROOT>/rounds/r34/sims/counts-33400.csv.gz  # one batch
uv run --group paper snakemake <ROOT>/runs/runa/train/n800/settings.json  # one checkpoint
uv run --group paper snakemake <SCORE_CACHE>/germany-days/slots-<hash>.done    # one checkpoint's German scores
```

The settings in `workflow/config.yaml`, with any `--configfile` or `--config` on top, are checked against `workflow/schemas/config.schema.yaml` when the workflow loads, so a misspelt or unknown setting is an error before any job runs.
A run in `runs` takes any `finetune` setting and no other.

Helpers: `task workflow:rulegraph` draws the rule graph, `task workflow:progress` lists what exists and what would run, `task workflow:report` writes Snakemake's HTML report, and `task workflow:check` runs the tests and dry runs CI runs.

## Machine settings

Paths and host details are in `config/local.env`, which git ignores; `config/local.env.example` lists every key.
The file is read by `workflow/flowlib/localconfig.py`, which the Taskfile and shell scripts call (`python3 workflow/flowlib/localconfig.py KEY`).
A value may use another setting, `$ROOT` say, in any order.
A git worktree without its own `config/local.env` reads the main checkout's, so a host keeps one local config for all its worktrees.
A variable in the environment wins, and with no file everything runs on this machine.
The workflow uses:

- paths: `ROOT` (rounds, and the runs in `ROOT/runs/`), `SCORE_CACHE`;
- what this machine does: `TARGETS` (what `all` runs; by default `simulate finetune score`, so a CPU machine sets `simulate score` and a GPU machine `finetune`), `SIMS_FROM` (`local` or `release`), `PUBLISH` (release each checkpoint as it is trained);
- resources: `MIN_FREE_GB` and `MIN_DISK_GB` (memory and disk a tick needs free to start), `HEADROOM_CORES` and `HEADROOM_GB` (cores and memory kept free), `CORES` and `MEM_MB` (set them outright instead), `SIM_SLOTS` (batches at once, 2 by default), `TRAINER_CORES` (cores kept for the trainer) and `DEVICE` (`auto`: CUDA, then MPS, then the CPU);
- sharing the rounds: `SIM_SEEDS` (the seeds this machine takes: seeds, `first-last` ranges or rounds as `r<N>`; empty: all), and on a helping machine `REMOTE` and `REMOTE_ROOT` (the rounds host and its rounds folder; empty: `ROOT`);
- `HOST_NAME`.

Nothing in the workflow needs a notes file. As an optional local extra, setting both `NOTIFY` and `MAILBOX` appends a one-line note to the file `MAILBOX` when a round is ready, a checkpoint is saved or published, the checked checkpoint misses zero-shot, scores land or a run fails. Both are empty by default, a note that cannot be written is only a warning, and no rule's output or success depends on it. The optional detached-job helpers in `tools/` use the same switch.

`LOCAL_CONFIG` names a local config outside the checkout in place of `config/local.env`, for the tasks, the tick and Snakemake alike.

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

- SLURM with a shared filesystem: `COMPUTE=slurm` runs every job as a Slurm job through `workflow/profiles/slurm` (`snakemake-executor-plugin-slurm`, in the `paper` group), with the site's account and partitions file from `config/local.env` (`SLURM_ACCOUNT`, `SLURM_PARTITIONS`). Each GPU job asks Slurm for its own GPU, and every run `finetune` trains trains at once, as far as the cluster has GPUs free. Steps: [Run on a Slurm cluster](#run-on-a-slurm-cluster). Its dry run is tested (`workflow/tests/test_workflow_slurm.py`); a run on a cluster is not;
- a cloud batch service: add an executor plugin (`snakemake-executor-plugin-googlebatch` or `-kubernetes`) and a storage plugin (`snakemake-storage-plugin-gcs` or `-s3`), and in the profile `executor`, `default-storage-provider`, `default-storage-prefix` (the bucket), `shared-fs-usage: none` and the executor's project, region and machine settings. Files then move through the bucket, not rsync, and ROOT must be a path the storage prefix maps (see the plugin's docs). This is not tested here.

A machine that needs more names a profile of its own, kept outside the repository, with `SNAKEMAKE_PROFILE` (in `config/local.env`): it replaces the profile and budget `COMPUTE` would give, so start from a copy of `workflow/profiles/default` or `workflow/profiles/slurm`. The default and slurm profiles with the machine settings (`CORES`, `MEM_MB`, `HEADROOM_*`, `SIM_SLOTS`, `DEVICE`) cover what most machines need.
All of them rerun a step only when an input is newer (`rerun-triggers: mtime`), so editing a script does not reschedule finished work.
Settings that are the same on every machine are in `workflow/config.yaml`; override them with `--config` or `--configfile`.

Two machines share only GitHub releases: simulation batches (`sims-r<N>`, one asset per batch) and checkpoints (`chronos-slots-<hash>`, `chronos-ft-<hash>`).
Each machine reruns its task as releases land, in its coordinator run (`task workflow:run`).

## 1. Simulate (`rules/simulate.smk`)

Round r is `batches` batches of `batch` outbreaks, seeds from 30000 + 100 r, each with exactly one observation draw.
Each outbreak draws its settings independently from the grid (`simulate.grid_file`, `packages/nowcastfoundation/config/finetune/grid.yaml`), and a round is used whole for training or as the held-out round.
The simulator is `nowcastfoundation.simulator` (`simulate_batch`, run as `python -m nowcastfoundation.simulator GRID N SEED OUT`).
It draws each outbreak's design and values in Python and runs each outbreak as one numpyro model from the shared PyRenew components, its latent infections to its reported counts (`nowcastrenewal.simulation.outbreak`).

- `simulate_batch`: one batch of the simulator. Batches run by count (`sim_slots`), started by the scheduler as slots free up, each claiming its peak memory (5.5 GB). A batch that fails is retried once. The simulator must be the committed one, and it runs with JAX on one thread.
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
No run waits on another: each run's trainer takes one `gpu` slot, so with a GPU per run every run trains at once, and with fewer they take the GPUs in turn, the runs of `train_runs` first in order, then the others in the order of `runs`.

One continuous run of `finetune-stream` (`nowcastfoundation.finetune.stream`) from base Chronos-2, cut into one segment per checkpoint prefix (`finetune.prefixes`).
Segment N resumes from the state kept with the prefix before and stops once `n<N>` is saved (`--until N`).
The state carries the learning-rate schedule, the step, the batch position and torch's random state, so the segments in turn are the run that never stopped.

- `seed_list`: the training seeds in order, the rounds' in seed order or a fixed `seed_file`. A run takes the grid's rounds (`grid_rounds`): it validates on round 33, trains on 34 to 37 and tests on 38. Rounds 9 to 32 belong to earlier versions of the grid (`retired_rounds`) and are never simulated again, as a round sets its seeds and its release `sims-r<N>`. When the grid changes, its rounds join `retired_rounds` and it takes new ones. The `release` target and each upload refuse a retired round.
- `fetch_batch`: a batch into `<ROOT>/runs/<run>/sims/`, hard linked from this machine's rounds, copied from the rounds host's (`remote`, `REMOTE` and `REMOTE_ROOT`) or downloaded from its release (`release`; `sims_from`, or `SIMS_FROM` in `config/local.env`).
- The batches a machine without the rounds can have are listed afresh each time a run plans (`listed_batches` in rules/common.smk, recorded in `<ROOT>/runs/<run>/available-batches.txt`): those the rounds host's manifests list (mirrored with its QC summaries in `<ROOT>/runs/<run>/rounds/`), or those released. A segment is planned only once its batches are listed, counting the training seeds up to the first batch not yet listed, and nothing trains before the whole validation round is listed.
- `train_segment`: one segment on the GPU. Its inputs are the batches that hold its first N outbreaks, one spare, and the validation round's batches. A finished checkpoint is final and is never retrained.
- `evaluate_base`, `evaluate_checkpoint`: with `finetune.eval: offload` the trainer only trains, saving an adapter snapshot at step 0. `evaluate_base` scores step 0 alone (base Chronos-2, the start loss, `train/evals/step-0000000.json`) and `evaluate_checkpoint` each checkpoint (`n<N>/held-out.json`, beside its settings), which needs only step 0. Step 0 is kept in the score cache by model, held-out set, evaluator and precision (`held-out-base/`), so it is scored once. One forward pass gives every loss and the coverage (`finetune.evaluate.held_out_pass`). With the 7-day check and the validation forecasts, they run on the checkpoint scoring device (`compute.scoring_forecast`): the forecasting device (`forecast` in the config, the CPU by default) with the cores the trainer leaves less one unit's (`machine.scoring_threads`) while a run trains, so they never pause it, and a whole GPU once no run trains, in blocks packed to the GPU's memory. On the CPU, step 0's and a checkpoint's one pass split their batches over worker processes, one for each `forecast.threads` of the rule's threads (`Scoring.evaluation_workers`, passed as `EVALUATE_WORKERS`), each with its share of the job's threads and, on Linux, its cores (`nowcastutils.workers`). The batches are those one process runs, so the results are the same however many workers run them. A worker exits when its job's process does, and that process stops its workers on SIGTERM. Their memory floor is `flowlib.scoring.EVALUATION_WORKER_MB` a worker plus `EVALUATION_COORDINATOR_MB`. They run at the inference precision (`PRECISION`, checked against fp32 on a sample: `precision_check` in the sidecar), newest first, right after the hand-offs and the batches the next segment reads, above the references, the German and BDBV check scoring and the other batches (`compute.EVALUATION_PRIORITY`), so the dashboard's validation loss comes first among the CPU jobs. No release or check waits for them. Each reruns only when an input changes: its checkpoint, the run's held-out set (`held-out-blocks.json`) or the evaluator's code key, a digest of the repository's modules the evaluator imports (`flowlib.scoring.evaluator_code`, `nowcastutils.codekey`). The German reference and the BDBV check are keyed the same way, by the modules their jobs import, so a change to code they never run, such as the trainer, reruns neither. They use the trainer's own loss functions (`workflow/scripts/evaluate_offloaded.py`) in the `evaluate` group, so `WORKER_GROUPS="evaluate=server"` runs them off the trainer's machine. The dashboard, `checkpoint_check` and the release read the results with the trainer's (`flowlib.evaluations`).
- `seven_day_check`: each checkpoint's 7-day check on its run's held-out blocks (`n<N>/seven-day.json`, `nowcastfoundation.finetune.evaluate.seven_day_check`): the 7-day row's median against the 7-day sums of the same block's daily medians (median ratio, and the share of those sums in the row's 90% interval), and the roughness of the log 7-day median against the truth's. It runs on the forecasting device beside the trainer, for every run however it evaluates; `finetune` asks for it and the dashboard's checkpoint tables show it.
- `run_progress`: a run's progress tables from its saved checkpoints, as they stand (`<run>/reports/progress/`, `flowlib.progress`): the training loss by step, the validation loss by source at each evaluated step and by level at the start and each checkpoint, the 7-day check and, where its page is built, the BDBV check by checkpoint. The paper's `progress_<run>` rule plots them.
- `block_combinations`: the share of each combination of block layout and options over a run's training, from its final checkpoint's tally (`<run>/train/block-combinations.csv`, the SI table).
- `real_training_blocks`: a run's real-data training blocks when `finetune.real_training` is set (`<run>/real-training.pkl`): the German triangle's reports before the first scored German date, cut as the German tasks lay it out (`nowcastgermany.finetune_blocks`).
- `checkpoint_check`: a report of whether checkpoint `check_at` (800) beats zero-shot on the held-out simulations. No BDBV data is scored in development: BDBV is held out for the paper. It never stops training.
- `finetune`: every prefix, and its scoring: each checkpoint's German unit (`score_run_germany`), the reference units (`score_germany`), each trained run's German report (`german_report`) and each released checkpoint's BDBV check page (`ft_check_page`), kept when up to date.
  The trainer claims `finetune.cut_workers` plus one cores, its block-cutting workers and itself, so scoring and simulation beside it share what is left of the machine's budget.
  Those cores are kept for it: every other job takes its threads of `side_cores`, the budget less `TRAINER_CORES` (4 by default; 0 where nothing trains) for each GPU slot, so a trainer whose inputs appear mid-run never waits behind CPU jobs.
  Its memory is kept for it too: every other job also claims its memory of `side_mem_mb`, the memory budget less, for each GPU slot, the most any segment under `ROOT` took (`flowlib.memory.trainer_mb`), so the next segment starts the moment the one before is saved, never after the scoring that started while it trained.

Outputs are in `<ROOT>/runs/<run>/train/n<N>/`.

## 3. Publish (`rules/publish.smk`)

- `verify_checkpoint`: the checkpoint's hash recomputes from its commit (`finetune-verify`).
- `publish_checkpoint`: opt-in, the release `chronos-slots-<hash>` or `chronos-ft-<hash>` (`workflow/scripts/publish_checkpoint.py`). It never waits for the checkpoint's held-out losses.
- `ft_check_release`: a saved checkpoint's BDBV check page (`ft_check_page`), named by the hash in its settings (`<ck>/ft-check.done`). A fine-tuned checkpoint's BDBV check forecast waits for its release.
- `release_held_out`: an offloaded checkpoint's held-out losses (`held-out.json`), uploaded to its release once scored (`<ck>/held-out-released.txt`).
- `publish_germany_unit`: opt-in, attaches a released checkpoint's German unit (`<SCORE_CACHE>/germany-days/slots-<hash>`) to its release as `germany-forecasts.tar.gz` (`nowcastgermany unit-publish`), which the Pages build loads for the checkpoint `packages/nowcastfoundation/config/finetune.toml` pins.
- `publish_hf`: opt-in and named by its output only (`<ck>/huggingface.json`), uploads a verified checkpoint to the Hugging Face Hub repository `[huggingface] repo` in `packages/nowcastfoundation/config/finetune.toml`: its adapter, settings, `sha256sums.txt` and model card (`publish-hf`, `nowcastfoundation.hubpublish`), as the revision tagged with its spec. The token is `HF_TOKEN` from the environment. With `huggingface.dry_run` in the config, or no repository, it only writes the folder `<ck>/huggingface/`.
- `publish`: every saved checkpoint, with its held-out losses once scored.

## 4. Score (`rules/score.smk`)

- `release_list`: the repository's releases, the only rule that reads the network; `task workflow:score` and each tick rerun it in a run of its own before scoring, as Snakemake 9 ends a run after a forced checkpoint without the jobs it unlocks. The checkpoints scored are the slots ones first, then the others, oldest first, from `published_from`, or `score.labels`.
- `score_germany`, `score_release`: the units of `nowcastgermany.checkpoint` for each checkpoint and the references, newest checkpoint first. Each caches one file per date and resumes from it.
  A German unit (`score_germany`, `score_run_germany`) splits its chunks of dates over worker processes of one thread each, as many as its rule's threads (`score.workers`; `SCORE_WORKERS` overrides it per machine), so the scheduler counts its cores. Each worker forecasts whole chunks, the same as one process would, so the rows are the same however many workers run; on Linux each worker pins itself to its share of the job's cores. On CUDA it forecasts in one process, which holds the rule's GPU. A worker exits when its unit's process does, and that process stops its workers on SIGTERM. Its memory floor is `flowlib.scoring.WORKER_MB` a worker plus `COORDINATOR_MB` for that process. The held-out evaluation leaves a unit's workers free beside it.
  Units run on the forecasting device with its threads (`forecast` in the config; `FORECAST_DEVICE`/`FORECAST_THREADS` override it per machine), beside the trainer: `train_segment` takes the `gpu` resource, with the highest priority, so the next segment starts the moment the previous ends. With `forecast.device: cuda` forecasting jobs take it too, a whole GPU each, so they wait for a free one: jobs never share a GPU. A checkpoint's units take a whole GPU of their own when the machine has one that no run is left to train on, so once training ends the next run or tick scores on the GPU (`compute.scoring_forecast`); the BDBV check stays on the forecasting device.
  The `gpu` resource has a slot per CUDA GPU (`GPUS` in config/local.env, else those `nvidia-smi` lists), and each job taking it runs on a GPU of its own (`workflow/flowlib/gpus.py` sets `CUDA_VISIBLE_DEVICES`).
  Each job asks for the memory its own last benchmark shows (`max_rss`, with a margin; else a like job's, else its rule's minimum: `flowlib.memory`), and a job that runs out runs again with twice as much (the profiles' `retries`). On a shared machine (`COMPUTE` single, or each `ssh` worker) the trainer and the forecasting jobs start through `workflow/flowlib/launch.py`: it waits until the machine's available memory covers the job's and the headroom, then pins the job to as many cores as its threads, cores no other job holds, so pools that ignore the thread settings, such as JAX's, cannot spread over the machine; the trainer takes its own `TRAINER_CORES`, the last cores for each `gpu` slot. Under Slurm or another scheduler, which gives each job its cores and memory, the launcher is left out.
  BDBV is held out for the paper, so no BDBV unit runs.
- `score_run_germany`: a run's own saved checkpoint `n<N>`, loaded from `<ROOT>/runs/<run>/train/n<N>` with no release, cached in `<ROOT>/runs/<run>/scores/`.
- `run_scores`: every trained run's checkpoints and report; `score` includes it.
  On a machine that trains (`finetune` in `TARGETS`), a run's table, report or comparison asks for every checkpoint in reach, so asking for the paper trains and scores what it needs; on a machine that only scores, the checkpoints already saved.
- `run_table`: a run's saved checkpoints as table rows, German relative scores and held-out losses under the run's name (`<run>/reports/checkpoint-scores.csv`).
- `comparison`: the paper's live run's and comparison runs' `run_table` in one table (`<ROOT>/runs/comparison.csv`), for the paper's fine-tuning comparisons.
- `daily_table`: a run's saved checkpoints on both German views, daily counts and 7-day sums, relative to the German config's references (`<run>/reports/compare-daily.csv`).
- `compare_daily`: the paper's live run's and comparison runs' `daily_table` in one table (`<ROOT>/runs/compare-daily.csv`), the paper's `compare_daily`.
- `horizon_table`: a run's saved checkpoints on every German horizon, the nowcasts (days 0 and before) and the forecasts (days after), relative to the German config's references (`<run>/reports/horizon-scores.csv`), for the paper's progress figure of the run.
- `coupling_germany`: a checkpoint's German forecasts' coupling scores and summary (`nowcasteval.coupling`), from its German unit, in `<SCORE_CACHE>/coupling/`: each coupling against independent draws, the foundation model's online step (`online`) among them.
- `score_table`: the rows of every checkpoint in `<SCORE_CACHE>/tables/`, keeping other rules' rows of the committed tables (BDBV rows already there stay).
- `land`: opt-in, commit the tables to `evidence/` and push, only from the evidence branch (`score.branch`), never main.

## 5. Report (`rules/report.smk`)

Reports read the other steps' outputs back for people, each with one script.

- `round_summaries`: one row per outbreak of a round (`r<N>/summaries/outbreaks.csv`, `workflow/scripts/grid_summaries.py`): design, drawn values, size, dynamics, R, delays and the phase at the training cut-offs.
- `grid_figures`: the rounds in `rounds` side by side, each summary's ECDF and the shares of patterns and phases.
- `sim_report`: a round's report, `r<N>/report/sim-report.html`; `sim_report_run` one report of every round in `rounds`; `sim_reports` all of them. `sim_diseases` gives the grid's training rounds' ranges against the published values by disease (`report/sim-diseases.csv`), for the SI's disease table.
- `german_report`: the run's checkpoints against zero-shot and randomwalknowcast on the German development half (`workflow/scripts/german_report.py`), in `<ROOT>/runs/<run>/reports/german/`.
- `hub_rounds`: our last `hub_rounds.rounds` hub rounds against every hub team on the hub's latest data, `<ROOT>/hub/hub-rounds.html` (`hub_rounds_report`, `workflow/scripts/hub_rounds_report.py`), copied to `REVIEW_DIR` when set. `hub_data` tidies the hub's data, `hub_submissions_table` reads which chain model went out from the ledger branch, and `hub_comparison` scores the hub's files as submitted beside `hub_rounds.models` as the fit store holds them for each round (`fit-store load --models`, then `compare-hub`). They rerun when `<ROOT>/hub/head.txt` changes, which `workflow/scripts/hub_watch.sh` writes when the hub's data commit moves before it builds the target. Not part of `all`.
- `hub_head`: the hub's data commit (`<ROOT>/hub/head.txt`) when a host has none, fetched with `gh`; `workflow/scripts/hub_watch.sh` keeps it current. `hub_data` reads the hub with the `gh` token.
- `ft_check`: the BDBV foundation-model check of `ft_check.checkpoint` (else `packages/nowcastfoundation/config/finetune.toml`'s pin), copied to `REVIEW_DIR` as `bdbv-ft-check.html`; `ft_check_page` builds one checkpoint's page, `<ROOT>/hub/ft-check/<ckpt>/bdbv-ft-check-<ckpt>.html` (`bdbvhub ft-check`, `bdbvhub.ftcheck`), copied to `REVIEW_DIR` under that name, so each checkpoint is its own target; `finetune` asks for the page of every released checkpoint (a segment's `published.json`). `ft_check_forecast` forecasts every hub release with one model into `<ROOT>/hub/ft-check/forecasts/<model>@<code key>` (`flowlib.scoring.ft_check_key`): `ft_check.models` once per code key, and `ft_check.finetuned` as `ft-<checkpoint>@<code key>`, so a new checkpoint reruns only it and a folder from other code is never reused. Chronos-2 models build the active-zone and by-province blocks. The page gives each model's pairwise relative skill on the CRPS and log CRPS, scaled to `ft_check.reference`, overall and by weeks back, with the forecasts and their 50% and 90% intervals. Not part of `all`.
- `rt_check`: the Rt sense check of a stored renewal fit before its `wk R_t` is submitted, `<rt_check.out_dir>/<model>-<origin>.html` (`bdbvhub rt-check`, `bdbvhub.reporting.rtcheck`): Rt by country, province and fitted zone over their cases, with BVDOutbreakSize's Rt. It reads the fit the store (`rt_check.store`) holds for the current code and is copied to `REVIEW_DIR` as `rt-check.html` when set. Not part of `all`.
- `this_round`: this week's round, `<ROOT>/hub/this-round.html` (`this_round_report`, `workflow/scripts/this_round_report.py`), copied to `REVIEW_DIR` when set: every candidate's forecasts and nowcasts against the data as of the round, the renewal candidates' wk R_t, the gate's pick and why, and the renewal fits' convergence; no scores. `this_round_inputs` downloads the latest Submit run on main's artefacts, fetches the fit store and archives every candidate's hub table for the round (`workflow/scripts/this_round_inputs.sh`). It reruns when `<ROOT>/hub/round/run.txt` or `round/store.txt` changes, which `workflow/scripts/hub_watch.sh` writes. Not part of `all`.
- `benchmarks`: opt-in, the seconds and peak memory of every forecast and score job and the seconds by model and date, as one CSV in `<SCORE_CACHE>/` (`flowlib.benchmarks`).

## 6. Dashboard (`workflow/flowlib/dashboard/`)

`task workflow:dashboard`, or `uv run --group paper python workflow/scripts/dashboard.py`, serves a page for a running fine-tune on the machine settings' `DASHBOARD_HOST` and `DASHBOARD_PORT` (default 127.0.0.1, port 8765); it is not a Snakemake job, so it never holds the workflow's lock.
It is one process: the page is rebuilt in the background and a client that leaves is dropped, so it needs no proxy or restart loop.

It runs two ways, through the one script (`flowlib.dashboard.serve`):

- with the run: `dashboard.serve: true` in the workflow config (the default), on a machine whose targets (`TARGETS`) include `finetune`, `simulate` or `push`, starts it from the run's own checkout and local config when the run starts, showing the runs it trains (the Snakefile's `onstart`), unless one is already serving (`<DASHBOARD_ROOT>/dashboard/serve.pid`, log beside it). It ends by itself when the coordinator (`task workflow:run`) ends, or with that Snakemake run when there is no coordinator, so it stays up across ticks and needs no worktree or tmux loop of its own;
- alone: `task workflow:dashboard`, from any checkout, in the foreground.

Either shows the runs under `DASHBOARD_ROOT` (default `ROOT`) that have trained there, the most recently active first, found again on each refresh, so a new run appears in the run picker by itself; `--run NAME` shows only the runs named.
The machine's monitor serves it under `/finetune/` (see Monitor), so the monitor's port is the one address to open or forward.

The same server shows the simulations under `DASHBOARD_ROOT` at `/sims/` (`flowlib.dashboard.sims`), and the monitor serves them under `/sims/`.
Its sources are each round's `rounds/r<N>/sims/` and each run's `runs/<run>/sims/`.
Each simulation is summarised from its first observation draw, every stratum summed: its grid factors and drawn values, each under its grid layer (infections, strata, observation, delays, reporting), its cases, strata, weeks, peak, mean delay and share reported within 7 days, and its weekly counts by reference date.
A batch is summarised once and cached in `<DASHBOARD_ROOT>/dashboard/sims/` until its files change; the index is refreshed every `dashboard.refresh_every` seconds.
The page has a summary (tiles and medians by the levels of a chosen factor), each layer's factors and drawn values against the whole source, every simulation's weekly counts on one panel and one panel per level, and a table.
Clicking a level, a histogram bin or a table row filters or selects; a deep dive shows one simulation's counts by stratum, its counts as known at three cut-offs, its delays, its reporting triangle, its infections, onsets and R where the batch kept them, and its settings by layer.
`task workflow:dashboard-static` builds the rule `dashboard`, a static render of the same page in `<ROOT>/reports/dashboard.html`.

The page has a sticky bar (section links, the live run's step, one run and reference selector every chart follows) and sections: Status (tiles), Loss (training loss, and validation loss on the full held-out set at step 0 and each checkpoint, by part, layout and level), Attention bias, Performance on real data (scores by study, score, target and source), Nowcasts (every dataset), Recent activity (with a mailbox) and Tables. Every chart has an info marker (`page/explain.json`).
The loss section also sets each checkpoint's and base Chronos-2's forecasts of a fixed sample of the run's held-out blocks against the truth, one block and row at a time (rules `validation_forecasts` and `validation_forecasts_base`, on the checkpoint scoring device below every other job; `flowlib.dashboard.validation`).

Settings:

- `workflow/config.yaml`'s `dashboard`: references (matched on rows by model and checkpoint, with a role: `baseline`, `start` or `other`), score studies and their labels, level order and defaults, the datasets (Germany: quantiles and WIS; BDBV is held out and never shown), what is held out (`held_out`: BDBV, the German held-out half from its split date, and the grid's test round, none of which the dashboard reads), the notes filter.
- `workflow/config.yaml`'s `runs`: the runs shown, each from `<ROOT>/runs/<name>`, the first live. `--run NAME` shows only the named runs, the first live.
- `config/local.env` (or the file `LOCAL_CONFIG` names, outside the checkout): `ROOT`, the scoring cache (`SCORE_CACHE`) the score table and each dataset's forecasts are read from, the mailbox (`MAILBOX`), where it listens (`DASHBOARD_HOST`, `DASHBOARD_PORT`) and whose runs it shows (`DASHBOARD_ROOT`).
- `--ref`: more references, or ones replacing the configured.

```sh
uv sync --group paper                                          # once
LOCAL_CONFIG=/path/to/machine.env task workflow:dashboard -- --run runa
# open the address it prints
```

## 7. Paper (`rules/paper.smk`, `paper/workflow/Snakefile`)

A module of this workflow (`use rule * from paper_workflow`), run from the repository root; its paths are from the root and its outputs are in `paper/results/<mode>/`.
`development` runs the development seeds and the German first half only, `paper` the held-out seeds and both German halves, and `render` renders `paper/main.qmd` and `paper/si.qmd` from the run `publish_paper` last copied to `paper/build/`.

- Data: `germany_fetch`, `germany_triangle`, `germany_dev_stats`, `ebola_hub`, `ebola_data` (the hub's releases, downloaded on the driver for `ebola_backtest`), `data`, and `synthetic_forecasts_development` and `synthetic_forecasts_paper` (the synthetic runner's forecasts of each run's seeds).
- Benchmarks: `germany_development`, `germany_heldout` (paper runs only), `ebola_backtest` and `ebola_backtest_draws`.
- Round releases: our submissions are read from the hub's clone (`ebola_hub`), and `round_releases` reads the models each round did not submit from that round's release (`round-<reference date>`, its `hub.zip`), with the submission ledger's table naming what each round submitted.
- Forecasts and scores: `forecasts`, `table`, `truth`, `scores`, `joint_scores` and `real_finetune`.
- Evidence read from git (`evidence_checkpoint_scores`), gathered by `evidence`.
- Outputs: `figures_synthetic`, `figures_germany`, `figures_ebola`, `score_tables_synthetic`, `score_tables_germany`, `score_tables_ebola`, `sim_examples`, `sim_realism` (the grid's training rounds against the German development half), `pairing_germany`, `pairing_synthetic` and `si_copula` (the walks' scores with and without pairing, and their German rolling $\rho_p$, on development data), `grid_tables`, `summary` and `publish_paper`.
- Progress figures: `progress_runa`, `progress_noslots`, `progress_nodelays` and `progress_realdata`, one per paper run (`live_run` and `comparisons` in `paper/workflow/config.yaml`), from the run's `run_progress` and `horizon_table` tables, or on Pages from the stored copies in the `finetune` set.
- Stored inputs: `release_paper_synthetic`, `release_paper_finetune`, `release_paper_simulations` and `release_paper_germany` publish each set a host makes as a `paper-<set>-<hash>` release, gathered by the opt-in target `paper_inputs`.

## A fine-tuning run

One run of the configured `train_runs` on the grid's rounds (`grid_rounds`), on a machine with a GPU, is one coordinator run of the `finetune` target.
Each tick simulates the batches the next segment needs, trains that segment, releases the checkpoint, and brings its German scores, the references (zero-shot Chronos-2 and randomwalknowcast), the German report and the BDBV check page up to date.

Its local config, kept outside the checkout, sets the run's folder and the machine's budget:

```sh
ROOT=$HOME/<run folder>
SCHEDULE_TARGETS=finetune
PUBLISH=1
SIMS_FROM=local
DEVICE=cuda
FORECAST_DEVICE=cpu
HEADROOM_CORES=<cores kept free>
HEADROOM_GB=<GB kept free>
SIM_SLOTS=<batches at once>
```

`HEADROOM_CORES` and `HEADROOM_GB` keep cores and memory free for the machine's user; the trainer, the scoring units (`forecast.threads` each) and the simulation batches share the rest.
Check the plan, then start the coordinator from a driver checkout at the commit to run:

```sh
LOCAL_CONFIG=<run folder>/local.env python3 workflow/scripts/tick.py --dry-run
LOCAL_CONFIG=<run folder>/local.env nohup task workflow:run > <run folder>/coordinator.log 2>&1 &
```

Stop it with `touch output/coordinator.stop` in the driver checkout, which ends it after the running tick.
The run serves its dashboard for as long as the coordinator lives, and the machine's monitor at `MONITOR_HOST:MONITOR_PORT` shows it at `/finetune/` (section 6).

## Scheduled runs

The workflow updates itself as results come in. Each machine runs its own targets in one persistent coordinator run, one tick after another; Snakemake reruns only what changed. With `COMPUTE=ssh` one coordinator runs every machine's jobs.

| Where | `SCHEDULE_TARGETS` | Does, each tick |
|---|---|---|
| the machine that scores (CPU) | `score land` | lists new checkpoint releases, scores them, rebuilds the table, and commits and pushes it to the evidence branch only if it changed |
| the machine that trains (GPU) | `finetune publish` | trains the next segment once its batches are there, and releases each new checkpoint |
| a simulating machine | `simulate release` | simulates the configured rounds and releases each batch |

A tick (`task workflow:tick`, `workflow/scripts/tick.py`) runs the pool targets (`simulate`, `push`, `finetune`, `score`) as one Snakemake run at nice 19 with the default profile and the machine's budget, so a free core or the GPU takes whichever queued job fits.
When jobs compete, the trainer goes first (priority 10^6, above any sum of the others, as Snakemake runs the ready jobs whose priorities sum highest), then quick hand-offs (90: the copies of the batches it reads, and each checkpoint's verification and release for its BDBV check), then the references every checkpoint is compared with (30), then each checkpoint's scoring and evaluation, newest first (20), then simulation batches (15); `workflow/flowlib/compute.py` sets the order.
Before it, the lists the pool plans from (the checkpoint releases, and on a machine without the rounds the batches it can have) are refreshed in a run of their own.
A failed refresh leaves the last lists, and the pool still runs.
The outward targets (`land`, `publish`, `release`) follow in their configured order.
A failed run skips `land` and `release`, so `land` never follows a failed `score`, but `publish` still releases the checkpoints saved.
A tick's run trains every segment within reach of the batches there are, so the next segment starts as soon as the one before is saved, while that checkpoint is scored, evaluated and published on the CPU.
Work that appears during a run, such as a landed round or a newly listed batch, waits for the next tick's run.
Snakemake locks the worktree it runs in, so each machine has one driver worktree.
A tick is skipped when another tick is still running (a lock in `output/`) or when less than `MIN_FREE_GB` of memory or `MIN_DISK_GB` of disk is free. Its log is `output/workflow-tick.log`.

`task workflow:run` (`workflow/scripts/coordinate.py`) is the coordinator run: it runs a tick, and as soon as one ends starts the next, after `COORDINATE_PAUSE` seconds (60), or `COORDINATE_BACKOFF` (600) after a failed tick. No timer runs it: start it once, under nohup or a service manager, and stop it with `touch output/coordinator.stop` (it ends after the running tick) or by its pid.

On CI, the site and the paper (`.github/workflows/pages.yml`) rebuild every 6 hours (`43 */6 * * *`, after the fit store's run) and whenever `evidence/` changes on main, so new checkpoint scores reach the docs and the paper without a manual step.

Nothing here needs an agent or a notes file; with `NOTIFY` set, rules still add their optional notes.

## Run on a Slurm cluster

With `COMPUTE=slurm` the workflow runs every job as a Slurm job, through Snakemake's Slurm executor plugin (`snakemake-executor-plugin-slurm`, in the `paper` group).
The driver, one Snakemake run, submits the jobs through the plugin and waits for them.
The nodes share the checkout and `ROOT`, so no files move.
Each rule asks for its own threads, memory, runtime and GPU (`workflow/profiles/slurm/config.yaml`).

### Steps

1. Clone the repository onto the shared filesystem, and run `uv sync --all-groups` once on the login node.
   The nodes need `git` and the checkout with its `.git`, as each simulation batch checks that the simulator is the committed one.
2. Copy `workflow/profiles/slurm/partitions.example.yaml` outside the repository.
   Rename its partitions and set their limits to the cluster's.
   The plugin sends a job that takes a GPU only to a partition with `max_gpu`, and other jobs to the first partition that fits.
3. Write `config/local.env` from `config/local.env.example`:

   ```sh
   COMPUTE=slurm
   ROOT=/shared/bdbv-rounds
   SLURM_ACCOUNT=<account>
   SLURM_PARTITIONS=/shared/site/partitions.yaml
   DEVICE=cuda
   FORECAST_DEVICE=cuda
   HF_HOME=/shared/hf
   ```

   Set `DEVICE`, and `FORECAST_DEVICE` if forecasting jobs should use GPUs, as the login node has no GPU to detect.
   Leave `SLURM_ACCOUNT` empty to let the plugin guess the account.
   Leave `SLURM_PARTITIONS` empty to send every job to `sinfo`'s default partition.
4. Fill `HF_HOME` with the base Chronos-2 model once, from a machine with internet access.
5. Set these in the shell that starts the driver, which Slurm passes to each job:

   ```sh
   export UV_NO_SYNC=1
   export UV_OFFLINE=1       # nodes without internet access
   export HF_HUB_OFFLINE=1   # once HF_HOME holds Chronos-2
   ```

6. Dry-run first: `task workflow:finetune -- -n` lists the jobs it would submit, with their resources.
7. Start the driver on the submit host under `tmux` or `nohup` with `task workflow:run`, which runs `SCHEDULE_TARGETS` one tick after another.
   Where login nodes forbid long processes, run it as a long Slurm job with few resources, for example `sbatch --time=7-0 --cpus-per-task=2 --mem=8G --wrap "task workflow:run"`.
   Stop it with `touch output/coordinator.stop`, which ends it after the running tick.

### What Slurm decides

Slurm places and sizes every job.
The one-machine caps do not apply: there is no `gpu`, core or memory budget and no cores or memory kept for the trainer (`TRAINER_CORES`).
Each GPU job asks for one GPU (`--gpus=1`), and Slurm sets `CUDA_VISIBLE_DEVICES` to it, so `workflow/flowlib/gpus.py` runs it as it is.
Runs do not wait on each other, so every run `finetune` trains (`train_runs`) trains at once, each trainer a Slurm job on a GPU node, as far as the cluster has GPUs free.
With `forecast.device: cuda`, forecasting jobs need `forecast.gpu: exclusive`, so each asks Slurm for its own GPU.

### Changing the job sizes

Each rule's memory, runtime and threads are in the profile's `set-resources` and `set-threads`.
To change them, copy `workflow/profiles/slurm` outside the repository, edit the copy and set `SNAKEMAKE_PROFILE` to it in `config/local.env`.
That profile then replaces the one `COMPUTE` gives, with the default resources, account and partitions file, so add those to the copy's `default-resources` and `slurm-partition-config`.

## Monitor

`task workflow:monitor` serves [Panoptes](https://github.com/panoptes-organization/panoptes), Snakemake's web monitor: every run's jobs, their state, what is running and what has failed, in one page.
It is each machine's one entry: a fine-tuning or simulating run starts it, unless one is already serving (`<ROOT>/monitor/serve.pid`), and it keeps serving after the run ends; it serves the fine-tuning dashboard at `/finetune/` and the simulations at `/sims/`, linked from its pages beside the DAG.
Every Snakemake run the tasks, ticks and paper scripts start reports to it through the logger plugin `snakemake-logger-plugin-panoptes` (`workflow/flowlib/monitor.py`), on by default.
Runs report to `MONITOR_URL`; another machine's runs report to the serving machine's address, and an empty `MONITOR_URL` switches reporting off.
The plugin drops events it cannot send, so a run goes on when no server is listening.
The page is served on `MONITOR_HOST` and `MONITOR_PORT`, with its database in `<ROOT>/monitor/`, all from `config/local.env`.
Panoptes has no DAG view, so the same server adds one at `/dag/` (`workflow/flowlib/monitor_server.py`, `workflow/flowlib/dag.py`).
For each `MONITOR_DAG` target (`NAME=TARGET`) it draws the rule graph and the job DAG, each job pending, running, done or failed.
A job Snakemake would not run is done; Panoptes gives the rest their state, so the colours follow the run on each visit.
The graphs are built with `MONITOR_DAG_OPTIONS` every `MONITOR_DAG_EVERY` seconds, without a lock; `task workflow:dag -- NAME=TARGET` builds one now.
In the workflow (`rules/monitor.smk`), `monitor_dag` builds one target's graphs into `<ROOT>/monitor/dag/<name>.json` and `monitor` every `MONITOR_DAG` target's; both are opt-in.
Each tick also writes its machine's status to `<ROOT>/status/<host>.json` every `STATUS_EVERY` seconds (`workflow/flowlib/runstatus.py`).
The fine-tuning dashboard shows the science (losses, scores, nowcasts, views), not jobs.

## CI

| Target | On CI | Why |
|---|---|---|
| tests and dry runs of every target (`task workflow:check`) | yes | fast, no network |
| the paper and its figures and tables | yes, as now (Pages) | from stored forecasts and pinned evidence |
| `score_table` from stored caches | yes, once the paper module lands | cheap |
| checkpoint forecasts, slots check | no | hours, needs the checkpoints |
| `simulate`, `finetune` | never | heavy R, GPU |
| `release`, `publish`, `land` | never | outward-facing |

`all` is the default target.

Back to the [repository README](../README.md).
