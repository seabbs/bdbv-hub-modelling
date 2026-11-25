# Run on a Slurm cluster

The workflow runs every job as a Slurm job with `COMPUTE=slurm`.
The driver, one Snakemake run, submits the jobs through `snakemake-executor-plugin-slurm` and waits for them.
The nodes share the checkout and `ROOT`, so no files move.
Each rule asks for its own threads, memory, runtime and GPU (`workflow/profiles/slurm/config.yaml`).

## Steps

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

## What Slurm decides

Slurm places and sizes every job.
The one-machine caps do not apply: there is no `gpu` or core budget and no cores kept for the trainer (`TRAINER_CORES`).
Each GPU job asks for one GPU (`--gpus=1`), and Slurm sets `CUDA_VISIBLE_DEVICES` to it, so `workflow/flowlib/gpus.py` runs it as it is.
Runs do not wait on each other, so every run `finetune` trains (`train_runs`) trains at once, each trainer a Slurm job on a GPU node, as far as the cluster has GPUs free.
With `forecast.device: cuda`, forecasting jobs need `forecast.gpu: exclusive`, so each asks Slurm for its own GPU.

## Changing the job sizes

Each rule's memory, runtime and threads are in the profile's `set-resources` and `set-threads`.
To change them, copy `workflow/profiles/slurm` outside the repository, edit the copy and set `SNAKEMAKE_PROFILE` to it in `config/local.env`.
That profile then replaces the one `COMPUTE` gives, with the default resources, account and partitions file, so add those to the copy's `default-resources` and `slurm-partition-config`.
