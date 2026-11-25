# Slurm profile

`COMPUTE=slurm` runs every workflow job as a Slurm job with this profile, through Snakemake's Slurm executor plugin (`snakemake-executor-plugin-slurm`).
The steps to set it up and launch it are in [Run on a Slurm cluster](../../README.md#run-on-a-slurm-cluster).

- `config.yaml`: the profile, with each rule's threads, memory and runtime.
- `partitions.example.yaml`: the example site file. Copy it outside the repository, set the cluster's partitions in it and point `SLURM_PARTITIONS` in `config/local.env` at the copy. The plugin sends each GPU job to a partition with GPUs.

The account goes in `SLURM_ACCOUNT` in `config/local.env`.
