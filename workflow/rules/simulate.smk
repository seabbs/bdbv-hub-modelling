# Simulation rounds.
#
# Round r is the batches of `flowlib.segments.round_seeds`, each
# `batch` simulations of the grid (`simulate.grid_file`; each simulation
# the joint PyRenew model, nowcastrenewal.simulation) with exactly
# `draws` observation draws.
# Into
# <ROOT>/rounds/r<r>/sims/. Each batch gets a manifest row, the round's
# manifest joins them, the sim QC (`sim-qc`) checks the round
# against the grid's thresholds (its `qc`),
# and `round_ready` writes the seed lists and READY last. Every
# simulation is kept: the QC only records summaries (qc/).
#
# Targets: `simulate` (the rounds in `rounds`), and the opt-in `release`,
# which uploads each batch to the release sims-r<r>. Opt-in too:
# <ROOT>/rounds/bench/simulator-<cpu|cuda>-n<n>-s<seed>.json, the
# simulator's benchmark on one seeded batch of the grid
# (nowcastfoundation.simulator.bench).
#
# Two hosts can share the rounds (flowlib.hosts): each lists the seeds
# it takes in its local config (SIM_SEEDS; empty: all). A helping
# host's target is `push`, which sends each of its batches to the rounds
# host (REMOTE) with its manifest row; the rounds host's `simulate`
# simulates its own seeds and waits for the rest.


wildcard_constraints:
    round=r"\d+",
    seed=r"\d+",
    n=r"\d+",
    device=r"cpu|cuda",


rule simulate_batch:
    output:
        counts=f"{ROUNDS_DIR}/r{{round}}/sims/counts-{{seed}}.csv.gz",
        settings=f"{ROUNDS_DIR}/r{{round}}/sims/settings-{{seed}}.csv",
        latent=f"{ROUNDS_DIR}/r{{round}}/sims/latent-{{seed}}.csv.gz",
    log:
        f"{ROUNDS_DIR}/r{{round}}/sims/logs/{{seed}}.log",
    # In one pool, after the trainer (20) and before scoring (10 and
    # below): the trainer waits on these batches. Among themselves in
    # seed order, the order training reads them, so a run over several
    # rounds finishes each round before the next.
    priority: batch_priority
    retries: 1
    conda:
        TOOLS_ENV
    threads: 1
    # Batches are run by count, not by memory: at most
    # `--resources sim_slots=N` at once (SIM_SLOTS, 2 by default), which
    # the scheduler starts as slots free up.
    resources:
        mem_mb=0,
        sim_slots=1,
    params:
        sims=lambda wc, output: Path(output.counts).parent,
        batch=SIM["batch"],
        draws=SIM["draws"],
        grid=GRID_FILE,
        engine=" ".join(simulator.ENGINE),
        jax_cache=LOCAL["JAX_CACHE"],
        jax_cache_gb=LOCAL["JAX_CACHE_GB"],
        jax_env=simulator.jax_cache_env(LOCAL["JAX_CACHE"]),
    shell:
        # The simulator (its code and the grid) must be the committed one,
        # as the manifest records the commit. A retry keeps the first
        # attempt's log. The simulator writes temporary files to a folder of
        # the batch's own, removed however the batch ends. JAX runs on one
        # thread, as the job has one, and shares its compiles with the
        # host's other batches (flowlib.simulator.jax_cache_env).
        """
        if ! git diff --quiet HEAD -- {params.engine} {params.grid}; then
            echo "the simulator or its grid differs from HEAD" >{log}
            exit 1
        fi
        [ -s {log} ] && mv {log} {params.sims}/logs/{wildcards.seed}.first.log
        tmp={params.sims}/tmp/{wildcards.seed}
        rm -rf "$tmp" && mkdir -p "$tmp"
        trap 'rm -rf "$tmp"' EXIT
        workflow/scripts/trim_jax_cache.sh "{params.jax_cache}" {params.jax_cache_gb}
        uv=$(command -v uv)
        TMPDIR="$tmp" ENW_DRAWS={params.draws} JAX_PLATFORMS=cpu {params.jax_env} \
            XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
            workflow/scripts/timed.sh \
            "$uv" run --frozen python -m nowcastfoundation.simulator \
            {params.grid} {params.batch} {wildcards.seed} {params.sims} >{log} 2>&1
        gzip -t {output.counts}
        """


rule simulator_benchmark:
    # Two passes of one batch, first then again, as a simulate_batch job
    # runs (one thread, the round's draws) on the given JAX device. The
    # benchmarks of one simulator code share a compile cache of their own,
    # kept between runs: the first pass is cold on an empty cache and,
    # after another seed's benchmark, times what a batch on a host with a
    # warm cache sees. Each code has its own folder, so two versions
    # benchmarked from one ROOT never read each other's compiles.
    output:
        f"{ROUNDS_DIR}/bench/simulator-{{device}}-n{{n}}-s{{seed}}.json",
    log:
        f"{ROUNDS_DIR}/bench/logs/simulator-{{device}}-n{{n}}-s{{seed}}.log",
    conda:
        TOOLS_ENV
    threads: 1
    # A cuda benchmark takes a GPU, as a scheduler must know.
    resources:
        gpu=lambda wc: int(wc.device == "cuda"),
    params:
        draws=SIM["draws"],
        grid=GRID_FILE,
        jax_env=lambda wc: simulator.jax_cache_env(
            f"{ROUNDS_DIR}/bench/jax-cache-{wc.device}-{simulator.engine_id()}"
        ),
    shell:
        """
        ENW_DRAWS={params.draws} JAX_PLATFORMS={wildcards.device} \
            {params.jax_env} \
            XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
            uv run --frozen python -m nowcastfoundation.simulator.bench \
            {params.grid} {wildcards.n} {wildcards.seed} {output} >{log} 2>&1
        """


rule await_batch:
    # A batch another host simulates (not in SIM_SEEDS): done once that
    # host has pushed its files and then its row to this round's
    # manifest. One waits at a time (`--resources awaiting=1`).
    output:
        f"{ROUNDS_DIR}/r{{round}}/arrived/{{seed}}.done",
    log:
        f"{ROUNDS_DIR}/r{{round}}/logs/await-{{seed}}.log",
    localrule: True
    conda:
        TOOLS_ENV
    resources:
        awaiting=1,
    params:
        manifest=f"{ROUNDS_DIR}/r{{round}}/manifest.csv",
        poll=60,
    shell:
        """
        exec >{log} 2>&1
        until grep -q '^{wildcards.round},{wildcards.seed},' \
            {params.manifest} 2>/dev/null; do
            sleep {params.poll}
        done
        touch {output}
        """


rule batch_record:
    input:
        unpack(batch_files),
    output:
        f"{ROUNDS_DIR}/r{{round}}/records/record-{{seed}}.csv",
    log:
        f"{ROUNDS_DIR}/r{{round}}/logs/record-{{seed}}.log",
    localrule: True
    params:
        sims=f"{ROUNDS_DIR}/r{{round}}/sims",
        old=f"{ROUNDS_DIR}/r{{round}}/manifest.csv",
    run:
        write_record(input, output, params, wildcards)


rule round_manifest:
    input:
        round_records,
    output:
        f"{ROUNDS_DIR}/r{{round}}/manifest.csv",
    log:
        f"{ROUNDS_DIR}/r{{round}}/logs/manifest.log",
    localrule: True
    run:
        manifest.join(input, output[0])


rule sim_qc:
    input:
        rules.round_manifest.output,
    output:
        summary=f"{ROUNDS_DIR}/r{{round}}/qc/summary.txt",
    log:
        f"{ROUNDS_DIR}/r{{round}}/qc/run.log",
    conda:
        TOOLS_ENV
    params:
        round_dir=lambda wc, input: Path(input[0]).parent,
    shell:
        "uv run sim-qc {params.round_dir} > {log} 2>&1"


rule round_ready:
    input:
        rules.sim_qc.output.summary,
    output:
        seeds=f"{ROUNDS_DIR}/r{{round}}/seeds.txt",
        ready=f"{ROUNDS_DIR}/r{{round}}/READY",
    log:
        f"{ROUNDS_DIR}/r{{round}}/logs/ready.log",
    localrule: True
    run:
        mark_ready(input, output, params, wildcards)


# Refuses to upload into the release of an earlier grid's round
# (`retired_rounds`, flowlib.grids).
GRID_GUARD = """
        if [ "{params.owner}" = retired ]; then
            echo "{params.tag} holds an earlier grid's batches: not uploading"
            exit 1
        fi
"""


rule release_batch:
    # Opt-in (target `release`): uploads a batch to the release sims-r<r>,
    # so another machine can train on it before the round ends.
    input:
        unpack(batch_files),
        record=rules.batch_record.output,
    output:
        touch(f"{ROUNDS_DIR}/r{{round}}/released/{{seed}}.done"),
    log:
        f"{ROUNDS_DIR}/r{{round}}/logs/release-{{seed}}.log",
    localrule: True
    conda:
        TOOLS_ENV
    params:
        repo=SIM["repo"],
        tag="sims-r{round}",
        owner=release_owner,
        counts=rules.simulate_batch.output.counts,
        settings=rules.simulate_batch.output.settings,
    shell:
        """
        exec >{log} 2>&1
        """
        + GRID_GUARD
        + """
        gh release view {params.tag} -R {params.repo} >/dev/null 2>&1 \
            || gh release create {params.tag} -R {params.repo} --prerelease \
                --title "Simulations, round {wildcards.round}" \
                --notes "Fine-tuning simulations of round {wildcards.round}, one batch per asset, with their manifest rows." \
                >/dev/null 2>&1 \
            || gh release view {params.tag} -R {params.repo} >/dev/null
        gh release upload {params.tag} -R {params.repo} --clobber \
            {params.counts} {params.settings} {input.record}
        """


rule push_batch:
    # A helping host's batch, sent to the rounds host's round: its files,
    # then its manifest row (flowlib.hosts.push).
    input:
        counts=rules.simulate_batch.output.counts,
        settings=rules.simulate_batch.output.settings,
    output:
        touch(f"{ROUNDS_DIR}/r{{round}}/sent/{{seed}}.done"),
    log:
        f"{ROUNDS_DIR}/r{{round}}/logs/push-{{seed}}.log",
    retries: 3
    localrule: True
    params:
        sims=lambda wc, input: str(Path(input.counts).parent),
        there=lambda wc: f"{REMOTE_ROUNDS}/r{wc.round}",
    run:
        push_record(input, output, params, wildcards)


rule push:
    # A helping host's target: its seeds (SIM_SEEDS) of the rounds in
    # `rounds`.
    input:
        [
            f"{ROUNDS_DIR}/r{r}/sent/{s}.done"
            for r in config["rounds"]
            for s in seeds_of(r)
            if here(s)
        ],


rule simulate:
    input:
        expand(f"{ROUNDS_DIR}/r{{round}}/READY", round=config["rounds"]),


rule release:
    input:
        # Never an earlier grid's round: its release holds that grid's
        # batches.
        release_check,
        [
            f"{ROUNDS_DIR}/r{r}/released/{s}.done"
            for r in config["rounds"]
            for s in seeds_of(r)
        ],
