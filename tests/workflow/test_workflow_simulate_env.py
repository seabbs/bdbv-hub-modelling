"""The simulator runs as the committed Python, with JAX on one thread."""

import re

from flowlib import simulator
from test_workflow_dryrun import dry_run

GRID = "config/finetune/grid.yaml"


def test_the_grid_identifier_is_read_from_the_grid(tmp_path):
    assert re.fullmatch(r"[0-9a-f]{12}", simulator.grid_id(GRID))
    other = tmp_path / "x.yaml"
    other.write_text("max_triangle_cells: 1\n")
    assert simulator.grid_id(other) != simulator.grid_id(GRID)


def test_a_batch_runs_the_python_simulator_with_no_r(tmp_path):
    batch = tmp_path / "rounds" / "r4" / "sims" / "counts-30400.csv.gz"
    out = dry_run(tmp_path, str(batch), extra=["-p", "--quiet", "host"])
    assert out.returncode == 0, out.stderr
    flat = " ".join(out.stdout.split("rule simulate_batch", 1)[1].split())
    assert f"-m nowcastfoundation.simulator {GRID}" in flat
    assert "JAX_PLATFORMS=cpu" in flat
    for r in ("Rscript", "ENW_LIB", "RSCRIPT"):
        assert r not in flat, r
    # The whole simulator, the joint model included, must be committed.
    for folder in simulator.ENGINE:
        assert folder in flat
    assert "nowcastfoundation/r" not in " ".join(simulator.ENGINE)


def test_heavy_jobs_wait_below_the_load_ceiling(tmp_path):
    (tmp_path / "local.env").write_text(
        f"ROOT={tmp_path}\nHOST_NAME=test-host\nMAX_LOAD=12\n"
    )
    batch = tmp_path / "rounds" / "r4" / "sims" / "counts-30400.csv.gz"
    out = dry_run(tmp_path, str(batch), extra=["-p", "--quiet", "host"])
    assert out.returncode == 0, out.stderr
    command = " ".join(out.stdout.split("rule simulate_batch", 1)[1].split())
    sims = tmp_path / "rounds" / "r4" / "sims"
    assert f"wait_for_room.sh 20 16 {sims} 12" in command


def test_the_compile_cache_env_names_the_folder_and_its_cap():
    env = simulator.jax_cache_env("/x/jax-cache", "2")
    assert "JAX_COMPILATION_CACHE_DIR=/x/jax-cache" in env
    assert "JAX_COMPILATION_CACHE_MAX_SIZE=2000000000" in env
    # Every compile is cached, however small or quick.
    assert "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0" in env
    assert "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0" in env
    assert simulator.jax_cache_env("", "2") == ""


def test_batches_share_a_compile_cache_under_root(tmp_path):
    (tmp_path / "local.env").write_text(
        f"ROOT={tmp_path}\nHOST_NAME=test-host\n"
    )
    batch = tmp_path / "rounds" / "r4" / "sims" / "counts-30400.csv.gz"
    out = dry_run(tmp_path, str(batch), extra=["-p", "--quiet", "host"])
    assert out.returncode == 0, out.stderr
    command = " ".join(out.stdout.split("rule simulate_batch", 1)[1].split())
    assert f"JAX_COMPILATION_CACHE_DIR={tmp_path}/jax-cache" in command


def test_an_empty_cache_setting_turns_the_cache_off(tmp_path):
    (tmp_path / "local.env").write_text(
        f"ROOT={tmp_path}\nHOST_NAME=test-host\nJAX_CACHE=\n"
    )
    batch = tmp_path / "rounds" / "r4" / "sims" / "counts-30400.csv.gz"
    out = dry_run(tmp_path, str(batch), extra=["-p", "--quiet", "host"])
    assert out.returncode == 0, out.stderr
    assert "JAX_COMPILATION_CACHE_DIR" not in out.stdout
