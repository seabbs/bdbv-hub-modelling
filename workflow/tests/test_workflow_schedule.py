"""Scheduled workflow runs: what a tick runs."""

import pytest
from flowlib import schedule


def test_a_tick_runs_this_machine_s_targets_in_order():
    cfg = {
        "SCHEDULE_TARGETS": "score land",
        "CORES": "4",
        "MEM_MB": "32768",
        "SIM_SLOTS": "2",
    }
    got = schedule.plan(cfg)
    assert [t for t, _, _ in got] == ["refresh", "score", "land"]
    score = got[1][1]
    assert got[1][2] is True
    assert score[:3] == ["nice", "-n", "19"]
    assert "workflow/profiles/default" in score
    assert score[score.index("--cores") :][:2] == ["--cores", "4"]
    assert "mem_mb=32768" in score


def test_every_target_uses_the_default_profile():
    got = schedule.plan({"SCHEDULE_TARGETS": "simulate finetune publish"})
    assert all("workflow/profiles/default" in cmd for _, cmd, _ in got)


def test_with_workers_a_tick_coordinates_them(tmp_path, monkeypatch):
    monkeypatch.setenv("REMOTE_STATE", str(tmp_path))
    cfg = {
        "SCHEDULE_TARGETS": "simulate",
        "COMPUTE": "ssh",
        "WORKERS": "server=local",
        "ROOT": str(tmp_path),
        "CORES": "4",
        "MEM_MB": "8192",
    }
    (_, cmd, _), *_ = schedule.plan(cfg)
    assert "workflow/profiles/ssh" in cmd
    assert cmd[cmd.index("--jobs") + 1] == "4"
    assert (tmp_path / "budgets.json").is_file()


def snakemake_targets(cmd):
    """The targets of a planned Snakemake command."""
    start = cmd.index("snakemake") + 1
    return cmd[start : cmd.index("--workflow-profile")]


def test_the_pool_targets_are_one_snakemake_run():
    # One run over the machine's whole budget, so a free core or the GPU
    # takes whichever queued job fits.
    got = schedule.plan(
        {"SCHEDULE_TARGETS": "simulate finetune score land publish"}
    )
    assert [t for t, _, _ in got] == [
        "refresh",
        "simulate finetune score",
        "land",
        "publish",
    ]
    pool = got[1][1]
    assert snakemake_targets(pool) == ["simulate", "finetune", "score"]
    assert pool.count("--workflow-profile") == 1
    assert "--forcerun" not in pool


def test_the_outward_targets_keep_their_order_after_the_pool():
    got = schedule.plan({"SCHEDULE_TARGETS": "publish simulate release"})
    assert [t for t, _, _ in got] == ["simulate", "publish", "release"]


def test_a_pool_without_score_lists_no_releases():
    got = schedule.plan({"SCHEDULE_TARGETS": "simulate finetune"})
    assert "release_list" not in got[0][1]


def test_land_and_release_need_the_pool_publish_does_not():
    # publish reads only saved checkpoints, so it never retries a failed
    # job; land and release would rerun failed scoring or simulation.
    assert {"land", "release"} == schedule.NEEDS_POOL


def test_no_targets_means_nothing_runs():
    assert schedule.plan({"SCHEDULE_TARGETS": ""}) == []


def test_unknown_targets_are_an_error():
    with pytest.raises(ValueError, match="nope"):
        schedule.plan({"SCHEDULE_TARGETS": "score nope"})


def test_a_tick_waits_for_room(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemAvailable:   1048576 kB\n")  # 1 GB
    cfg = {"MIN_FREE_GB": "20", "MIN_DISK_GB": "0"}
    ok, why = schedule.room(cfg, tmp_path, meminfo)
    assert not ok
    assert "memory" in why
    ok, _ = schedule.room(
        {"MIN_FREE_GB": "0", "MIN_DISK_GB": "0"}, tmp_path, meminfo
    )
    assert ok


def test_no_meminfo_skips_the_memory_check(tmp_path):
    ok, _ = schedule.room(
        {"MIN_FREE_GB": "999", "MIN_DISK_GB": "0"}, tmp_path, tmp_path / "none"
    )
    assert ok


def test_runs_never_overlap(tmp_path):
    lock = tmp_path / "tick.lock"
    with schedule.locked(lock) as first:
        assert first
        with schedule.locked(lock) as second:
            assert not second
    with schedule.locked(lock) as again:
        assert again


def test_the_listings_are_refreshed_in_a_run_of_their_own():
    # Snakemake 9 ends a run after a forced checkpoint without the jobs
    # it unlocks, so the pool must plan from lists refreshed beforehand.
    got = schedule.plan(
        {"SCHEDULE_TARGETS": "finetune score", "SIMS_FROM": "remote"}
    )
    (name, refresh, _), (_, pool, _) = got[:2]
    assert name == "refresh"
    lists = {"release_list"}
    assert set(snakemake_targets(refresh)) == lists
    assert set(refresh[refresh.index("--forcerun") + 1 :]) == lists
    assert "--forcerun" not in pool


def test_finetune_needs_no_refresh_step():
    # It lists its batches as it plans, wherever they come from.
    got = schedule.plan({"SCHEDULE_TARGETS": "finetune", "SIMS_FROM": "remote"})
    assert [t for t, _, _ in got] == ["finetune"]
    assert "--forcerun" not in got[0][1]


def test_a_failed_refresh_still_leaves_the_pool_to_run():
    # Its lists are then those of the last tick.
    got = schedule.plan({"SCHEDULE_TARGETS": "score"})
    assert got[0][0] == "refresh"
    assert got[0][2] is True


def test_the_paper_is_not_a_tick_target():
    # The paper run pins the checkpoint into tracked config and copies
    # into docs/, so it is run on purpose (`task paper`), not on a timer.
    with pytest.raises(ValueError, match="paper"):
        schedule.plan({"SCHEDULE_TARGETS": "score paper"})


def test_a_tick_plans_every_segment_in_reach():
    # The next segment trains as soon as the one before is saved, beside
    # that checkpoint's scoring, not a tick later.
    for targets in ("simulate finetune", "score"):
        got = schedule.plan({"SCHEDULE_TARGETS": targets})
        assert "--config" not in got[-1][1]


def test_a_step_says_whether_it_is_the_pool():
    got = schedule.plan({"SCHEDULE_TARGETS": "score land publish"})
    assert [pool for _, _, pool in got] == [True, True, False, False]


def test_a_tick_does_not_build_the_dashboard():
    # The static page has no inputs, so a tick on file times built it once
    # and never again; a run serves it live (dashboard.serve) and
    # `task workflow:dashboard-static` writes the page.
    assert "dashboard" not in schedule.TARGETS
    assert "dashboard" not in schedule.POOL


def test_a_machine_still_listing_the_dashboard_keeps_ticking(capsys):
    steps = schedule.plan({"SCHEDULE_TARGETS": "score dashboard"})
    assert all("dashboard" not in step[1] for step in steps)
    assert "no longer a tick target" in capsys.readouterr().err
