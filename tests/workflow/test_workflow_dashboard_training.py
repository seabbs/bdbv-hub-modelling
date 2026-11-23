"""A run's schedule and losses, for the status tiles and loss plots."""

import json
from datetime import datetime

import pytest
from flowlib.dashboard import logs, runs, training


def test_the_log_reads_any_source_names(tmp_path):
    path = tmp_path / "train.log"
    path.write_text(
        "2026-11-17 10:00:00 [train] step 200: loss 0.5, held-out (first 100)"
        " 0.4, by source 3day 0.31, daily+univariate 0.29, weekly 0.12\n"
    )
    row = logs.Log(path).rows()[0]
    assert row["src"] == {
        "3day": 0.31,
        "daily+univariate": 0.29,
        "weekly": 0.12,
    }


def test_run_params_from_the_settings_and_history(tmp_path):
    n = tmp_path / "train" / "n250"
    n.mkdir(parents=True)
    settings = {
        "learning_rate": 1e-5,
        "lora": {"r": 8},
        "series_bias": {"learning_rate": 1e-3},
        "lr_decay": "linear to zero over the outbreaks",
    }
    (n / "settings.json").write_text(json.dumps(settings))
    hist = [{"step": 100, "lr": {"LoRA": 9e-6, "bias": 9e-4}}]
    got = training.run_params(tmp_path, hist, 100, (250, 500))
    assert got["groups"] == {"LoRA": 1e-5, "bias": 1e-3}
    assert got["current"]["bias"] == 9e-4
    assert got["lr_outbreaks"] == 500
    assert got["prefixes"] == [250, 500]


def test_steps_per_simulation_follow_the_last_checkpoint():
    cks = [("n500", {"steps": 2000, "n_outbreaks": 500}, None)]
    assert training.steps_per_simulation(cks) == 4.0
    assert training.steps_per_simulation([]) == 1.0


def test_loss_runs_take_held_out_by_source_from_history(tmp_path):
    folder = tmp_path / "run"
    (folder / "train").mkdir(parents=True)
    (folder / "train" / "history.json").write_text(
        json.dumps(
            [
                {"step": 0, "held_out_loss": 0.5, "by_source": {"weekly": 0.3}},
                {
                    "step": 200,
                    "loss": 0.45,
                    "by_source": {"weekly": 0.25, "3day": 0.4},
                },
            ]
        )
    )
    run = runs.make_run("r", str(folder), live=True)
    got = training.loss_run(
        run,
        rows=[],
        checkpoints=[],
        sims=lambda s: s / 4,
        base=None,
        colour="#111",
    )
    # Against simulations trained: 4 steps a simulation.
    assert got["bysrc"] == {"weekly": [[50, 0.25]], "3day": [[50, 0.4]]}
    assert got["full0"] == 0.5
    assert got["src0"] == {"weekly": 0.3}
    assert got["train"] == [[50, 0.45]]


def test_loss_runs_carry_the_nowcast_and_forecast_parts(tmp_path):
    folder = tmp_path / "run"
    (folder / "train").mkdir(parents=True)
    (folder / "train" / "history.json").write_text(
        json.dumps(
            [
                {
                    "step": 0,
                    "held_out_loss": 0.5,
                    "held_out_loss_nowcast": 0.2,
                    "held_out_loss_forecast": 0.3,
                },
            ]
        )
    )
    run = runs.make_run("r", str(folder), live=True)
    rows = [
        {
            "step": 100,
            "loss": 0.5,
            "held": 0.4,
            "parts": {"nowcast": 0.1, "forecast": 0.3},
        },
        {"step": 200, "loss": 0.5, "held": 0.39, "parts": {}},
    ]
    cks = [
        (
            "n250",
            {
                "steps": 1000,
                "held_out_loss": 0.35,
                "held_out_loss_nowcast": 0.1,
                "held_out_loss_forecast": 0.25,
            },
            None,
        ),
        ("n500", {"steps": 2000, "held_out_loss": 0.3}, None),
    ]
    got = training.loss_run(
        run,
        rows=rows,
        checkpoints=cks,
        sims=lambda s: s / 4,
        base=None,
        colour="#111",
    )
    assert got["parts0"] == {"nowcast": 0.2, "forecast": 0.3}
    assert got["bparts0"] == {"nowcast": 0.2, "forecast": 0.3}
    assert got["held_parts"] == {
        "nowcast": [[25, 0.1]],
        "forecast": [[25, 0.3]],
    }
    # Checkpoints are at their own simulations.
    assert got["ck_parts"] == {
        "nowcast": [[250, 0.1]],
        "forecast": [[250, 0.25]],
    }
    assert got["ck"] == [[250, 0.35], [500, 0.3]]


def test_steps_map_to_simulations_from_checkpoints_and_chunks():
    cks = [("n250", {"steps": 1500, "n_outbreaks": 250}, None)]
    # Chunks: 250 sims in 1500 steps (also the checkpoint), then 250 in
    # 1000; a chunk after a gap in the logs is not used.
    chunks = {0: (250, 1500), 250: (500, 1000), 750: (1000, 900)}
    pairs = training.anchors(cks, chunks)
    assert pairs == [(0, 0), (1500, 250), (2500, 500)]
    assert training.sims_at(pairs, 750) == 125
    assert training.sims_at(pairs, 2000) == 375
    # Past the last pair, at the mean rate so far: 5 steps a simulation.
    assert training.sims_at(pairs, 3000) == 600


def test_a_live_run_before_its_first_checkpoint_maps_through_its_chunks():
    pairs = training.anchors([], {0: (250, 2000)})
    assert training.sims_at(pairs, 1000) == 125


def test_with_nothing_known_steps_map_at_the_given_rate():
    pairs = training.anchors([], {})
    assert training.sims_at(pairs, 600, spo=6.0) == 100


def test_a_run_from_the_workflow_decays_over_its_simulations(tmp_path):
    # The settings say the rate falls over the run's simulations
    # (`default`).
    n = tmp_path / "train" / "n250"
    n.mkdir(parents=True)
    (n / "settings.json").write_text(
        json.dumps(
            {
                "learning_rate": 1e-5,
                "lr_decay": "linear to zero over the outbreaks",
            }
        )
    )
    got = training.run_params(tmp_path, [], 3000, (250, 8400), sims=2100)
    assert got["lr_outbreaks"] == 8400
    assert got["current"]["adapter"] == pytest.approx(1e-5 * 0.75)


def test_the_step_a_checkpoint_is_saved_at_comes_from_planned_chunks():
    cks = [("n250", {"steps": 1500, "n_outbreaks": 250}, None)]
    pairs = training.anchors(cks, {0: (250, 1500), 250: (500, 1400)})
    # n500's chunk is planned: its step is known.
    assert training.step_at(pairs, 500) == (2900, True)
    # n750 is not: at the run's mean rate so far, 5.8 steps a simulation.
    assert training.step_at(pairs, 750) == (4350, False)
    assert training.step_at([(0, 0)], 250, spo=4.0) == (1000, False)


def test_the_rate_uses_logged_steps_and_the_status_over_a_long_enough_window():
    # Evaluations every 400 steps leave one logged step in 30 minutes; the
    # status file's step and time make the second point.
    t0 = 1_000_000.0
    points = [(t0, 800), (t0 + 2400, 1200), (t0 + 2400 + 900, 1350)]
    rate, span = training.recent_rate(points)
    assert rate == pytest.approx(550 / 3300)
    assert span == pytest.approx(3300)
    # With points only far apart, the last two are used.
    rate, _ = training.recent_rate([(t0, 0), (t0 + 20000, 4000)])
    assert rate == pytest.approx(0.2)
    # One point, or none: no rate.
    assert training.recent_rate([(t0, 5)]) == (None, None)
    assert training.recent_rate([]) == (None, None)


def test_loss_runs_carry_checkpoint_losses_by_layout_and_level(tmp_path):
    # The Loss type selector draws by layout and by level too.
    folder = tmp_path / "run"
    (folder / "train").mkdir(parents=True)
    (folder / "train" / "history.json").write_text(
        json.dumps(
            [
                {
                    "step": 0,
                    "held_out_loss": 0.5,
                    "by_level": {"national": 0.4, "leaf": 0.6},
                }
            ]
        )
    )
    run = runs.make_run("r", str(folder), live=True)
    cks = [
        (
            "n250",
            {
                "steps": 1000,
                "n_outbreaks": 250,
                "held_out_loss": 0.35,
                "held_out_loss_by_source": {"daily": 0.3, "weekly": 0.2},
                "held_out_loss_by_level": {"national": 0.3, "leaf": 0.5},
            },
            None,
        )
    ]
    got = training.loss_run(
        run, [], cks, sims=lambda s: s / 4, base=None, colour="#111"
    )
    assert got["ck_src"] == {"daily": [[250, 0.3]], "weekly": [[250, 0.2]]}
    assert got["ck_lvl"] == {"national": [[250, 0.3]], "leaf": [[250, 0.5]]}
    assert got["lvl0"] == {"national": 0.4, "leaf": 0.6}
    assert got["blvl0"] == got["lvl0"]


def test_an_offloaded_run_s_losses_reach_the_loss_plot(tmp_path):
    # WP10: the trainer logs no held-out losses; the evaluate rules write
    # them per step, and the plot reads them as if logged.
    folder = tmp_path / "run"
    train = folder / "train"
    (train / "evals").mkdir(parents=True)
    (train / "history.json").write_text(
        json.dumps([{"step": 0}, {"step": 200, "loss": 0.45}])
    )
    (train / "evals" / "step-0000000.json").write_text(
        json.dumps(
            {"step": 0, "held_out_loss": 0.5, "by_source": {"weekly": 0.3}}
        )
    )
    (train / "evals" / "step-0000200.json").write_text(
        json.dumps({"step": 200, "by_source": {"weekly": 0.25}})
    )
    run = runs.make_run("r", str(folder), live=True)
    got = training.loss_run(
        run,
        rows=[],
        checkpoints=[],
        sims=lambda s: s / 4,
        base=None,
        colour="#111",
    )
    assert got["full0"] == 0.5
    assert got["bysrc"] == {"weekly": [[50, 0.25]]}
    assert got["train"] == [[50, 0.45]]


def test_block_views_are_summarised_from_the_run_s_record(tmp_path):
    train = tmp_path / "train"
    train.mkdir()
    # No record: no system was over the block limit.
    assert training.views_summary(tmp_path) is None
    (train / "views.csv").write_text(
        "part,outbreak,source,cut,kind,places_before,places_after,"
        "series_before,series_after,collapsed\n"
        "held-out,1-0001,daily,3,top,40,6,3990,300,strata\n"
        "held-out,1-0001,daily,3,subtree,40,12,3990,900,\n"
        "1-100,2-0001,weekly,0,top,30,1,2000,40,strata+places\n"
        "1-100,2-0001,weekly,0,subtree,30,9,2000,800,\n"
        "1-100,2-0001,weekly,0,subtree,30,9,2000,820,\n"
    )
    assert training.views_summary(tmp_path) == {
        "systems_over_limit": 2,
        "held_out": 1,
        "blocks": 5,
        "largest_series_before": 3990,
        "largest_series_after": 900,
    }


def test_an_offloaded_run_s_steps_are_read_without_a_held_out_loss(tmp_path):
    path = tmp_path / "n100.log"
    path.write_text(
        "2026-11-22 12:22:31 [train] step 200: loss 0.0792, held-out "
        "(monitor, by_source) left to the evaluator, 1078 s\n"
    )
    row = logs.Log(path).rows()[0]
    assert row["step"] == 200
    assert row["loss"] == 0.0792
    assert row["held"] is None


def test_the_mean_rate_leaves_out_the_gaps_of_restarts():
    t0 = 1_000_000.0
    # 100 steps in 100 s, a stop of two hours, then 100 steps in 50 s.
    points = [(t0, 0), (t0 + 100, 100), (t0 + 7300, 100), (t0 + 7350, 200)]
    assert training.mean_rate(points) == pytest.approx(200 / 150)
    assert training.mean_rate([(t0, 5)]) is None


def test_an_offloaded_run_s_step_evaluations_are_its_held_out_line(tmp_path):
    # The trainer logs no held-out loss when it offloads evaluation; each
    # step's subset loss and its parts come from train/evals.
    folder = tmp_path / "run"
    train = folder / "train"
    (train / "evals").mkdir(parents=True)
    (train / "history.json").write_text(json.dumps([{"step": 0}]))
    for step, loss in ((200, 0.3), (400, 0.2)):
        (train / "evals" / f"step-{step:07d}.json").write_text(
            json.dumps(
                {
                    "step": step,
                    "eval_loss": loss,
                    "eval_loss_nowcast": loss - 0.1,
                    "eval_loss_forecast": 0.1,
                }
            )
        )
    run = runs.make_run("r", str(folder), live=True)
    rows = [{"step": 200, "loss": 0.4, "held": None}]
    got = training.loss_run(run, rows, [], lambda s: s / 4, None, "#111")
    assert got["held"] == [[50.0, 0.3], [100.0, 0.2]]
    assert got["held_parts"]["nowcast"] == [[50.0, 0.2], [100.0, 0.1]]
    assert got["held_parts"]["forecast"] == [[50.0, 0.1], [100.0, 0.1]]


def test_the_recent_rate_starts_at_the_trainer_s_resume(tmp_path):
    # Run A stopped 10:09 to 11:10: a window spanning it gave 0.10 steps/s
    # and an ETA of days. The rate is over the steps since the resume.
    before = [(0.0, 0.0), (600.0, 600.0)]
    after = [(4200.0, 600.0), (4300.0, 680.0), (4400.0, 760.0)]
    rate, span = training.recent_rate(
        before + after, window=4400.0, since=4200.0
    )
    assert rate == pytest.approx(0.8)
    assert span == pytest.approx(200.0)
    # One point since the resume: no rate yet, not the gap's.
    got = training.recent_rate([*before, (4200.0, 610.0)], since=4200.0)
    assert got == (None, None)
    # The resume is read from the newest segment log.
    train = tmp_path / "train"
    train.mkdir()
    (train / "n3200.log").write_text(
        "2026-11-23 13:09:17 [train] 90 held-out outbreaks, 3660 blocks\n"
        "2026-11-23 13:09:18 [train] resumed at step 41800, outbreaks 1801-\n"
    )
    (train / "n3200.log.pin-014a2bac5").write_text("")
    want = datetime(2026, 11, 23, 13, 9, 18).timestamp()
    assert training.resumed_at(tmp_path) == want


def test_only_checkpoint_folders_count_as_checkpoints(tmp_path):
    # n3200.log.pin-014a2bac5 is a file: its digits made a checkpoint of
    # 320,001,425 simulations.
    train = tmp_path / "train"
    for n in (100, 200):
        (train / f"n{n}").mkdir(parents=True)
        (train / f"n{n}" / "settings.json").write_text("{}")
    (train / "n3200.log.pin-014a2bac5").write_text("log")
    (train / "n3200.log").write_text("log")
    assert [c[0] for c in logs.checkpoints(tmp_path)] == ["n100", "n200"]
    assert logs.prefixes(tmp_path, [400]) == [100, 200, 400]
