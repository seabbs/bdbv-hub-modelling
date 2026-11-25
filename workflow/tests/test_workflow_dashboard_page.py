"""The dashboard page, built from a fixture machine."""

import csv
import gzip
import json
import os
import re
import shutil
import time
from datetime import datetime, timedelta

import pytest

pytest.importorskip("matplotlib")

from flowlib.dashboard import page, runs, sources

LINE = "2026-11-16 16:{m:02d}:00 [train] step {s}: loss {l}, held-out (x) {h}"


def _gdata(section):
    """Return the nowcasts section's page data (its `gdata` JSON)."""
    return json.loads(section.split("id=gdata>", 1)[1].split("</script>", 1)[0])


def _machine(tmp_path):
    run = tmp_path / "run"
    (run / "train" / "n250" / "finetuned-ckpt").mkdir(parents=True)
    (run / "train" / "n250" / "settings.json").write_text(
        json.dumps(
            {
                "hash": "aaaaaa",
                "n_outbreaks": 250,
                "steps": 1140,
                "held_out_loss": 0.41,
                "held_out_loss_start": 0.47,
                "held_out_loss_by_source": {"weekly": 0.2, "daily": 0.5},
            }
        )
    )
    (run / "train" / "history.json").write_text(
        json.dumps(
            [{"step": 0, "held_out_loss": 0.47, "by_source": {"weekly": 0.25}}]
        )
    )
    (run / "train" / "n250.log").write_text(
        "\n".join(
            LINE.format(m=i, s=10 * i, l=0.5 - i / 100, h=0.45)
            for i in range(30)
        )
        + "\n"
    )
    # A running trainer's status: where the run is.
    (run / "train" / "stream-status.json").write_text(
        json.dumps(
            {
                "stage": "training",
                "step": 290,
                "outbreaks": 260,
                "time": "2026-11-16 16:29:00",
            }
        )
    )
    scores_dir = tmp_path / "scores" / "tables"
    scores_dir.mkdir(parents=True)
    with (scores_dir / "checkpoint-scores.csv").open("w") as f:
        w = csv.DictWriter(
            f,
            [
                "rule",
                "study",
                "level",
                "model",
                "checkpoint",
                "outbreaks",
                "rel_wis",
                "rel_crps",
                "loss",
                "loss_zero_shot",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "rule": "R",
                "study": "germany",
                "level": "national",
                "model": "foundation",
                "checkpoint": "slots-aaaaaa",
                "outbreaks": 250,
                "rel_wis": 0.9,
            }
        )
        w.writerow(
            {
                "rule": "R",
                "study": "held_out",
                "level": "all",
                "model": "",
                "checkpoint": "slots-aaaaaa",
                "outbreaks": 250,
                "loss": 0.41,
                "loss_zero_shot": 0.47,
            }
        )
    ger = tmp_path / "scores" / "germany" / "slots-aaaaaa"
    ger.mkdir(parents=True)
    with gzip.open(ger / "2022-02-09.csv.gz", "wt") as f:
        w = csv.DictWriter(
            f,
            [
                "model",
                "location",
                "target_end_date",
                "quantile",
                "value",
                "truth",
                "truth_vintage",
            ],
        )
        w.writeheader()
        for i, q in enumerate(
            ("0.025", "0.1", "0.25", "0.5", "0.75", "0.9", "0.975")
        ):
            w.writerow(
                {
                    "model": "foundation",
                    "location": "DE",
                    "target_end_date": "2022-02-01",
                    "quantile": q,
                    "value": 8 + i,
                    "truth": 10,
                    "truth_vintage": "2022-03-01",
                }
            )
    box = tmp_path / "box.md"
    box.write_text("2026-11-16 16:00 x: slots n250 published\nother\n")
    rs = [
        runs.make_run("mine", run, label="my run", colour="#123456", live=True)
    ]
    refs = runs.load_references(
        [
            {
                "key": "rw",
                "label": "randomwalknowcast",
                "colour": "#666",
                "model": "^randomwalknowcast$",
            }
        ],
        [],
    )
    return page.Context(
        runs=rs,
        references=refs,
        source=sources.Sources({"SCORE_CACHE": str(tmp_path / "scores")}),
        rule="R",
        outbreaks=8000,
        learning_rate=1e-5,
        prefixes=(250, 500, 750, 1000, 2000, 4000, 8000),
        mailbox=box,
        mail_filter="slots",
        once=True,
        datasets=[
            {
                "key": "germany",
                "label": "Germany",
                "format": "quantiles",
                "metric": "WIS",
                "truth": "t",
                "groups": [{"name": "national", "match": "^DE$"}],
            }
        ],
        view={
            "studies": {"germany": "Germany"},
            "levels": ["national"],
            "defaults": {"target": "7day", "source": "daily"},
        },
    )


def test_every_section_is_built_for_the_configured_runs(tmp_path):
    out = page.build(_machine(tmp_path))
    for h in (
        "Status",
        "Loss",
        "Attention bias",
        "LoRA update",
        "Nowcasts",
        "Recent activity",
        "Tables",
    ):
        assert f"<h2>{h}</h2>" in out
    assert "<b>my run</b>" in out
    for anchor in (
        "status",
        "loss",
        "bias",
        "lora",
        "nowcasts",
        "mailbox",
        "tables",
    ):
        assert f'<a href="#{anchor}">' in out
    assert "#123456" in out
    assert (
        "n250 of 8000" not in out
    )  # the live step is 290, n is steps / 4.5625
    assert re.search(r"Simulations trained</span> <b>\d+/8000</b>", out)
    assert "slots n250 published" in out
    assert "other" not in out.split("Recent activity")[1]
    assert "id=gdata" in out
    assert '"key":"germany"' in out
    assert "id=ldata" in out
    assert "id=bdata" in out
    assert "id=seldata" in out
    assert '<option value="daily" selected>daily counts</option>' in out
    # No hard-coded runs.
    assert "no slots" not in out


def test_the_page_never_reloads_and_a_served_one_updates_in_place(tmp_path):
    ctx = _machine(tmp_path)
    assert "location.reload" not in page.build(ctx)
    assert "/sections.json" not in page.build(ctx)
    ctx.once = False
    built = page.build(ctx)
    assert "location.reload" not in built
    assert "/sections.json" in built


def test_sections_are_the_page_s_parts(tmp_path):
    ctx = _machine(tmp_path)
    got = page.sections(ctx)
    assert {"bar", "status", "loss", "bias", "nowcasts", "tables"} <= set(got)
    built = page.build(ctx)
    for key in ("loss", "bias", "tables"):
        assert got[key] in built


def test_checkpoint_sizes_also_come_from_the_score_table(tmp_path):
    ctx = _machine(tmp_path)
    # A checkpoint with no saved settings, sized in the score table.
    sizes = page.checkpoint_sizes(
        ctx,
        {r.key: {"ck": page._checkpoints(r)} for r in ctx.runs},
        [{"checkpoint": "other-ft-bbbbbb", "outbreaks": "500"}],
    )
    assert sizes["bbbbbb"] == 500
    assert sizes["aaaaaa"] == 250


def test_the_mailbox_section_is_hidden_without_a_mailbox(tmp_path):
    ctx = _machine(tmp_path)
    ctx.mailbox = None
    assert "Recent activity" not in page.build(ctx)


def test_checkpoint_tables_have_a_column_per_source(tmp_path):
    ctx = _machine(tmp_path)
    out = page.build(ctx)
    tables = out.split("<h2>Tables</h2>")[1]
    assert "<th>weekly<th>daily" in tables or "<th>daily<th>weekly" in tables
    assert "<td>0.2000" in tables


def test_run_sizes_drop_prefixes_past_the_run_and_end_at_it():
    assert page.run_sizes((250, 8000, 9000), 8400) == [250, 8000, 8400]
    assert page.run_sizes((250, 500), 500) == [250, 500]


def test_the_run_size_comes_from_the_config_without_a_trainer(tmp_path):
    ctx = _machine(tmp_path)
    ctx.outbreaks = 8400
    ctx.prefixes = (250, 500, 1000, 2000, 4000, 8000)
    out = page.build(ctx)
    assert re.search(r"Simulations trained</span> <b>\d+/8400</b>", out)
    assert "of 8000" not in out
    planned = page.run_sizes(ctx.prefixes, ctx.outbreaks)
    st = page.live_status(ctx.runs[0], ctx.log(ctx.runs[0]), planned)
    assert st.params["prefixes"][-2:] == [8000, 8400]


def test_wis_has_no_section_of_its_own(tmp_path):
    # Each model's WIS is in the nowcasts' score card only.
    ctx = _machine(tmp_path)
    table = tmp_path / "scores" / "tables" / "checkpoint-scores.csv"
    with table.open() as f:
        rows = list(csv.DictReader(f))
    parts = ("wis", "dispersion", "underprediction", "overprediction")
    rows[0] |= dict(zip(parts, ("1.0", "0.5", "0.2", "0.3"), strict=True))
    with table.open("w") as f:
        w = csv.DictWriter(f, [*rows[0]])
        w.writeheader()
        w.writerows(rows)
    out = page.build(ctx)
    assert "<h2>Mean WIS</h2>" not in out
    assert '<a href="#wis">' not in out
    assert "wis" not in page.sections(ctx)


def test_a_segment_s_empty_folder_is_not_a_checkpoint_being_scored(tmp_path):
    ctx = _machine(tmp_path)
    folder = ctx.runs[0].folder
    # Snakemake makes the next segment's output folders when it starts.
    (folder / "train" / "n500" / "finetuned-ckpt").mkdir(parents=True)
    out = page.build(ctx)
    assert "<span class=kvk>Scoring</span>" not in out
    bar = out.split("id=livebar>", 1)[1].split("</span></span>", 1)[0]
    # Simulations done of the next prefix, and when it is due.
    assert re.search(r"n\d+ / n500", bar)
    assert "ETA" in bar


def test_a_written_checkpoint_is_scored_as_itself_not_as_the_next(tmp_path):
    ctx = _machine(tmp_path)
    ckpt = ctx.runs[0].folder / "train" / "n500" / "finetuned-ckpt"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_model.safetensors").write_bytes(b"")
    out = page.build(ctx)
    assert re.search(
        r"Scoring</span> <b>n500</b>.*?Since</span> <b>\d\d:\d\d</b>", out
    )
    assert re.search(r"n\d+ / n750", out)


def test_the_bar_s_clock_is_spaced_from_the_status(tmp_path):
    bar = page.sections(_machine(tmp_path))["bar"]
    assert "</span> <span class=mut>" in bar


def test_the_learning_rate_falls_over_simulations_not_constant(tmp_path):
    ctx = _machine(tmp_path)
    folder = ctx.runs[0].folder
    s = json.loads((folder / "train/n250/settings.json").read_text())
    s |= {
        "learning_rate": 1e-5,
        "lr_decay": "linear to zero over the outbreaks",
    }
    (folder / "train/n250/settings.json").write_text(json.dumps(s))
    ctx.outbreaks = 1000
    ctx.prefixes = (250, 500, 1000)
    out = page.build(ctx)
    assert "linear decay to 0 at 1000 simulations" in out
    assert ">constant<" not in out
    assert "Learning rate vs simulations trained" in out


def test_the_last_log_line_is_the_running_segment_s(tmp_path):
    ctx = _machine(tmp_path)
    folder = ctx.runs[0].folder
    now = datetime.now().replace(microsecond=0)
    (folder / "train" / "n500.log").write_text(
        f"{now:%Y-%m-%d %H:%M:%S} [train] resumed at step 1140, "
        "outbreaks 251-\n"
    )
    out = page.build(ctx)
    tile = out.split("Last log line", 1)[1].split("</div></div>", 1)[0]
    assert f"{now:%H:%M:%S}" in tile
    assert "STALE" not in tile
    assert "resumed at step 1140" in tile


def test_the_next_checkpoint_names_its_simulations_and_eta(tmp_path):
    ctx = _machine(tmp_path)
    out = page.build(ctx)
    tile = out.split("Next checkpoint", 1)[1].split("Segments", 1)[0]
    assert ">n500<" in tile
    # The status's 260 simulations of n500's 500, and no planned step.
    assert "260/500</b>" in tile
    assert "Step</span>" not in tile
    assert "ETA</span> <b>n/a</b>" not in tile


def test_every_configured_dataset_is_in_the_selector(tmp_path):
    # Every configured dataset, each listed whether or not it
    # has nowcasts yet.
    ctx = _machine(tmp_path)
    ctx.datasets = [
        *ctx.datasets,
        {"key": "other", "label": "Other"},
    ]
    sel = page.sections(ctx)["nowcasts"]
    data = _gdata(sel)
    got = [
        (d["key"], d["label"], bool(d.get("empty"))) for d in data["datasets"]
    ]
    assert ("other", "Other", True) in got


def test_speed_and_etas_come_from_the_status_file_when_steps_are_sparse(
    tmp_path,
):
    # One logged step in the last 30 minutes (evaluations every 400 steps)
    # and the trainer's status: a rate, simulations a minute and ETAs.
    ctx = _machine(tmp_path)
    folder = ctx.runs[0].folder
    now = datetime.now().replace(microsecond=0)
    (folder / "train" / "n250.log").write_text(
        LINE.format(m=0, s=800, l=0.5, h=0.45).replace(
            "2026-11-16 16:00:00",
            f"{now - timedelta(minutes=40):%Y-%m-%d %H:%M:%S}",
        )
        + "\n"
    )
    (folder / "train" / "stream-status.json").write_text(
        json.dumps(
            {
                "stage": "training",
                "step": 1200,
                "outbreaks": 250,
                "time": f"{now - timedelta(minutes=1):%Y-%m-%d %H:%M:%S}",
            }
        )
    )
    out = page.build(ctx)
    tile = out.split("Steps / s", 1)[1].split("</div></div>", 1)[0]
    assert "n/a" not in tile
    eta = out.split("ETA n8000", 1)[1].split("</div></div>", 1)[0]
    assert "n/a" not in eta


def test_the_loss_panel_offers_each_loss_type(tmp_path):
    out = page.build(_machine(tmp_path))
    for k in ("total", "nowcast", "forecast", "layout", "level"):
        assert f"data-lt={k}>" in out


def test_checkpoint_tables_have_no_real_data_validation_column(tmp_path):
    # Its loss is on BDBV pseudo-origins, held out.
    out = page.build(_machine(tmp_path))
    assert "real-val" not in out


def test_wis_parts_are_told_apart_by_fill_and_stack_in_one_order(tmp_path):
    # Dispersion solid, over- and underprediction hatched the
    # two ways, stacked underprediction, dispersion, overprediction.
    sel = page.sections(_machine(tmp_path))["nowcasts"]
    data = _gdata(sel)
    fills = dict(data["parts"])
    assert fills == {
        "dispersion": "solid",
        "overprediction": "hatch-up",
        "underprediction": "hatch-down",
    }
    assert data["stack"] == ["underprediction", "dispersion", "overprediction"]


def test_checkpoint_tables_split_the_held_out_loss_by_horizon(tmp_path):
    ctx = _machine(tmp_path)
    s = ctx.runs[0].folder / "train" / "n250" / "settings.json"
    settings = json.loads(s.read_text())
    settings |= {"held_out_loss_nowcast": 0.15, "held_out_loss_forecast": 0.26}
    s.write_text(json.dumps(settings))
    out = page.build(ctx)
    assert "<th>nowcast part<th>forecast part" in out
    assert "<td>0.4100<td>0.1500<td>0.2600" in out


def test_the_score_card_has_the_horizon_parts(tmp_path):
    from nowcasteval import labels

    data = _gdata(page.sections(_machine(tmp_path))["nowcasts"])
    assert [h[0] for h in data["horizon"]] == list(labels.HORIZON_PARTS)
    assert {h[0]: h[2] for h in data["horizon"]} == labels.HORIZON_FILLS


OFFLOADED = (
    "2026-11-16 16:{m:02d}:00 [train] step {s}: loss 0.5, held-out "
    "(monitor, by_source) left to the evaluator, 10 s"
)


def _status(folder, step, outbreaks, stage="training"):
    (folder / "train" / "stream-status.json").write_text(
        json.dumps(
            {
                "stage": stage,
                "step": step,
                "outbreaks": outbreaks,
                "time": "2026-11-16 16:29:00",
            }
        )
    )


def test_an_offloaded_run_shows_its_speed_steps_and_segment_etas(tmp_path):
    ctx = _machine(tmp_path)
    folder = ctx.runs[0].folder
    # An offloaded run logs steps with no held-out loss.
    (folder / "train" / "n250.log").write_text(
        "\n".join(OFFLOADED.format(m=i, s=60 * i) for i in range(30)) + "\n"
    )
    _status(folder, step=1740, outbreaks=300)
    ctx.outbreaks = 1000
    ctx.prefixes = (250, 500, 1000)
    planned = page.run_sizes(ctx.prefixes, ctx.outbreaks)
    st = page.live_status(ctx.runs[0], ctx.log(ctx.runs[0]), planned)
    assert st.rate == pytest.approx(1.0)
    assert st.mean_rate == pytest.approx(1.0)
    tiles, _ = page.status_tiles(st, ctx.runs[0])
    speed = tiles.split("Steps / s")[1].split("Next checkpoint")[0]
    assert "n/a" not in speed
    assert "Mean</span> <b>1</b>" in tiles
    # Each segment still to come, with its ETA.
    for n in (500, 1000):
        assert f"n{n}</span> <b>{st.eta(n)}</b>" in tiles


def test_a_stopped_run_names_the_batches_its_next_segment_waits_on(tmp_path):
    ctx = _machine(tmp_path)
    folder = ctx.runs[0].folder
    (folder / "train" / "n250.log").write_text(
        "\n".join(OFFLOADED.format(m=i, s=60 * i) for i in range(30)) + "\n"
    )
    status = folder / "train" / "stream-status.json"
    status.parent.mkdir(parents=True, exist_ok=True)
    status.write_text(json.dumps({"stage": "stopped at n250"}))
    old = time.time() - 3600
    os.utime(status, (old, old))
    (folder / "seeds.txt").write_text("".join(f"{s}\n" for s in range(50)))
    (folder / "sims").mkdir(exist_ok=True)
    for s in (0, 1, 2, 4, 5, 6, 7):
        (folder / "sims" / f"settings-{s}.csv").write_text(
            "outbreak\n" + "".join(f"{s}-{i}\n" for i in range(50))
        )
    ctx.outbreaks = 1000
    ctx.prefixes = (250, 500, 1000)
    planned = page.run_sizes(ctx.prefixes, ctx.outbreaks)
    st = page.live_status(ctx.runs[0], ctx.log(ctx.runs[0]), planned)
    tiles, _ = page.status_tiles(st, ctx.runs[0])
    trainer = tiles.split("Trainer")[1].split("Last log line")[0]
    assert "Waiting for batches" in trainer
    assert all(f"<b>{s}</b>" in trainer for s in (3, 8, 9, 10))


def test_the_trainer_status_is_folded_and_the_loss_open(tmp_path):
    ctx = _machine(tmp_path)
    page_html = page.build(ctx)
    status = page.sections(ctx)["status"]
    fold = re.search(r"<details class=about>(.*?)</details>", status, re.S)
    assert fold
    assert "Trainer status" in fold[1]
    assert "<details class=sec id=loss open>" in page_html


def test_checkpoint_tables_show_the_7_day_check(tmp_path):
    # Lead 00:55: the 7-day row against its daily rows' sums (median
    # ratio, sums inside its 90% interval) and its roughness over the
    # truth's (1 is as smooth as the truth), from the checkpoint's
    # seven-day.json (rule seven_day_check).
    ctx = _machine(tmp_path)
    where = ctx.runs[0].folder / "train" / "n250"
    (where / "seven-day.json").write_text(
        json.dumps(
            {"seven_day": {"ratio": 1.02, "inside90": 0.91, "rough_ratio": 1.5}}
        )
    )
    out = page.build(ctx)
    assert (
        "<th>7-day / daily sums<th>sums in 7-day 90%<th>7-day roughness / truth"
        in out
    )
    assert "<td>1.02<td>91%<td>1.50" in out


def test_the_trainer_s_losses_are_labelled_validation(tmp_path):
    # The trainer scores the grid's validation round; the held-out (test)
    # set is the paper's alone (lead 15:20), so no panel says held-out.
    out = page.build(_machine(tmp_path))
    text = re.sub(r"<script.*?</script>", "", out, flags=re.S)
    assert "Validation loss" in out
    assert "held-out loss" not in text.lower()


def test_a_just_restarted_run_shows_its_position_from_the_status(tmp_path):
    # Lead 00:20: after a restart the new segment log has no step lines,
    # and the page showed an old evaluation's step (37,200, 1592/3200)
    # with an ETA from it. Step and simulations come only from the
    # trainer's status; with no step since the restart there is no ETA.
    ctx = _machine(tmp_path)
    folder = ctx.runs[0].folder
    (folder / "train" / "stream-status.json").write_text(
        json.dumps(
            {
                "stage": "cutting held-out blocks of 90 outbreaks",
                "step": 51250,
                "outbreaks": 2200,
                "segment": "n3200",
                "target": 3200,
                "time": "2026-11-23 18:03:21",
            }
        )
    )
    (folder / "train" / "n250.log").write_text(
        "2026-11-23 18:02:50 [train] resumed at step 51250\n"
    )
    ctx.outbreaks = 3200
    ctx.prefixes = (250, 1600, 3200)
    planned = page.run_sizes(ctx.prefixes, ctx.outbreaks)
    st = page.live_status(ctx.runs[0], ctx.log(ctx.runs[0]), planned)
    assert st.step == 51250
    assert st.trained == 2200
    assert st.eta(3200) == "n/a"
    tiles, _ = page.status_tiles(st, ctx.runs[0])
    assert "Simulations trained</span> <b>2200/3200</b>" in tiles


def test_the_etas_come_from_the_status_and_the_rate_alone(tmp_path):
    # Lead 01:05 sweep (14): no step or simulations interpolated from the
    # checkpoints and logged chunks. The status gives the step and the
    # simulations (so its steps a simulation), the log the rate.
    ctx = _machine(tmp_path)
    folder = ctx.runs[0].folder
    (folder / "train" / "n250.log").write_text(
        "\n".join(OFFLOADED.format(m=i, s=60 * i) for i in range(30)) + "\n"
    )
    _status(folder, step=1740, outbreaks=300)
    ctx.outbreaks = 1000
    ctx.prefixes = (250, 500, 1000)
    planned = page.run_sizes(ctx.prefixes, ctx.outbreaks)
    st = page.live_status(ctx.runs[0], ctx.log(ctx.runs[0]), planned)
    assert (st.step, st.trained) == (1740, 300)
    assert st.eta(500) == page._dur((500 - 300) * 1740 / 300 / st.rate)
    tiles, _ = page.status_tiles(st, ctx.runs[0])
    assert "Final step" not in tiles
    assert "estimated" not in tiles


def test_without_a_status_the_position_is_not_known(tmp_path):
    ctx = _machine(tmp_path)
    (ctx.runs[0].folder / "train" / "stream-status.json").unlink()
    planned = page.run_sizes(ctx.prefixes, ctx.outbreaks)
    st = page.live_status(ctx.runs[0], ctx.log(ctx.runs[0]), planned)
    assert (st.step, st.trained) == (None, None)
    assert st.eta(max(planned)) == "n/a"


def _status_section(out):
    return out.split("<h2>Status</h2>", 1)[1].split("<h2>", 1)[0]


def test_every_run_training_at_once_has_its_status(tmp_path):
    # Runs train at once, a GPU each: the status names each one training,
    # not the live run's alone.
    ctx = _machine(tmp_path)
    alone = _status_section(page.build(ctx)).count("<div class=tiles>")
    live = ctx.runs[0].folder
    for name in ("other", "idle"):
        shutil.copytree(live, tmp_path / name)
    _status(tmp_path / "other", step=120, outbreaks=110)
    idle = tmp_path / "idle" / "train" / "stream-status.json"
    old = time.time() - 3600
    os.utime(idle, (old, old))
    ctx.runs += [
        runs.make_run("other", tmp_path / "other", label="other run"),
        runs.make_run("idle", tmp_path / "idle", label="idle run"),
    ]
    status = _status_section(page.build(ctx))
    assert "Also training" in status
    assert "<b>other run</b>" in status
    assert "idle run" not in status
    assert status.count("<div class=tiles>") == 2 * alone


def test_one_run_training_shows_its_status_alone(tmp_path):
    status = _status_section(page.build(_machine(tmp_path)))
    assert "Also training" not in status
