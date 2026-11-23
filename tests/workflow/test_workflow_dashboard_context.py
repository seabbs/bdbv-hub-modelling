"""A dashboard is configured from workflow config, the machine and the CLI."""

import pytest

pytest.importorskip("matplotlib")

from flowlib.dashboard import context

CONFIG = {
    "finetune": {
        "outbreaks": 8400,
        "learning_rate": 1e-5,
        "prefixes": [250, 500, 8400],
    },
    "dashboard": {
        "runs": [
            {"key": "realdata", "source": "realdata-ft-", "colour": "#222"}
        ],
        "references": [
            {
                "key": "rw",
                "label": "randomwalknowcast",
                "colour": "#666",
                "model": "^randomwalknowcast$",
            }
        ],
        "rule": "R",
        "mail_filter": "slots|ckpt",
        "mirror": "output/dashboard/mirror",
        "datasets": [
            {
                "key": "germany",
                "label": "Germany",
                "truth": "t",
                "groups": [{"name": "national", "match": "^DE$"}],
            }
        ],
    },
}


def _workflow(tmp_path):
    """A config with two workflow runs and this machine's settings."""
    cfg = {**CONFIG, "runs": {"runa": {}, "noslots": {}}}
    cfg["dashboard"] = {
        k: v
        for k, v in CONFIG["dashboard"].items()
        if k not in ("runs", "mirror")
    }
    local = {"ROOT": str(tmp_path), "SCORE_CACHE": "/x", "MAILBOX": ""}
    return cfg, local


def test_runs_come_from_the_workflow_runs(tmp_path):
    cfg, local = _workflow(tmp_path)
    ctx = context.build(cfg, local)
    assert [r.key for r in ctx.runs] == ["runa", "noslots"]
    assert ctx.runs[0].folder == tmp_path / "runs" / "runa"
    assert next(r for r in ctx.runs if r.live).key == "runa"


def test_a_run_is_picked_by_name(tmp_path):
    cfg, local = _workflow(tmp_path)
    ctx = context.build(cfg, local, runs=["noslots"])
    assert [r.key for r in ctx.runs] == ["noslots"]
    assert ctx.runs[0].live


def test_an_unknown_run_name_is_an_error(tmp_path):
    cfg, local = _workflow(tmp_path)
    with pytest.raises(SystemExit, match="runa, noslots"):
        context.build(cfg, local, runs=["other"])


def test_no_local_file_changes_the_dashboard(tmp_path):
    cfg, local = _workflow(tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "dashboard.local.yaml").write_text(
        "rule: other\nruns: [{key: x, source: x-ft-}]\n"
    )
    ctx = context.build(
        {**cfg}, {**local, "DASHBOARD_CONFIG": ""}, root=tmp_path
    )
    assert ctx.rule == "R"
    assert [r.key for r in ctx.runs] == ["runa", "noslots"]


def test_runs_come_from_all_three_layers(tmp_path):
    local = {
        "DASHBOARD_RUNS": f"slots={tmp_path},live",
        "REMOTE": "",
        "SCORE_CACHE": str(tmp_path / "s"),
        "MAILBOX": str(tmp_path / "m"),
    }
    ctx = context.build(CONFIG, local, runs=["extra=x-ft-"], refs=[])
    assert [r.key for r in ctx.runs] == ["realdata", "slots", "extra"]
    assert next(r for r in ctx.runs if r.live).key == "slots"
    assert [r.key for r in ctx.references] == ["rw"]
    assert ctx.rule == "R"
    assert ctx.outbreaks == 8400
    assert ctx.prefixes == (250, 500, 8400)
    assert (
        ctx.source.scores == tmp_path / "s" / "tables" / "checkpoint-scores.csv"
    )
    assert ctx.mailbox == tmp_path / "m"


def test_a_remote_machine_reads_a_mirror(tmp_path):
    local = {
        "DASHBOARD_RUNS": "",
        "REMOTE": "h",
        "REMOTE_ROOT": "r",
        "SCORE_CACHE": "/x",
        "MAILBOX": "",
    }
    ctx = context.build(CONFIG, local, runs=[], refs=[], root=tmp_path)
    assert ctx.source.root == tmp_path / "output/dashboard/mirror"
    assert ctx.mailbox is None


def test_no_runs_at_all_is_an_error():
    cfg = {**CONFIG, "dashboard": {**CONFIG["dashboard"], "runs": []}}
    local = {
        "DASHBOARD_RUNS": "",
        "REMOTE": "",
        "SCORE_CACHE": "/x",
        "MAILBOX": "",
    }
    with pytest.raises(SystemExit, match="no runs"):
        context.build(cfg, local, runs=[], refs=[])


def test_a_reference_can_take_its_paper_colour(tmp_path):
    from flowlib.dashboard import paperstyle

    cfg = {
        **CONFIG,
        "dashboard": {
            **CONFIG["dashboard"],
            "references": [
                {
                    "key": "rw",
                    "label": "randomwalknowcast",
                    "model": "^randomwalknowcast$",
                    "paper_model": "randomwalknowcast",
                }
            ],
        },
    }
    local = {
        "DASHBOARD_RUNS": f"a={tmp_path}",
        "REMOTE": "",
        "SCORE_CACHE": "/x",
        "MAILBOX": "",
    }
    ctx = context.build(cfg, local)
    assert ctx.references[0].colour == paperstyle.model_colour(
        "randomwalknowcast"
    )


def test_runs_and_references_take_the_shared_display_names(tmp_path):
    from pathlib import Path

    import yaml
    from flowlib.dashboard import paperstyle

    cfg = {
        **CONFIG,
        "dashboard": {
            **CONFIG["dashboard"],
            "runs": [],
            "references": [
                {
                    "key": "rw",
                    "model": "^r$",
                    "paper_model": "randomwalknowcast",
                }
            ],
        },
    }
    local = {
        "DASHBOARD_RUNS": f"a={tmp_path}",
        "REMOTE": "",
        "SCORE_CACHE": "/x",
        "MAILBOX": "",
    }
    ctx = context.build(cfg, local)
    assert ctx.references[0].label == "Random walk nowcast"
    # A run with no label of its own is the fine-tuned model.
    assert ctx.runs[0].label == "Fine-tuned (delays)"
    assert ctx.runs[0].colour == paperstyle.model_colour(context.RUN_MODEL)
    # The workflow's references are named and coloured from the paper.
    shipped = yaml.safe_load(
        (Path(__file__).parents[2] / "workflow" / "config.yaml").read_text()
    )["dashboard"]["references"]
    assert all("paper_model" in r and "label" not in r for r in shipped)
    names = {paperstyle.model_label(r["paper_model"]) for r in shipped}
    assert names == {
        "Zero-shot (delays)",
        "Zero-shot (complete data only)",
        "Random walk nowcast",
    }


def test_runs_take_the_shared_name_and_colour_whatever_their_label(tmp_path):
    # The live run is plain "Fine-tuned (delays)", earlier
    # runs "Fine-tuned (delays) - <grid> grid" (else their key), never a
    # configured label such as "grid r6 (live)".
    import json

    from flowlib.dashboard import paperstyle, runs

    blue = paperstyle.model_colour(context.RUN_MODEL)
    one = context.named_runs(
        [runs.make_run("r6", "/x", label="grid r6 (live)", colour="#db2777")]
    )
    assert (one[0].label, one[0].colour) == ("Fine-tuned (delays)", blue)
    old = tmp_path / "old"
    (old / "train" / "n250").mkdir(parents=True)
    (old / "train" / "n250" / "settings.json").write_text(
        json.dumps({"grid": "r5", "n_outbreaks": 250})
    )
    several = context.named_runs(
        [
            runs.make_run("a", str(old), label="grid r5"),
            runs.make_run("slots", "slots-ft-"),
            runs.make_run("r6", "/y", label="grid r6 (live)", live=True),
        ]
    )
    assert [r.label for r in several] == [
        "Fine-tuned (delays) - r5 grid",
        "Fine-tuned (delays) - slots",
        "Fine-tuned (delays)",
    ]
    # The live run is drawn in the model's colour, the others in its
    # shades, so every run reads as the fine-tuned model.
    assert several[2].colour == blue
    assert {several[0].colour, several[1].colour} <= set(context.RUN_SHADES)
    assert several[0].colour != several[1].colour


def test_references_take_their_line_style_from_the_shared_map():
    # Zero-shot dotted, complete data dashed, as in the paper.
    from flowlib.dashboard import paperstyle

    assert paperstyle.model_dash("fm_chronos_daily_ragged") == "dotted"
    assert paperstyle.model_dash("fm_chronos_daily_complete") == "dashed"
    assert paperstyle.model_dash("fm_chronos_daily_ragged_ft") == "solid"


def test_the_workflow_s_runs_show_with_no_other_setting(tmp_path):
    # Each run in workflow/config.yaml's `runs` is shown from its folder,
    # <ROOT>/runs/<name>, with nothing on the command line.
    (tmp_path / "runs" / "runa").mkdir(parents=True)
    cfg = {
        **CONFIG,
        "runs": {"runa": {}},
        "dashboard": {**CONFIG["dashboard"], "runs": []},
    }
    local = {
        "ROOT": str(tmp_path),
        "DASHBOARD_RUNS": "",
        "REMOTE": "",
        "SCORE_CACHE": "/x",
        "MAILBOX": "",
    }
    ctx = context.build(cfg, local, runs=[], refs=[])
    assert [r.key for r in ctx.runs] == ["runa"]
    assert ctx.runs[0].folder == tmp_path / "runs" / "runa"
