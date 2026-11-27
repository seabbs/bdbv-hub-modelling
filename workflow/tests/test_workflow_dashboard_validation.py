"""Validation forecasts: planned with fine-tuning, shown on the dashboard.

Each checkpoint's and base Chronos-2's forecasts of a sample of the
run's held-out (validation) blocks (`rule validation_forecasts`), at the
lowest priority on the checkpoint scoring device, which the loss section
sets against the truth.
"""

import gzip
import json
import sys
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
from flowlib.dashboard import runs, validation

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT / "workflow" / "scripts"))


def test_finetune_plans_each_checkpoint_s_validation_forecasts(tmp_path):
    pytest.importorskip("snakemake")
    from test_workflow_dryrun import FT, _job_priorities, dry_run

    out = dry_run(tmp_path, "finetune", config=FT, quiet=False)
    assert out.returncode == 0, out.stderr + out.stdout
    jobs = _job_priorities(out.stdout)
    rules = [r for r, _, _ in jobs]
    assert rules.count("validation_forecasts") == 2
    assert rules.count("validation_forecasts_base") == 1
    mine = [p for r, _, p in jobs if r.startswith("validation_forecasts")]
    from flowlib import compute

    others = [
        p
        for r, _, p in jobs
        if r in ("train_segment", "fetch_batch")
        or (r == "simulate_batch" and p >= compute.NEXT_BATCH_PRIORITY)
    ]
    # Below the trainer, its hand-offs and the batches it waits on.
    assert max(mine) < min(others)
    # While the run trains, on the CPU (the checkpoint scoring device).
    text = out.stdout
    start = text.index("rule validation_forecasts")
    assert "--device cpu" in text[start : start + 3000]


def _write(path, model, n, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "block": 4,
        "source": "daily",
        "cut": 100,
        "mix": "national",
        "nowcast": 2,
        "rows": [
            {
                "name": "DE",
                "level": "national",
                "history": [1.0, 2.0, None],
                "truth": [3.0, 4.0, None],
                "q": [[value + i] * 3 for i in range(7)],
            }
        ],
    }
    with gzip.open(path, "wt") as f:
        json.dump({"model": model, "n_outbreaks": n, "blocks": [rec]}, f)


def _run(tmp_path):
    train = tmp_path / "finetuned" / "train"
    _write(train / validation.BASE_FILE, "base", 0, 10)
    _write(train / "n100" / validation.FILE, "n100", 100, 20)
    (train / "n100" / "settings.json").write_text('{"n_outbreaks": 100}')
    return runs.make_run("finetuned", str(tmp_path / "finetuned"), live=True)


def test_a_run_s_validation_index_lists_its_blocks_and_models(tmp_path):
    got = validation.index(_run(tmp_path))
    assert [m["key"] for m in got["models"]] == ["base", "n100"]
    assert got["models"][0]["label"] == "Zero-shot"
    assert got["models"][1]["n"] == 100
    assert got["blocks"] == [
        {
            "block": 4,
            "source": "daily",
            "cut": 100,
            "mix": "national",
            "rows": ["DE"],
        }
    ]


def test_one_block_has_each_model_s_quantiles_beside_the_truth(tmp_path):
    got = validation.block(_run(tmp_path), 4)
    assert got["rows"][0]["truth"] == [3.0, 4.0, None]
    assert got["q"]["base"][0][3] == [13.0] * 3
    assert got["q"]["n100"][0][3] == [23.0] * 3
    assert validation.block(_run(tmp_path), 9) is None


def test_a_run_with_none_has_an_empty_index(tmp_path):
    run = runs.make_run("nobiases", str(tmp_path / "nobiases"), live=False)
    assert validation.index(run) == {"models": [], "blocks": []}


def test_the_server_answers_a_run_s_index_and_blocks(tmp_path):
    import dashboard

    run = _run(tmp_path)
    srv = ThreadingHTTPServer(
        ("127.0.0.1", 0), dashboard.validation_handler([run])
    )
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        with urllib.request.urlopen(
            f"{base}/validation/finetuned/index.json"
        ) as r:
            assert json.loads(r.read())["blocks"][0]["block"] == 4
        with urllib.request.urlopen(f"{base}/validation/finetuned/4.json") as r:
            assert "n100" in json.loads(r.read())["q"]
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(f"{base}/validation/nope/4.json")
    finally:
        srv.shutdown()


def test_the_loss_section_has_the_validation_forecast_card(tmp_path):
    from flowlib.dashboard import page
    from test_workflow_dashboard_page import _machine

    out = page.sections(_machine(tmp_path))["loss"]
    assert "id=valcard" in out
    js = (ROOT / "workflow/flowlib/dashboard/page/validation.js").read_text()
    assert 'fetch("/' not in js
    assert "validation/" in js
