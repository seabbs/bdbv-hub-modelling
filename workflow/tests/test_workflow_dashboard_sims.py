"""The simulated data dashboard: its index, one simulation, its pages."""

import gzip
import json
import os
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
from flowlib.dashboard import serve, sims

ROOT = Path(__file__).parents[2]
GRID = ROOT / "packages" / "nowcastfoundation" / "config" / "finetune"
GRID = GRID / "grid.yaml"

LAYERS = {
    "trajectory": "infections",
    "strata": "strata",
    "observation": "observation",
    "delay_family": "delays",
    "report_days": "reporting",
    "gt_mean": "infections",
    "quiet": "strata",
}


def _batch(folder: Path, seed: int, latent: bool = False) -> None:
    """Write a batch of two outbreaks as the simulator writes one.

    Outbreak 1: two strata, 10 cases on each of 14 days, half reported on
    the day and half a week later; its second draw is ignored. Outbreak 2:
    one stratum, 3 cases a day for 7 days, all on the day.
    """
    folder.mkdir(parents=True, exist_ok=True)
    days = [f"2026-01-{d:02d}" for d in range(1, 15)]
    rows = ["outbreak,timestep,location,reference_date,report_date,count"]
    for loc in ("P01-A1", "P01-A2"):
        for i, d in enumerate(days):
            later = f"2026-01-{i + 8:02d}" if i + 8 <= 31 else d
            for draw in (1, 2):
                rows.append(f"{seed}-0001-{draw},day,{loc},{d},{d},5")
                rows.append(f"{seed}-0001-{draw},day,{loc},{d},{later},5")
    for d in days[:7]:
        rows.append(f"{seed}-0002-1,day,P01,{d},{d},3")
    with gzip.open(folder / f"counts-{seed}.csv.gz", "wt") as f:
        f.write("\n".join(rows) + "\n")
    (folder / f"settings-{seed}.csv").write_text(
        "outbreak,trajectory,strata,observation,delay_family,report_days,"
        "gt_mean,quiet,place_lags,cases\n"
        f"{seed}-0001,multi_wave,places_ages,negbin1d,lognormal,daily,"
        "10.5,0.2,0;7,280\n"
        f"{seed}-0002,decline,single,poisson,exponential,weekly,"
        "12.0,0.0,0,21\n"
    )
    if latent:
        lines = ["day,infections,onsets,r,reff,date,location,outbreak"]
        for i, d in enumerate(days):
            lines.append(f"{i},20,15,1.1,1.0,{d},all,{seed}-0001")
            lines.append(f"{i},10,7,1.1,1.0,{d},P01-A1,{seed}-0001")
        with gzip.open(folder / f"latent-{seed}.csv.gz", "wt") as f:
            f.write("\n".join(lines) + "\n")


@pytest.fixture
def root(tmp_path):
    _batch(tmp_path / "rounds" / "r1" / "sims", 101, latent=True)
    _batch(tmp_path / "rounds" / "r1" / "sims", 102)
    _batch(tmp_path / "runs" / "runa" / "sims", 101)
    (tmp_path / "rounds" / "r2" / "sims").mkdir(parents=True)
    return tmp_path


def test_each_grid_setting_has_its_layer():
    got = sims.layers(GRID)
    for name, layer in LAYERS.items():
        assert got[name] == layer
    assert set(got.values()) == set(sims.LAYER_ORDER)


def test_the_sources_are_the_rounds_and_runs_with_simulations(root):
    got = sims.sources(root)
    assert list(got) == ["r1", "runa"]
    assert got["r1"] == root / "rounds" / "r1" / "sims"


def test_the_index_has_one_entry_per_simulation(root, tmp_path):
    index = sims.Index(root, tmp_path / "cache", LAYERS)
    got = index.data("r1")
    assert got["ids"] == ["101-0001", "101-0002", "102-0001", "102-0002"]
    assert got["batches"] == 2
    traj = got["factors"]["trajectory"]
    assert [traj["levels"][c] for c in traj["codes"]] == [
        "multi_wave",
        "decline",
        "multi_wave",
        "decline",
    ]
    assert got["layers"]["trajectory"] == "infections"
    assert got["values"]["gt_mean"] == [10.5, 12.0, 10.5, 12.0]
    # Lists of values per stratum or lag are not one value.
    assert "place_lags" not in got["values"]
    stats = got["stats"]
    # The first draw only, every stratum summed.
    assert stats["cases"][:2] == [280, 21]
    assert stats["strata"][:2] == [2, 1]
    assert stats["days"][:2] == [14, 7]
    assert stats["delay"][:2] == [3.5, 0.0]
    assert stats["within7"][:2] == [0.5, 1.0]
    assert got["series"][0] == [140, 140]
    assert got["series"][1] == [21]


def test_the_index_reads_a_batch_again_only_when_it_changes(root, tmp_path):
    cache = tmp_path / "cache"
    sims.Index(root, cache, LAYERS).data("r1")
    assert len(list((cache / "r1").glob("*.json"))) == 2
    counts = root / "rounds" / "r1" / "sims" / "counts-102.csv.gz"
    stat = counts.stat()
    counts.write_bytes(b"x" * stat.st_size)
    os.utime(counts, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    # The cached summary is used while the files are as they were.
    got = sims.Index(root, cache, LAYERS).data("r1")
    assert len(got["ids"]) == 4


def test_an_unreadable_batch_is_left_out(root, tmp_path):
    (root / "rounds" / "r1" / "sims" / "counts-102.csv.gz").write_bytes(b"x")
    got = sims.Index(root, tmp_path / "cache", LAYERS).data("r1")
    assert got["ids"] == ["101-0001", "101-0002"]


def test_an_unknown_source_has_no_index(root, tmp_path):
    assert sims.Index(root, tmp_path / "cache", LAYERS).data("r9") is None


def test_one_simulation_s_detail(root):
    got = sims.simulation(root / "rounds" / "r1" / "sims", "101-0001", LAYERS)
    assert got["id"] == "101-0001"
    assert len(got["dates"]) == 14
    assert set(got["strata"]) == {"P01-A1", "P01-A2"}
    assert got["strata"]["P01-A1"] == [10] * 14
    # Each cut-off's counts are those reported by then.
    first = got["known"][0]
    assert all(v <= 20 for v in first["counts"])
    assert got["delays"][0] == pytest.approx(0.5)
    assert got["delays"][7] == pytest.approx(0.5)
    by = {(s["layer"], s["name"]): s["value"] for s in got["settings"]}
    assert by[("infections", "trajectory")] == "multi_wave"
    assert by[("other", "cases")] == "280"
    assert got["latent"]["infections"] == [20.0] * 14
    assert got["triangle"]["counts"][0][0] == 70
    assert got["triangle"]["counts"][0][1] == 70


def test_a_simulation_without_latent_values_has_none(root):
    got = sims.simulation(root / "runs" / "runa" / "sims", "101-0002", LAYERS)
    assert got["latent"] is None
    assert got["strata"] == {"P01": [3] * 7}


def test_an_unknown_simulation_has_no_detail(root):
    folder = root / "rounds" / "r1" / "sims"
    assert sims.simulation(folder, "101-0009", LAYERS) is None
    assert sims.simulation(folder, "999-0001", LAYERS) is None
    assert sims.simulation(folder, "../x", LAYERS) is None


def test_the_page_reuses_the_dashboard_style():
    page = sims.page()
    assert "<title>Simulations</title>" in page
    assert "--c1:" in page
    # Relative URLs, so it works under the monitor's /sims/ and alone.
    assert 'fetch("/' not in page


@pytest.fixture
def server(root, tmp_path):
    sys.path.insert(0, str(ROOT / "workflow" / "scripts"))
    import dashboard

    index = sims.Index(root, tmp_path / "cache", LAYERS)
    index.refresh()
    srv = ThreadingHTTPServer(
        ("127.0.0.1", 0), dashboard.sims_handler(index, LAYERS)
    )
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def _get(url):
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as err:
        return err.code, ""


def test_the_server_answers_the_page_index_and_one_simulation(server):
    status, body = _get(server + "/sims/")
    assert status == 200
    assert "<title>Simulations</title>" in body
    status, body = _get(server + "/sims/sources.json")
    assert status == 200
    assert [s["key"] for s in json.loads(body)] == ["r1", "runa"]
    status, body = _get(server + "/sims/index/r1.json")
    assert status == 200
    assert len(json.loads(body)["ids"]) == 4
    status, body = _get(server + "/sims/sim/r1/101-0001.json")
    assert status == 200
    assert json.loads(body)["id"] == "101-0001"
    assert _get(server + "/sims/index/r9.json")[0] == 404
    assert _get(server + "/sims/sim/r1/101-0009.json")[0] == 404


def test_the_monitor_serves_the_simulations_under_sims():
    flask = pytest.importorskip("flask")
    import http.server

    from flowlib import monitor_server

    class Upstream(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = f"upstream {self.path}".encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    up = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    try:
        app = flask.Flask("test")
        cfg = {
            "DASHBOARD_HOST": "127.0.0.1",
            "DASHBOARD_PORT": str(up.server_port),
        }
        monitor_server.add_finetune(app, cfg)
        client = app.test_client()
        got = client.get("/sims/index/r1.json")
        assert got.get_data(as_text=True) == "upstream /sims/index/r1.json"
        assert client.get("/sims").status_code == 308
    finally:
        up.shutdown()
    assert 'href="/sims/"' in monitor_server.LINK


def test_a_simulating_run_serves_the_dashboard(monkeypatch):
    monkeypatch.delenv(serve.JOB_ENV, raising=False)
    on = {"dashboard": {"serve": True}}
    assert serve.wanted(on, ["simulate"])
    assert serve.wanted(on, ["push"])
    assert not serve.wanted(on, ["score"])


def test_each_setting_has_its_grid_note():
    got = sims.notes(GRID)
    # A drawn value's `why`, a factor's comment above it in the grid.
    assert got["quiet"].startswith("even outbreaks reach most places")
    assert "stops for good" in got["report_stop"]


def test_the_index_keeps_the_grid_s_order_and_notes(root, tmp_path):
    index = sims.Index(
        root, tmp_path / "cache", LAYERS, notes={"quiet": "why quiet"}
    )
    got = index.data("r1")
    assert got["order"] == list(LAYERS)
    assert got["notes"] == {"quiet": "why quiet"}


def test_the_newest_run_is_the_default_source(root, tmp_path):
    import os

    _batch(root / "runs" / "runb" / "sims", 101)
    for name, when in (("runa", 1000), ("runb", 2000)):
        for f in (root / "runs" / name / "sims").iterdir():
            os.utime(f, (when, when))
    listing = sims.Index(root, tmp_path / "cache", LAYERS).listing()
    assert [s["key"] for s in listing if s["default"]] == ["runb"]
    assert {s["key"]: s["kind"] for s in listing}["r1"] == "round"


def test_the_page_picks_one_layer_at_a_time():
    page = sims.page()
    assert "id=layerpick" in page
    assert "id=constants" in page


def test_each_simulation_s_derived_summaries(root, tmp_path):
    # Outbreak 1: half reported on the day, half a week later, 20 a day
    # every day; outbreak 2: all on the day.
    got = sims.Index(root, tmp_path / "cache", LAYERS).data("r1")
    cdf = dict(zip(sims.DELAY_POINTS, got["derived"]["cdf"][0], strict=True))
    assert cdf[0] == 0.5
    assert cdf[5] == 0.5
    assert cdf[7] == 1.0
    assert got["derived"]["cdf"][1][0] == 1.0
    assert got["derived"]["refdow"][0] == [1.0] * 7
    assert got["derived"]["repdow"][0] == [1.0] * 7
    stats = got["stats"]
    assert stats["within14"][0] == 1.0
    assert stats["first_report"][0] == 0
    assert stats["gap"][0] == 1
    assert stats["releases"][0] == 7.0
    assert stats["growth"][0] == 0.0
    assert stats["noise"][0] == 0.0


def test_a_batch_is_read_once_for_its_simulations(root, monkeypatch):
    calls = []
    real = sims._read_counts

    def counting(path):
        calls.append(path)
        return real(path)

    monkeypatch.setattr(sims, "_read_counts", counting)
    sims._batch_rows.cache_clear()
    folder = root / "rounds" / "r1" / "sims"
    assert sims.simulation(folder, "101-0001", LAYERS)
    assert sims.simulation(folder, "101-0002", LAYERS)
    assert len(calls) == 1


def test_the_page_splits_the_derived_summaries_by_the_group():
    page = sims.page()
    assert "id=derived" in page


def test_each_simulation_s_delay_structure_in_the_index(root, tmp_path):
    got = sims.Index(root, tmp_path / "cache", LAYERS).data("r1")
    for k in ("delay_time", "ref_delay", "release_days", "gaps"):
        assert len(got["derived"][k]) == 4
    # Outbreak 1 is reported every day: no wait for a release.
    assert got["stats"]["wait"][0] == 0
    assert got["stats"]["release_weekdays"][0] == 7
    # Two strata with the same delays.
    assert got["stats"]["strata_spread"][0] == 0
    assert got["stats"]["strata_spread"][1] is None
    assert got["time_bins"] == 10


def test_one_simulation_s_delay_structure(root):
    got = sims.simulation(root / "rounds" / "r1" / "sims", "101-0001", LAYERS)
    st = got["structure"]
    assert st["overall"]["mean"] == 3.5
    assert st["by_ref"]["week"][0] == "2026-01-01"
    names = [s["name"] for s in st["strata"]]
    assert names == ["all", "P01", "P01-A1", "P01-A2"]


def test_the_german_and_bdbv_delays_are_served_when_built(root, tmp_path):
    path = sims.reference_path(root)
    assert path == root / "rounds" / "report" / "delay-reference.json"
    index = sims.Index(root, tmp_path / "cache", LAYERS)
    assert index.reference() == {}
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"germany": {"label": "Germany"}}))
    assert index.reference()["germany"]["label"] == "Germany"


def test_the_server_answers_the_delay_reference(server):
    status, body = _get(server + "/sims/reference.json")
    assert status == 200
    assert json.loads(body) == {}


def test_the_page_has_the_delay_structure_section():
    page = sims.page()
    assert "id=delays" in page
    assert "reference.json" in page


DELAY_FACTORS = {
    "delay_change_strata": "strata",
    "delay_change_switch": "delays",
    "delay_change_combine": "delays",
    "delay_counts": "delays",
    "delay_strata_noise": "delays",
    "bursts": "reporting",
}
"""The grid's factors for the delay's structure by stratum and bursts."""


def test_the_delay_structure_factors_have_their_layers():
    got = sims.layers(GRID)
    for name, layer in DELAY_FACTORS.items():
        assert got[name] == layer


def test_the_delay_structure_factors_group_the_simulations(root, tmp_path):
    folder = root / "rounds" / "r1" / "sims"
    for seed in (101, 102):
        path = folder / f"settings-{seed}.csv"
        lines = path.read_text().splitlines()
        extra = list(DELAY_FACTORS)
        lines[0] += "," + ",".join(extra)
        lines[1] += ",related,diverge,combined,strata,both,yes"
        lines[2] += ",shared,none,single,none,none,none"
        path.write_text("\n".join(lines) + "\n")
    layer_of = LAYERS | DELAY_FACTORS
    got = sims.Index(root, tmp_path / "cache", layer_of).data("r1")
    for name, layer in DELAY_FACTORS.items():
        assert name in got["factors"]
        assert got["layers"][name] == layer
        # The page's "Colour and group by" lists the index's factors in
        # the grid's order.
        assert name in got["order"]
    bursts = got["factors"]["bursts"]
    assert [bursts["levels"][c] for c in bursts["codes"]] == [
        "yes",
        "none",
        "yes",
        "none",
    ]


def test_each_simulation_s_strata_delays_in_the_index(root, tmp_path):
    got = sims.Index(root, tmp_path / "cache", LAYERS).data("r1")
    stats = got["stats"]
    for k in (
        "strata_corr",
        "counts_slope",
        "burst_ratio",
        "withdrawn_share",
        "noise_time",
    ):
        assert k in stats
    # Nothing withdrawn; the first report is half the final count.
    assert stats["withdrawn_share"][0] == 0
    assert stats["burst_ratio"][0] == 0.5


def test_one_simulation_s_strata_delays_and_drawn_noise(root):
    got = sims.simulation(root / "rounds" / "r1" / "sims", "101-0001", LAYERS)
    by = got["structure"]["by_stratum"]
    assert [s["name"] for s in by["strata"]] == ["P01-A1", "P01-A2"]
    assert len(by["strata"][0]["mean"]) == len(by["week"])
    assert set(got["structure"]["noise"]) >= {"time_share", "strata_share"}
    assert got["noise_drawn"] == {
        "time": None,
        "strata": None,
        "strata_time": None,
    }


def test_the_page_draws_each_stratum_s_delays():
    page = sims.page()
    assert "strataWeeks" in page
    assert "First report over the final count" in page
    assert "noise_residual" in page


def _chromium() -> str | None:
    """Return a headless Chromium on this machine, None if there is none.

    `CHROMIUM` if set, else Playwright's cached headless shell.
    """
    import shutil

    if os.environ.get("CHROMIUM"):
        return os.environ["CHROMIUM"]
    cache = Path.home() / "Library" / "Caches" / "ms-playwright"
    if not cache.exists():
        cache = Path.home() / ".cache" / "ms-playwright"
    found = sorted(cache.glob("chromium_headless_shell-*/*/*shell"))
    return str(found[-1]) if found else shutil.which("chromium")


def test_the_page_renders_each_stratum_s_delays_in_chromium(server, root):
    import subprocess

    chrome = _chromium()
    if chrome is None:
        pytest.skip("no headless Chromium on this machine")
    import pandas as pd
    from flowlib import delay_structure as ds

    # The data to compare with, two strata over 12 weeks.
    days = pd.date_range("2026-01-05", periods=84)
    rows = pd.DataFrame(
        {
            "stratum": ["DE-BY", "DE-HB"] * 42,
            "ref": days,
            "rep": days + pd.Timedelta(days=2),
            "count": 20,
        }
    )
    st = ds.structure(rows)
    entry = {
        "short": "Germany",
        "label": "Germany",
        "structure": st,
        "scalars": ds.scalars(st),
        "curves": ds.curves(st),
    }
    sims.reference_path(root).parent.mkdir(parents=True)
    sims.reference_path(root).write_text(json.dumps({"germany": entry}))
    got = subprocess.run(
        [
            chrome,
            "--headless",
            "--disable-gpu",
            "--enable-logging=stderr",
            "--v=0",
            "--virtual-time-budget=15000",
            "--dump-dom",
            server + "/sims/#src=r1&sim=101-0001",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    import re

    # The page as drawn, not its script's text.
    dom = re.sub(r"<script>.*?</script>", "", got.stdout, flags=re.S)
    log = got.stderr
    assert "Uncaught" not in log, log
    assert "<svg" in dom
    # The delay section's new numbers and the deep dive's strata.
    for text in (
        "Largest weekly first report over final count",
        "Delay noise: share by stratum and week",
        "Mean delay by stratum and reference week",
        "Share reported by day 7, by stratum",
        "Mean delay against the log daily count, by week",
        "First report over the final count, by stratum",
        "Delay noise by part",
        "Strata's weekly mean delays correlate",
    ):
        assert text in dom, text
    # Each stratum a line in the deep dive's strata panels.
    assert dom.count("P01-A2") >= 4
