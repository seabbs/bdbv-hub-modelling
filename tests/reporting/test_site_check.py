"""`germany-site-check`: the built German page names no held-out date."""

from pathlib import Path

from nowcastgermany.evaluation import load_germany_config
from nowcastgermany.reporting import check_site as main

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "germany.toml"
SPLIT = load_germany_config(CONFIG).split_date


def _site(tmp_path: Path, date: str) -> Path:
    site = tmp_path / "_site"
    (site / "germany").mkdir(parents=True)
    (site / "figures").mkdir()
    (site / "germany" / "index.html").write_text(f"<p>Up to {date}.</p>")
    for name in ("scoresummary-germany-latest", "explorer-germany"):
        (site / "figures" / f"{name}.json").write_text("{}")
    return site


def test_a_page_up_to_the_split_date_passes(tmp_path: Path) -> None:
    site = _site(tmp_path, SPLIT)
    assert main([str(site), "--config", str(CONFIG)]) == 0


def test_a_held_out_date_fails(tmp_path: Path, capsys) -> None:
    site = _site(tmp_path, "2099-01-01")
    assert main([str(site), "--config", str(CONFIG)]) == 1
    assert "2099-01-01" in capsys.readouterr().out


def test_an_unbuilt_site_fails(tmp_path: Path) -> None:
    assert main([str(tmp_path), "--config", str(CONFIG)]) == 1
