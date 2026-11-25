from __future__ import annotations

from pathlib import Path

from bdbvhub.reporting.store_note import store_note, write_store_note


def _row(origin: str, used: str | None, current: bool, failed: bool = False):
    return {
        "model": "renewal",
        "origin": origin,
        "key": "k",
        "used": used,
        "commit": "a" * 40 if used else None,
        "current": current,
        "failed": failed,
    }


def test_current_fits_name_their_commit() -> None:
    note = store_note([_row("2026-10-24", "k", True)], "b" * 40)
    assert "aaaaaaa" in note
    assert "!!!" not in note


def test_fits_not_made_are_listed() -> None:
    status = [
        _row("2026-11-01", None, False),
        _row("2026-10-24", "old", False),
        _row("2026-10-17", "old", False, failed=True),
    ]
    note = store_note(status, "b" * 40)
    assert '!!! note "Fits not made for this code"' in note
    assert "bbbbbbb" in note
    assert "`renewal` at 2026-10-24 and 2026-11-01." in note
    assert "failed for `renewal` at 2026-10-17." in note
    assert "older code are shown for `renewal` at 2026-10-17 and" in note
    assert "Not shown: `renewal` at 2026-11-01." in note
    assert "run" not in note


def test_no_record_says_no_fits(tmp_path: Path) -> None:
    write_store_note(tmp_path, tmp_path / "gen")
    text = (tmp_path / "gen" / "fit-store.md").read_text()
    assert "No fitted model has a stored fit yet." in text


def test_the_site_results_are_made_on_main_with_no_preview_head() -> None:
    # Fit store's site-results job writes the note from main's store, so
    # a preview shows main's results and names no pull request head.
    import yaml

    workflows = Path(__file__).parents[3] / ".github" / "workflows"
    job = yaml.safe_load((workflows / "fit-store.yml").read_text())["jobs"][
        "site-results"
    ]
    runs = "\n".join(s.get("run", "") for s in job["steps"])
    assert "bdbvhub report" in runs
    for step in job["steps"]:
        assert "PREVIEW_HEAD" not in (step.get("env") or {})
