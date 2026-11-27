"""Each key endpoint's stable release names its current version.

`<endpoint>-main` holds `main.json`, which names the version's own
release. Promotion runs only from a commit on a configured branch, and
only for a version that is released. Readers take the record and never
guess a newest release. No test reaches GitHub: `gh` and `git` are a
stand-in (`FakeReleases`, the root conftest).
"""

from __future__ import annotations

import json

import pytest

from nowcastutils import pointers

REPO = "owner/repo"


def _promote(gh, target="thing-abc", **kw):  # type: ignore[no-untyped-def]
    args = {
        "hash": "abc123",
        "produced_by": "rule make_thing",
        "branches": ("main", "next"),
        "repo": REPO,
        "title": "Thing, main",
        "run": gh,
    }
    return pointers.promote("thing", target, **{**args, **kw})


def test_the_pointer_tag_is_the_endpoint_s_main() -> None:
    assert pointers.tag("chronos") == "chronos-main"
    assert pointers.tag("paper-germany") == "paper-germany-main"


def test_promoting_writes_the_record_to_the_stable_release(
    fake_releases,
) -> None:
    gh = fake_releases({"thing-abc": {"data.tar.gz": ""}})
    record = _promote(gh, needs="data.tar.gz", extra={"run": "runa"})
    assert record["endpoint"] == "thing"
    assert record["tag"] == "thing-abc"
    assert record["hash"] == "abc123"
    assert record["commit"] == gh.head
    assert record["produced_by"] == "rule make_thing"
    assert record["run"] == "runa"
    assert record["promoted_at"]
    stored = json.loads(gh.releases["thing-main"][pointers.ASSET])
    assert stored == record
    create = next(c for c in gh.calls if c[2] == "create")
    assert "--latest=false" in create
    # A later version replaces the record in place.
    gh.releases["thing-def"] = {"data.tar.gz": ""}
    _promote(gh, "thing-def", needs="data.tar.gz")
    upload = [c for c in gh.calls if c[2] == "upload"]
    assert upload
    assert "--clobber" in upload[-1]
    assert pointers.read("thing", REPO, run=gh)["tag"] == "thing-def"


def test_an_unreleased_version_is_never_promoted(fake_releases) -> None:
    gh = fake_releases({})
    with pytest.raises(RuntimeError, match="thing-abc"):
        _promote(gh)
    # Released without the asset its readers need.
    gh = fake_releases({"thing-abc": {}})
    with pytest.raises(RuntimeError, match="data.tar.gz"):
        _promote(gh, needs="data.tar.gz")
    assert "thing-main" not in gh.releases


def test_only_a_commit_on_a_configured_branch_promotes(fake_releases) -> None:
    gh = fake_releases({"thing-abc": {}}, branches=("feature",))
    with pytest.raises(RuntimeError, match="main, next"):
        _promote(gh)
    assert "thing-main" not in gh.releases
    with pytest.raises(RuntimeError, match="no branch"):
        _promote(fake_releases({"thing-abc": {}}), branches=())
    gh = fake_releases({"thing-abc": {}}, branches=("next",))
    assert _promote(gh)["commit"] == gh.head


def test_a_missing_pointer_fails_rather_than_guess(fake_releases) -> None:
    # A newer version's release is never taken without the pointer.
    gh = fake_releases({"thing-abc": {}, "thing-def": {}})
    with pytest.raises(LookupError, match="thing-main"):
        pointers.read("thing", REPO, run=gh)
    gh.releases["thing-main"] = {}
    with pytest.raises(LookupError, match=pointers.ASSET):
        pointers.read("thing", REPO, run=gh)


def test_a_record_that_names_no_release_fails(fake_releases) -> None:
    gh = fake_releases({"thing-main": {pointers.ASSET: json.dumps({})}})
    with pytest.raises(ValueError, match="names no release"):
        pointers.read("thing", REPO, run=gh)
