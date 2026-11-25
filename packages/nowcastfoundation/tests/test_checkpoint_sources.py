"""The fine-tuned checkpoint from the Hugging Face Hub, GitHub or a folder.

No test reaches the network: the Hub is a folder per revision, read
through `forecaster._hub_file`.
"""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path

import pytest

from nowcastfoundation import forecaster as fc


def _by_spec(seen: list[str] | None = None):
    """A `resolve_checkpoint` stand-in that gives its spec back."""

    @functools.cache
    def resolve(spec: str) -> str:
        if seen is not None:
            seen.append(spec)
        return spec

    return resolve


ADAPTER = {
    "adapter_config.json": b"{}",
    "adapter_model.safetensors": b"weights",
}


def _sums(files: dict[str, bytes]) -> str:
    return "".join(
        f"{hashlib.sha256(data).hexdigest()}  {name}\n"
        for name, data in files.items()
    )


def _write(folder: Path, settings: dict, files: dict[str, bytes]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (folder / name).write_bytes(data)
    (folder / "settings.json").write_text(json.dumps(settings))
    (folder / fc.SUMS_FILE).write_text(_sums(ADAPTER))


@pytest.fixture
def hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake Hub: `<root>/<repo>/<revision>/<file>`."""
    root = tmp_path / "hub"

    def hub_file(repo: str, revision: str, name: str) -> Path:
        path = root / repo / revision / name
        if not path.is_file():
            raise FileNotFoundError(f"{repo}@{revision} has no {name}")
        return path

    monkeypatch.setattr(fc, "_hub_file", hub_file)
    fc.hub_checkpoint.cache_clear()
    return root


def _config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "finetune.toml"
    path.write_text(text)
    return path


def test_a_hub_checkpoint_is_its_release_s_checkpoint(hub: Path) -> None:
    _write(
        hub / "org" / "chronos-ft" / "v1",
        {"hash": "abc", "series_bias": True},
        ADAPTER,
    )
    got = fc.hub_checkpoint("org/chronos-ft", "v1")
    # The same checkpoint as its GitHub release: the same hash and sums,
    # so the same repr and fit key; only where it is read from differs.
    release = fc.Checkpoint(
        "slots-abc",
        tuple((n, hashlib.sha256(d).hexdigest()) for n, d in ADAPTER.items()),
    )
    assert got == release
    assert repr(got) == repr(release)
    assert got.hub == ("org/chronos-ft", "v1")
    adapter = fc.finetuned_adapter(got)
    assert (adapter / "adapter_model.safetensors").read_bytes() == b"weights"


def test_a_hub_file_that_changed_fails_the_checksum(hub: Path) -> None:
    folder = hub / "org" / "m" / "v1"
    _write(folder, {"hash": "abc"}, ADAPTER)
    checkpoint = fc.hub_checkpoint("org/m", "v1")
    (folder / "adapter_model.safetensors").write_bytes(b"other")
    with pytest.raises(ValueError, match="SHA-256"):
        fc.finetuned_adapter(checkpoint)


def test_hub_checksums_must_cover_the_adapter(hub: Path) -> None:
    folder = hub / "org" / "m" / "v1"
    _write(folder, {"hash": "abc"}, ADAPTER)
    (folder / fc.SUMS_FILE).write_text(_sums({"adapter_config.json": b"{}"}))
    with pytest.raises(ValueError, match="adapter_model"):
        fc.hub_checkpoint("org/m", "v1")


def test_the_config_picks_the_source(
    hub: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(fc.FINETUNED_ENV, raising=False)
    _write(hub / "org" / "m" / "v2", {"hash": "abc"}, ADAPTER)
    conf = _config(
        tmp_path,
        'source = "huggingface"\nhash = "x"\nlocal = ""\n'
        '[huggingface]\nrepo = "org/m"\nrevision = "v2"\n',
    )
    assert fc.finetuned_checkpoint(conf).hub == ("org/m", "v2")
    seen = []
    monkeypatch.setattr(fc, "resolve_checkpoint", _by_spec(seen))
    conf = _config(
        tmp_path,
        'source = "github"\nhash = "slots-abc"\nlocal = ""\n'
        '[huggingface]\nrepo = ""\nrevision = ""\n',
    )
    assert fc.finetuned_checkpoint(conf) == "slots-abc"
    segment = tmp_path / "n100"
    _write(segment / "finetuned-ckpt", {"hash": "abc"}, ADAPTER)
    (segment / "settings.json").write_text(json.dumps({"hash": "abc"}))
    conf = _config(
        tmp_path,
        f'source = "local"\nhash = "x"\nlocal = "{segment}"\n'
        '[huggingface]\nrepo = ""\nrevision = ""\n',
    )
    assert fc.finetuned_checkpoint(conf).path == str(segment / "finetuned-ckpt")


def test_the_environment_pins_the_checkpoint(
    hub: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A run pins the hash it resolved once (BDBV_CHRONOS_FT). The Hub
    # gives it when its revision holds that checkpoint, else its release.
    _write(hub / "org" / "m" / "v2", {"hash": "abc"}, ADAPTER)
    monkeypatch.setattr(fc, "resolve_checkpoint", _by_spec())
    conf = _config(
        tmp_path,
        'source = "huggingface"\nhash = "x"\nlocal = ""\n'
        '[huggingface]\nrepo = "org/m"\nrevision = "v2"\n',
    )
    monkeypatch.setenv(fc.FINETUNED_ENV, "abc")
    assert fc.finetuned_checkpoint(conf).hub == ("org/m", "v2")
    monkeypatch.setenv(fc.FINETUNED_ENV, "slots-other")
    assert fc.finetuned_checkpoint(conf) == "slots-other"


@pytest.mark.parametrize(
    ("text", "match"),
    [
        (
            'source = "huggingface"\nhash = "x"\nlocal = ""\n'
            '[huggingface]\nrepo = ""\nrevision = ""\n',
            "repo",
        ),
        (
            'source = "huggingface"\nhash = "x"\nlocal = ""\n'
            '[huggingface]\nrepo = "org/m"\nrevision = ""\n',
            "revision",
        ),
        (
            'source = "local"\nhash = "x"\nlocal = ""\n'
            '[huggingface]\nrepo = ""\nrevision = ""\n',
            "local",
        ),
        (
            'source = "s3"\nhash = "x"\nlocal = ""\n'
            '[huggingface]\nrepo = ""\nrevision = ""\n',
            "source",
        ),
    ],
)
def test_an_incomplete_source_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str, match: str
) -> None:
    monkeypatch.delenv(fc.FINETUNED_ENV, raising=False)
    with pytest.raises(ValueError, match=match):
        fc.finetuned_checkpoint(_config(tmp_path, text))


def test_the_repository_s_config_names_a_source() -> None:
    root = Path(__file__).resolve().parents[3]
    conf = (
        root / "packages" / "nowcastfoundation" / "config" / "finetune.toml"
    ).read_text()
    import tomllib

    got = tomllib.loads(conf)
    assert got["source"] in fc.FINETUNED_SOURCES
    # The Hub repository is empty until one is set.
    assert set(got["huggingface"]) == {"repo", "revision"}


@pytest.mark.parametrize(
    ("error", "raised"),
    [
        ("EntryNotFoundError", FileNotFoundError),
        ("RevisionNotFoundError", LookupError),
        ("RepositoryNotFoundError", LookupError),
        ("LocalEntryNotFoundError", ConnectionError),
    ],
)
def test_hub_errors_say_what_failed(
    monkeypatch: pytest.MonkeyPatch, error: str, raised: type
) -> None:
    errors = pytest.importorskip("huggingface_hub.errors")
    import huggingface_hub

    cls = getattr(errors, error)

    def download(**kwargs: str) -> str:
        del kwargs
        # The HTTP errors want a response; a bare instance is enough here.
        raise cls.__new__(cls)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    with pytest.raises(raised, match="org/m@v1"):
        fc._hub_file("org/m", "v1", "settings.json")


def test_the_label_resolves_latest_from_the_listing_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The newest release by publish date, slots releases included; no
    # checksum file is read for the name.
    monkeypatch.delenv(fc.FINETUNED_ENV, raising=False)
    listing = [
        {"tag_name": t, "published_at": at, "draft": d}
        for t, at, d in [
            ("chronos-ft-old", "2026-11-01T00:00:00Z", False),
            ("chronos-slots-new", "2026-11-20T00:00:00Z", False),
            ("chronos-slots-draft", "2026-11-22T00:00:00Z", True),
            ("realdata-ft-de", "2026-11-23T00:00:00Z", False),
        ]
    ]
    monkeypatch.setattr(fc, "finetuned_releases", lambda: listing)
    conf = _config(tmp_path, 'source = "github"\nhash = "latest"\n')
    assert fc.latest_spec() == "slots-new"
    assert fc.finetuned_label(conf) == "slots-new"
    assert fc.finetuned_label(_config(tmp_path, 'hash = "abc"\n')) == "abc"
    monkeypatch.setenv(fc.FINETUNED_ENV, "slots-pinned")
    assert fc.finetuned_label(conf) == "slots-pinned"
