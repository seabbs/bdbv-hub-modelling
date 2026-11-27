from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nowcastbdbv.data import tidy
from nowcastbdbv.data.tidy import load_daily, load_delays, load_target
from nowcasteval.backtest import BacktestConfig
from nowcastutils.base import ForecastTask

# Chronos-2 never detects a device (`chronos2.torch_device`): the tests
# run on the CPU unless the environment names one, as CI does.
os.environ.setdefault("DEVICE", "cpu")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "tests" / "fixtures" / "data"
"""A frozen copy of the tidy hub data. It is never refreshed: the
workflows fetch the live data, and the tests read only this copy."""

FIXTURE_VINTAGES = (
    "2026-08-18",
    "2026-10-09",
    "2026-10-17",
    "2026-10-24",
    "2026-11-01",
    "2026-11-08",
)
"""The real vintages in the fixture, oldest first."""

# The loaders and the CLI read `data/` by default. Point both at the
# fixture before any test builds a parser or loads a table.
os.environ["BDBV_DATA_DIR"] = str(DATA)
tidy._DEFAULT_DATA_DIR = DATA


FAKE_CHECKPOINT = "fakefake0001"
"""Settings hash of the fake latest fine-tuned checkpoint release."""


def _write_checkpoint_release(
    root: Path,
    tag_hash: str,
    adapter: dict[str, bytes],
    settings_hash: str,
    prefix: str = "chronos-ft",
) -> None:
    """Write a fake release `<prefix>-<tag_hash>` under `root`.

    Its assets are the adapter files, `settings.json` and their
    `sha256sums.txt`, as `finetune-stream` writes them.
    """
    import hashlib
    import json

    assets = root / f"{prefix}-{tag_hash}"
    assets.mkdir(parents=True)
    files = {**adapter, "settings.json": json.dumps({"hash": settings_hash})}
    for name, data in files.items():
        raw = data.encode() if isinstance(data, str) else data
        (assets / name).write_bytes(raw)
    (assets / "sha256sums.txt").write_text(
        "".join(
            f"{hashlib.sha256((assets / n).read_bytes()).hexdigest()}  {n}\n"
            for n in files
        )
    )


def write_main(root: Path, spec: str, run: str = "runa") -> None:
    """Name checkpoint `spec` the main run's newest in fake `chronos-main`."""
    import json

    from nowcastfoundation import forecaster as foundation

    folder = root / foundation.MAIN_TAG
    folder.mkdir(parents=True, exist_ok=True)
    record = {"run": run, "spec": spec, "tag": foundation.release_tag(spec)}
    (folder / foundation.MAIN_FILE).write_text(json.dumps(record))


@pytest.fixture(autouse=True)
def _fake_checkpoint_releases(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Resolve the fine-tuned checkpoint from a fake release list.

    No test reaches GitHub. The main run's newest checkpoint
    (`chronos-main`) is the release `chronos-ft-<FAKE_CHECKPOINT>`, with
    its `sha256sums.txt` only, and the models take it
    (`BDBV_CHRONOS_FT=latest`), whatever
    packages/nowcastfoundation/config/finetune.toml pins.
    """
    from nowcastfoundation import forecaster as foundation

    root = tmp_path_factory.getbasetemp() / "checkpoint-releases"
    if not root.exists():
        _write_checkpoint_release(
            root, FAKE_CHECKPOINT, {"adapter_model.safetensors": b"w",
                                    "adapter_config.json": b"{}"},
            FAKE_CHECKPOINT,
        )  # fmt: skip
        for name in ("adapter_model.safetensors", "adapter_config.json"):
            (root / f"chronos-ft-{FAKE_CHECKPOINT}" / name).unlink()
        write_main(root, FAKE_CHECKPOINT)
    monkeypatch.setenv(foundation.FINETUNED_ENV, "latest")
    monkeypatch.setattr(foundation, "FINETUNED_URL", root.as_uri())
    monkeypatch.setattr(foundation, "RETRY_SECONDS", (0, 0, 0))
    foundation.resolve_checkpoint.cache_clear()
    yield root
    foundation.resolve_checkpoint.cache_clear()


@pytest.fixture(autouse=True)
def _keep_the_checkpoint_pin():  # type: ignore[no-untyped-def]
    """Restore `BDBV_CHRONOS_FT` after each test.

    `nowcastgermany.checkpoint._pin` writes it to `os.environ` for the
    rest of a scoring process, which a monkeypatch would not undo when it
    was unset before.
    """
    from nowcastfoundation.forecaster import FINETUNED_ENV

    before = os.environ.get(FINETUNED_ENV)
    yield
    if before is None:
        os.environ.pop(FINETUNED_ENV, None)
    else:
        os.environ[FINETUNED_ENV] = before


@pytest.fixture
def write_release():  # type: ignore[no-untyped-def]
    """Write a fake checkpoint release (`_write_checkpoint_release`)."""
    return _write_checkpoint_release


@pytest.fixture
def point_main():  # type: ignore[no-untyped-def]
    """Name a fake checkpoint the main run's newest (`write_main`)."""
    return write_main


class FakeReleases:
    """A stand-in for `gh release` and the `git` a promotion runs.

    `releases` maps each tag to its assets' names and text. HEAD is
    `head`, on the remote branches in `branches`. Every command is kept
    in `calls`.
    """

    def __init__(
        self,
        releases: dict[str, dict[str, str]] | None = None,
        head: str = "c0ffee",
        branches: tuple[str, ...] = ("main",),
    ) -> None:
        self.releases = releases if releases is not None else {}
        self.head = head
        self.branches = set(branches)
        self.calls: list[list[str]] = []

    def __call__(
        self, command: list[str], **_: object
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(command))

        def done(
            code: int, out: str = "", err: str = ""
        ) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(command, code, out, err)

        if command[0] == "git":
            if command[1] == "rev-parse":
                return done(0, self.head + "\n")
            if command[1] == "fetch":
                return done(0)
            if command[1:3] == ["merge-base", "--is-ancestor"]:
                branch = command[4].removeprefix("origin/")
                ok = command[3] == self.head and branch in self.branches
                return done(0 if ok else 1)
            raise AssertionError(command)
        action, tag = command[2], command[3]
        files = [
            Path(f) for f in command[4:]
            if not f.startswith("-") and "\n" not in f and Path(f).is_file()
        ]  # fmt: skip
        if action == "view":
            if tag not in self.releases:
                return done(1, err="release not found")
            names = [{"name": n} for n in self.releases[tag]]
            return done(0, json.dumps({"assets": names}))
        if action == "download":
            if tag not in self.releases:
                return done(1, err="release not found")
            name = command[command.index("-p") + 1]
            if name not in self.releases[tag]:
                return done(1, err="no assets match the file pattern")
            return done(0, self.releases[tag][name])
        if action == "create":
            if tag in self.releases:
                return done(1, err="a release with the same tag name exists")
            self.releases[tag] = {}
        if action in ("create", "upload"):
            if tag not in self.releases:
                return done(1, err="release not found")
            for f in files:
                self.releases[tag][f.name] = f.read_text()
            return done(0, "ok")
        if action == "edit":
            return done(0 if tag in self.releases else 1, "ok")
        raise AssertionError(command)


@pytest.fixture
def fake_releases() -> type[FakeReleases]:
    """Return the fake `gh release` and `git` (`FakeReleases`)."""
    return FakeReleases


@pytest.fixture
def fake_checkpoint() -> str:
    """Settings hash of the fake latest checkpoint release."""
    return FAKE_CHECKPOINT


@pytest.fixture(autouse=True)
def _one_province_at_a_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fit per-province variants in this process, where stand-ins reach.

    Tests that check the worker processes set the variable themselves.
    """
    monkeypatch.setenv("BDBV_PROVINCE_WORKERS", "1")


@pytest.fixture(scope="session")
def target() -> pd.DataFrame:
    return load_target(DATA)


@pytest.fixture(scope="session")
def vintages() -> list[str]:
    """`FIXTURE_VINTAGES`, for tests that pin expectations to them."""
    return list(FIXTURE_VINTAGES)


@pytest.fixture(scope="session")
def delays() -> pd.DataFrame:
    return load_delays(DATA)


@pytest.fixture(scope="session")
def daily() -> pd.DataFrame:
    return load_daily(DATA)


@pytest.fixture(scope="session")
def tasks_json() -> Path:
    return DATA / "hub-config" / "tasks.json"


@pytest.fixture(scope="session")
def schema_json() -> Path:
    return DATA / "hub-config" / "model-metadata-schema.json"


@pytest.fixture
def config() -> BacktestConfig:
    return BacktestConfig(n_samples=100, seed=1)


@pytest.fixture(scope="session")
def health_zones() -> pd.DataFrame:
    from nowcastbdbv.data.tidy import load_health_zones

    return load_health_zones(DATA)


@pytest.fixture(scope="session")
def daily_zones() -> pd.DataFrame:
    from nowcastbdbv.data.tidy import load_daily_zones

    return load_daily_zones(DATA)


class FlatModel:
    """Latest reported week carried forward, a stand-in for harness tests.

    The random walk models need release snapshots and cannot forecast
    most origins, so tests of the backtest, scoring and report machinery
    run this model instead.
    """

    name = "flat"
    supports_mask = False

    def forecast(self, task: ForecastTask) -> pd.DataFrame:
        rng = task.rng()
        weeks = list(task.target_end_dates)
        frames = []
        for loc in task.locations:
            series = (
                task.target.loc[task.target["location"] == loc]
                .set_index("target_end_date")["total"]
                .sort_index()
            )
            level = float(series.iloc[-1]) if len(series) else 0.0
            values = np.column_stack(
                [
                    np.full(task.n_samples, float(series[w]))
                    if w in series.index
                    else rng.poisson(level, task.n_samples).astype(float)
                    for w in weeks
                ]
            )
            frames.append(
                pd.DataFrame(
                    {
                        "location": loc,
                        "target_end_date": np.tile(weeks, task.n_samples),
                        "sample": np.repeat(
                            np.arange(1, task.n_samples + 1), len(weeks)
                        ),
                        "value": values.reshape(-1),
                    }
                )
            )
        final = pd.concat(frames, ignore_index=True)
        if not task.report_by:
            return final
        # Counts reported by a later date are the final counts to then.
        later = [
            final.loc[final["target_end_date"] <= date].assign(as_of=date)
            for date in task.report_by
        ]
        return pd.concat(
            [final.assign(as_of=np.nan), *later], ignore_index=True
        )


@pytest.fixture
def flat_model(monkeypatch: pytest.MonkeyPatch) -> str:
    """Register `FlatModel` as `flat` for one test and return its name."""
    from nowcastutils import base

    monkeypatch.setitem(
        base._REGISTRY,
        "flat",
        base.ModelSpec("flat", FlatModel, "Test stand-in."),
    )
    return "flat"
