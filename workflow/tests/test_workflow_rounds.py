from types import SimpleNamespace

import pytest
from flowlib import rounds

SIM = {
    "seed_base": 30000,
    "batches": 3,
    "batch": 20,
    "draws": 5,
    "grid_file": "packages/nowcastfoundation/config/finetune/grid.yaml",
}


def _local(**extra: str) -> dict[str, str]:
    return {
        "HOST_NAME": "box",
        "ROOT": "/r",
        "REMOTE": "",
        "REMOTE_ROOT": "",
        "SIM_SEEDS": "",
        "NOTIFY": "",
        **extra,
    }


class _Wildcards(dict):
    """Snakemake's wildcards: by name and as a mapping."""

    __getattr__ = dict.__getitem__


def _rules() -> SimpleNamespace:
    batch = SimpleNamespace(
        counts="/r/rounds/r{round}/sims/counts-{seed}.csv.gz",
        settings="/r/rounds/r{round}/sims/settings-{seed}.csv",
    )
    return SimpleNamespace(
        simulate_batch=SimpleNamespace(output=batch),
        await_batch=SimpleNamespace(
            output=["/r/rounds/r{round}/arrived/{seed}.done"]
        ),
    )


def _rounds(config: dict | None = None, **local: str) -> rounds.Rounds:
    return rounds.Rounds(
        {"simulate": SIM, "rounds": [2], **(config or {})},
        _local(**local),
        _rules(),
    )


def test_a_round_is_its_block_of_seeds() -> None:
    r = _rounds()
    assert r.seeds_of("2") == [30200, 30201, 30202]
    assert r.round_of("30201") == 2
    assert r.dir == "/r/rounds"
    assert r.remote_dir == "/r/rounds"


def test_the_remote_rounds_take_the_remote_root() -> None:
    r = _rounds(REMOTE="host", REMOTE_ROOT="far")
    assert r.remote == "host"
    assert r.remote_dir == "far/rounds"


def test_a_host_simulates_its_listed_seeds_or_all() -> None:
    assert _rounds().here(30201)
    r = _rounds(SIM_SEEDS="30200 r3")
    assert r.here("30200")
    assert r.here(30301)
    assert not r.here(30201)


def test_a_batch_is_simulated_here_or_awaited() -> None:
    wc = _Wildcards(round="2", seed="30201")
    mine = _rounds().batch_files(wc)
    assert mine == {
        "counts": "/r/rounds/r2/sims/counts-30201.csv.gz",
        "settings": "/r/rounds/r2/sims/settings-30201.csv",
    }
    theirs = _rounds(SIM_SEEDS="30200").batch_files(wc)
    assert theirs == {"arrived": "/r/rounds/r2/arrived/30201.done"}


def test_a_round_s_records_and_files() -> None:
    r, wc = _rounds(), SimpleNamespace(round="2")
    assert r.round_records(wc)[0] == "/r/rounds/r2/records/record-30200.csv"
    files = r.round_sims(wc)
    assert len(files) == 9
    assert files[:3] == [
        "/r/rounds/r2/sims/counts-30200.csv.gz",
        "/r/rounds/r2/sims/settings-30200.csv",
        "/r/rounds/r2/sims/latent-30200.csv.gz",
    ]


def test_sources_name_each_round() -> None:
    r = _rounds({"rounds": [2, 3]})
    assert r.round_sources(None, ["a.csv", "b.csv"]) == "r2=a.csv r3=b.csv"
    assert r.summary_sources([2], ["s.csv"]) == "r2=/r/rounds/r2/sims:s.csv"


def test_a_batch_s_fields_name_this_host_and_grid(tmp_path) -> None:
    grid = tmp_path / "grid.yaml"
    grid.write_text("a: 1\n")
    r = rounds.Rounds(
        {"simulate": {**SIM, "grid_file": str(grid)}, "rounds": [2]},
        _local(),
        _rules(),
    )
    fields = r.batch_fields()
    assert fields["host"] == "box"
    assert fields["draws"] == 5
    assert fields["batch"] == 20
    assert fields["grid"] == r.grid_id()


def test_a_retired_round_is_not_released() -> None:
    grid = {"rounds": [2], "validation_round": 3}
    retired = {"first": 0, "last": 1}
    config = {"grid_rounds": grid, "retired_rounds": retired}
    assert _rounds({**config, "rounds": [2]}).release_check(None) == []
    assert _rounds(config).release_owner(_Wildcards(round="2")) == "grid"
    with pytest.raises(ValueError, match="earlier grid"):
        _rounds({**config, "rounds": [1]}).release_check(None)
