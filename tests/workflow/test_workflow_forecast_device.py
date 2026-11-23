"""One device and thread count for every forecasting job (flowlib.forecast)."""

import re
from pathlib import Path

import pytest
import yaml
from flowlib import forecast

ROOT = Path(__file__).parents[2]
CONFIG = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text())


def test_the_config_names_the_device_and_threads():
    got = forecast.settings(CONFIG, {})
    assert got.device in forecast.DEVICES
    assert got.threads >= 1


def test_a_machine_overrides_the_config():
    got = forecast.settings(
        CONFIG, {"FORECAST_DEVICE": "cuda", "FORECAST_THREADS": "8"}
    )
    assert (got.device, got.threads) == ("cuda", 8)


def test_an_unknown_device_is_refused():
    with pytest.raises(ValueError, match="not one of"):
        forecast.settings(CONFIG, {"FORECAST_DEVICE": "auto"})


@pytest.mark.parametrize(
    ("device", "mode", "gpu"),
    [("cuda", "exclusive", 1), ("cuda", "share", 0), ("cpu", "exclusive", 0)],
)
def test_only_exclusive_cuda_takes_the_trainer_s_gpu(device, mode, gpu):
    assert forecast.Forecast(device, 2, mode).gpu == gpu


def test_the_env_names_torch_and_jax_devices_and_threads():
    env = forecast.Forecast("mps", 3).variables()
    assert env["DEVICE"] == "mps"
    # JAX has no MPS backend, so its models run on the CPU.
    assert env["JAX_PLATFORMS"] == "cpu"
    assert env["OMP_NUM_THREADS"] == "3"


PICKED = r"--device (cpu|cuda|mps|auto)\b|EVAL_DEVICE"
"""A device named in a rule rather than read from FORECAST."""


def _rule_shells() -> dict[str, str]:
    files = [
        *ROOT.glob("workflow/rules/*.smk"),
        ROOT / "paper/workflow/Snakefile",
    ]
    return {str(f.relative_to(ROOT)): f.read_text() for f in files}


def test_no_rule_picks_a_device_itself():
    # Every forecasting rule reads FORECAST; only the trainer and the
    # simulator (JAX on the CPU by design) name a device in their shells.
    found = []
    for name, text in _rule_shells().items():
        if name.endswith(("finetune.smk", "simulate.smk")):
            continue
        found += [f"{name}: {m.group(0)}" for m in re.finditer(PICKED, text)]
    assert found == []
