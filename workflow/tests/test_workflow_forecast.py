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
    ("device", "gpu"), [("cuda", 1), ("cpu", 0), ("mps", 0)]
)
def test_a_cuda_job_takes_a_whole_gpu(device, gpu):
    # Jobs never share a GPU: a CUDA job waits for a free one.
    assert forecast.Forecast(device, 2).gpu == gpu


def test_the_config_has_no_gpu_mode():
    assert set(CONFIG["forecast"]) == {"device", "threads"}


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


def test_cuda_lets_jax_fall_back_to_the_cpu():
    # Chronos-2 (torch) uses CUDA; JAX picks CUDA where jaxlib has it and
    # the CPU where it has not (archie's jaxlib): `cuda,cpu` fails there.
    fc = forecast.Forecast("cuda", 2)
    assert fc.variables()["DEVICE"] == "cuda"
    assert fc.variables()["JAX_PLATFORMS"] == ""
    assert "JAX_PLATFORMS=''" in fc.env


def test_jax_starts_with_the_cuda_platform_list_without_cuda():
    import os
    import subprocess
    import sys

    pytest.importorskip("jax")
    env = {**os.environ, **forecast.Forecast("cuda", 1).variables()}
    out = subprocess.run(
        [sys.executable, "-c", "import jax; print(jax.devices()[0].platform)"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, out.stderr[-1000:]
    assert out.stdout.strip() in {"cpu", "gpu", "cuda"}


SCORING_RULES = ("score_germany", "score_run_germany")
"""The German units, on the scoring device while a run trains."""


def _rule_text(name: str) -> str:
    for text in _rule_shells().values():
        if f"rule {name}:" in text:
            block = text.split(f"rule {name}:", 1)[1]
            return re.split(r"\nrule ", block, maxsplit=1)[0]
    raise AssertionError(f"no rule {name}")


def test_checkpoint_scoring_runs_on_the_scoring_device():
    for name in SCORING_RULES:
        text = _rule_text(name)
        assert "gpu=CHECKPOINT_FORECAST.gpu" in text, name
        assert "env=checkpoint_env" in text, name
        assert "FORECAST.device" not in text.replace(
            "CHECKPOINT_FORECAST.device", ""
        ), name
    # The BDBV check stays on the forecasting device: its blocks do not
    # fit an 8 GB GPU.
    assert "gpu=FORECAST.gpu" in _rule_text("ft_check_forecast")


EVALUATION_RULES = (
    "evaluate_base",
    "evaluate_checkpoint",
    "seven_day_check",
    "validation_forecasts",
    "validation_forecasts_base",
)
"""A checkpoint's held-out scoring: on the CPU while a run trains, a
whole GPU once none does, never pausing training."""


def test_evaluation_never_pauses_training():
    for name in EVALUATION_RULES:
        text = _rule_text(name)
        assert "gpu=CHECKPOINT_FORECAST.gpu" in text, name
        assert "env=checkpoint_env" in text, name
        assert "priority: evaluation_priority" in text, name
        assert "threads: EVALUATE_THREADS" in text, name
        assert "BOUNDARY" not in text, name


def test_evaluation_reruns_only_when_its_code_or_held_out_set_moves():
    # Sam, 25 Nov: outputs keyed by their inputs' content, so a restart
    # or another module's change never recomputes them.
    for name in EVALUATION_RULES:
        text = _rule_text(name)
        assert "code=SCORING.evaluator_key_input" in text, name
        assert "held_out=held_out_key" in text, name


def test_jobs_ask_for_the_memory_their_benchmarks_show():
    # Each from its own last benchmark (flowlib.memory), with a minimum in
    # the rule; no machine's figures in the config (Sam, 25 Nov).
    assert "memory" not in CONFIG
    assert "mem_mb" not in CONFIG["score"]
    for name in (
        "train_segment",
        "ft_check_forecast",
        "seven_day_check",
        "evaluate_checkpoint",
        "evaluate_base",
        "validation_forecasts",
        "score_germany",
        "score_run_germany",
    ):
        assert "mem_mb=memory.from_benchmark(" in _rule_text(name), name
