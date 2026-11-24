"""The fine-tuning trainer modules are type-checked (#631).

Every trainer module is typed, so none is left in a mypy override.
"""

import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[3]


def _untyped() -> set[str]:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    out: set[str] = set()
    for override in config["tool"]["mypy"].get("overrides", []):
        if override.get("ignore_errors"):
            out |= set(override["module"])
    return out


def test_the_typed_trainer_modules_are_checked():
    typed = {
        "nowcastfoundation.finetune.verify",
        "nowcastfoundation.finetune.evaluate",
        "nowcastfoundation.finetune.chronos",
        "nowcastfoundation.finetune.stream",
    }
    assert not typed & _untyped()
