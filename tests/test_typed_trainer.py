"""The fine-tuning trainer modules are type-checked (#631).

Each module leaves the mypy override in pyproject.toml once it is typed,
one module per PR.
"""

import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[1]


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
    }
    assert not typed & _untyped()
