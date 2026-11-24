"""Each flowlib module has its own test file, named after it."""

from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
FLOWLIB = ROOT / "workflow" / "flowlib"
TESTS = Path(__file__).parent


def _modules():
    for path in sorted(FLOWLIB.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        parts = path.relative_to(FLOWLIB).with_suffix("").parts
        yield "_".join(parts)


@pytest.mark.parametrize("module", list(_modules()))
def test_each_flowlib_module_has_its_test_file(module):
    assert (TESTS / f"test_workflow_{module}.py").is_file()
