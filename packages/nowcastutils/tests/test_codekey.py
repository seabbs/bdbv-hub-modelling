"""Code keys from the modules a job imports (`nowcastutils.codekey`)."""

from pathlib import Path

from nowcastutils import codekey


def test_the_closure_follows_first_party_imports_only():
    files = codekey.module_files(["nowcastgermany.evaluation"])
    names = {Path(f).as_posix() for f in files}
    assert any(n.endswith("nowcastgermany/evaluation.py") for n in names)
    # Its imports, and their packages' __init__, are in.
    assert any(n.endswith("nowcastutils/__init__.py") for n in names)
    # The trainer is not: no forecast imports it.
    assert not any(n.endswith("finetune/stream.py") for n in names)
    # Nothing outside the repository's packages.
    assert all("/packages/" in n for n in names)


def test_the_evaluator_s_closure_holds_the_trainer_it_imports():
    files = codekey.module_files(["nowcastfoundation.finetune.evaluate"])
    assert any(Path(f).as_posix().endswith("finetune/stream.py") for f in files)


def test_the_key_moves_with_a_file_s_contents(tmp_path):
    a = tmp_path / "a.txt"
    a.write_text("1")
    first = codekey.digest([a])
    a.write_text("2")
    assert codekey.digest([a]) != first
    assert len(first) == 12
