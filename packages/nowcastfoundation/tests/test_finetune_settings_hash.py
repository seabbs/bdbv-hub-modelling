"""One settings hash for the trainer, interim checkpoints and verify."""

import inspect

from nowcastfoundation.finetune import chronos as ft
from nowcastfoundation.finetune import interim, stream, verify


def test_memory_settings_are_outside_the_hash():
    # Verify leaves checkpoint_above out of the hash, so the trainer must
    # too, or its checkpoints fail verify and cannot be released.
    s = {"seed": 1, "note": "x"}
    want = verify.settings_hash(s)
    assert verify.settings_hash(s | {"checkpoint_above": 0}) == want
    assert verify.settings_hash(s | {"device": "cuda", "hash": "h"}) == want
    assert (
        verify.settings_problems(s | {"hash": want, "checkpoint_above": 9})
        == []
    )


def test_the_trainer_and_interim_use_verify_s_hash():
    assert not hasattr(ft, "_settings_hash")
    for module in (stream, interim):
        assert "verify.settings_hash(" in inspect.getsource(module)
