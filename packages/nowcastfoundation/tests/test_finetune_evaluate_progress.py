"""Pass progress of an offloaded evaluation (`finetune.evaluate`)."""

import pytest

pytest.importorskip("torch")

from nowcastfoundation.finetune import evaluate as fe  # noqa: E402


def test_each_pass_is_reported_inside_the_context():
    seen = []
    with fe.pass_progress(lambda name, s: seen.append((name, s >= 0))):
        assert fe.timed("held_out_loss", lambda x: x + 1, 1) == 2
    assert seen == [("held_out_loss", True)]
    # Outside the context a pass runs unreported.
    assert fe.timed("by_level", lambda: 3) == 3
    assert len(seen) == 1
