"""A segment's chunk timings outlast restarts: they are in its state."""

import json

from nowcastfoundation.finetune import interim
from nowcastfoundation.finetune import stream as fs
from nowcastfoundation.finetune import verify


def _split(wait=0.0, cut=0.0, train=0.0, eval_=0.0):
    return {"wait_s": wait, "cut_s": cut, "train_s": train, "eval_s": eval_}


def _through_state(timings):
    # The state is saved and read back as JSON (`save_state`).
    return json.loads(json.dumps({"timings": timings}))["timings"]


def test_a_new_segment_starts_its_own_timings():
    got = fs.segment_timings(None, "n200", now=100.0)
    assert got == {"segment": "n200", "started": 100.0, "chunks": [], "open": None}
    # The state kept with the previous checkpoint holds its timings.
    before = fs.segment_timings(None, "n100", now=10.0)
    fs.open_chunk(before, "1-100", 0, now=10.0)
    fs.close_chunk(before, _split(train=50.0), 400, now=60.0)
    got = fs.segment_timings(_through_state(before), "n200", now=100.0)
    assert got["segment"] == "n200"
    assert got["started"] == 100.0
    assert got["chunks"] == []


def test_a_resume_keeps_the_chunks_timed_before_it():
    timings = fs.segment_timings(None, "n200", now=0.0)
    fs.open_chunk(timings, "1-100", 0, now=0.0)
    first = fs.close_chunk(
        timings, _split(wait=1.0, cut=2.0, train=300.0, eval_=7.0), 400, now=310.0
    )
    assert first["chunk"] == "1-100"
    assert first["first"] == 1 and first["last"] == 100
    assert first["steps"] == 400
    assert first["seconds"] == 310.0
    assert first["started"] == 0.0 and first["ended"] == 310.0
    assert first["timed_from"] == 0 and first["resumed_at"] == []
    # The second chunk is saved part way, at batch 150, then the trainer
    # stops.
    split = fs.open_chunk(timings, "101-200", 0, now=320.0)
    assert split == _split()
    fs.note_split(timings, _split(cut=5.0, train=120.0))
    saved = _through_state(timings)
    # Restarted, the segment's timings are those saved.
    again = fs.segment_timings(saved, "n200", now=1000.0)
    assert again["started"] == 0.0
    assert again["chunks"] == [first]
    split = fs.open_chunk(again, "101-200", 150, now=1000.0)
    # The chunk's time so far carries on.
    assert split == _split(cut=5.0, train=120.0)
    split["cut_s"] += 3.0
    split["train_s"] += 200.0
    second = fs.close_chunk(again, split, 380, now=1210.0)
    assert second["seconds"] == 328.0
    assert second["timed_from"] == 0
    assert second["resumed_at"] == [150]
    assert second["started"] == 320.0
    assert [c["chunk"] for c in again["chunks"]] == ["1-100", "101-200"]
    assert again["open"] is None


def test_a_chunk_resumed_without_saved_timings_is_timed_from_there():
    # A state saved before timings were kept.
    timings = fs.segment_timings({}, "n200", now=50.0)
    split = fs.open_chunk(timings, "101-200", 150, now=50.0)
    assert split == _split()
    got = fs.close_chunk(timings, _split(train=100.0), 380, now=150.0)
    assert got["timed_from"] == 150
    assert got["resumed_at"] == [150]


def test_the_training_seconds_sum_every_attempt():
    timings = fs.segment_timings(None, "n200", now=0.0)
    fs.open_chunk(timings, "1-100", 0, now=0.0)
    fs.close_chunk(timings, _split(train=300.0, eval_=10.0), 400, now=310.0)
    fs.open_chunk(timings, "101-200", 0, now=320.0)
    fs.note_split(timings, _split(train=100.0))
    again = fs.segment_timings(_through_state(timings), "n200", now=900.0)
    split = fs.open_chunk(again, "101-200", 150, now=900.0)
    split["train_s"] += 50.0
    # The finished chunks and the open one's time over both attempts.
    assert fs.train_seconds(again, split) == 460.0
    assert fs.wall_seconds(again, now=1000.0) == 1000.0


def test_the_timings_are_written_beside_the_state(tmp_path):
    timings = fs.segment_timings(None, "n200", now=0.0)
    fs.open_chunk(timings, "1-100", 0, now=0.0)
    fs.close_chunk(timings, _split(train=300.0), 400, now=300.0)
    path = fs.write_timings(tmp_path, timings)
    assert path == tmp_path / fs.TIMINGS / "n200.json"
    got = json.loads(path.read_text())
    assert got["chunks"] == timings["chunks"]
    assert got["train_seconds"] == 300.0
    assert not list(path.parent.glob("*.tmp"))


def test_the_times_are_outside_the_hash_and_the_interim():
    for key in ("train_seconds", "wall_seconds"):
        assert key in verify.NOT_HASHED
        assert key in interim.NOT_ITS_OWN
    settings = {"a": 1, "hash": "x"}
    timed = settings | {"train_seconds": 5.0, "wall_seconds": 9.0}
    assert verify.settings_hash(timed) == verify.settings_hash(settings)
