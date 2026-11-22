"""Every structured value is one labelled box, in readable units."""

from flowlib.dashboard import kv


def test_keys_are_labels_and_suffixes_are_units():
    out = kv.box(
        {
            "stage": "training n800",
            "split": {"wait_s": 0.027, "cut_s": 120.0, "train_s": 3251.2},
            "peak_rss_kb": 14455576,
            "step": 19778,
            "rate": 0.74612,
            "gone": None,
        }
    )
    assert "<span class=kvk>Stage</span> <b>training n800</b>" in out
    assert "<span class=kvk>Wait</span> <b>0.027 s</b>" in out
    assert "<span class=kvk>Cut</span> <b>2 min</b>" in out
    assert "<span class=kvk>Train</span> <b>54 min</b>" in out
    assert "<b>14.5 GB</b>" in out
    assert "<b>19,778</b>" in out
    assert "<b>0.746</b>" in out
    assert "Gone" not in out


def test_long_times_read_in_hours_and_days():
    assert kv.duration(3 * 3600 + 35 * 60) == "3 h 35 min"
    assert kv.duration(25 * 3600) == "1 d 1 h"


def test_a_list_is_its_items():
    out = kv.box({"waiting_for": [33431, 33432]})
    assert out.count("class=kvi") == 3
    assert "<b>33,431</b>" in out
