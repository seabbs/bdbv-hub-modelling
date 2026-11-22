"""Ready the trainer's folder for one training segment.

    prepare_segment.py TRAIN [--prev DIR --prev-n N]

Puts the state the segment resumes from in TRAIN/state
(`flowlib.segments.resume_state`).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from flowlib import segments


def main() -> None:
    """Ready the segment's folder."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("train", type=Path)
    p.add_argument("--prev", type=Path)
    p.add_argument("--prev-n", type=int)
    a = p.parse_args()
    print("state:", segments.resume_state(a.train / "state", a.prev, a.prev_n))


if __name__ == "__main__":
    main()
