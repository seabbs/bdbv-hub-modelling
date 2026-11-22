# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Fine-tune Chronos-2 in one pass while the outbreaks are simulated.

The trainer is `nowcastfoundation.finetune.stream`.

    uv run --group chronos python scripts/finetune_stream.py SIM_DIR OUT \
        --train-seeds FIRST LAST --held-out-seeds FIRST LAST \
        --publish
"""

from pathlib import Path

from nowcastfoundation.finetune.stream import main

if __name__ == "__main__":
    main(Path(__file__).resolve().parents[1])
