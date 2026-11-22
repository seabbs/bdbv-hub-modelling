# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Score an adapter snapshot of `scripts/finetune_stream.py --eval offload`.

The scoring is `nowcastfoundation.finetune.evaluate`.

    uv run --group chronos python scripts/finetune_evaluate.py \
        TRAIN/evals/step-0000100 --train TRAIN --out result.json
"""

from nowcastfoundation.finetune.evaluate import main

if __name__ == "__main__":
    main()
