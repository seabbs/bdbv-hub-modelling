# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Check that a fine-tuned checkpoint's hash recomputes from its commit.

The check is `nowcastfoundation.finetune.verify`.

    python scripts/finetune_verify.py <ROOT>/runs/runa/train/n800
"""

from nowcastfoundation.finetune.verify import main

if __name__ == "__main__":
    main()
