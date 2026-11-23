"""Write a run's block combinations table (rule block_combinations).

    python workflow/scripts/block_combinations.py SEGMENT OUT

SEGMENT is the run's final checkpoint folder, `train/n<N>`; OUT the
table (`nowcastfoundation.finetune.combinations.shares`).
"""

import json
import sys
from pathlib import Path

from nowcastfoundation.finetune import combinations


def main(argv: list[str]) -> None:
    """Write the shares of the segment's combination counts."""
    segment, out = Path(argv[0]), Path(argv[1])
    counts = json.loads((segment / combinations.FILE).read_text())
    combinations.shares(counts).to_csv(out, index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
