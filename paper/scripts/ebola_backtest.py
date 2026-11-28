"""Run the paper's own Ebola release backtest of the models it names.

The workflow's `ebola_backtest_draws` rule runs it from the repository
root with the `bdbvhub backtest` options:

    python paper/scripts/ebola_backtest.py --config C --models M... \
        --samples N --results R --output O

Unlike `bdbvhub backtest`, it runs the stored fine-tuned model
(`foundation`) itself, from the checkpoint `BDBV_CHRONOS_FT` pins, as in
the other studies, rather than reading the fit store
(`bdbvhub.cli.paper_backtest`).
"""

from __future__ import annotations

import sys

from bdbvhub.cli import paper_backtest

if __name__ == "__main__":
    sys.exit(paper_backtest(sys.argv[1:]))
