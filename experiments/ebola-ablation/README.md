# Ebola ablation

Status: exploratory. It is not part of the paper or the workflow.
Findings are in #1123.

## What it tests

The fine-tuned Chronos-2 does much worse than zero-shot on the BDBV hub data, while it does well on simulations.
The experiment finds the feature of the BDBV data that causes this.
Six held-out simulated outbreaks become six provinces of a hub-shaped data set, and one BDBV feature at a time is imposed on them (`ebola_ablation.py sim`): the release history, late-starting zones, small counts, a weekday cycle, longer delays and zones that stop reporting.
The hub data can also be moved one feature at a time towards the simulations (`ebola_ablation.py bdbv`).
Each variant runs through the BDBV check (`bdbvhub ft-check forecast`) with zero-shot and fine-tuned Chronos-2, scored against `randomwalknowcast`.

## Files

- `ebola_ablation.py`: the variants' hub data (`sim`, `bdbv`), the backtest config (`config`) and each variant's scores (`summary`).
- `ebola_ablation_report.py`: the page, every number computed from the files above.
- `tests/`: their tests (`uv run pytest experiments/ebola-ablation/tests`).

## Running it

From the repository root, with the tidy hub data in `data/` (`uv run bdbvhub data --raw data/raw --out data`), the held-out simulations `counts-33300.csv.gz` to `counts-33309.csv.gz` of the training run in `SIMS`, and that run's `releases.csv` in `RELEASES`.
Everything is written to `output/ebola-ablation/`.

```sh
E=experiments/ebola-ablation
OUT=output/ebola-ablation
CKPT=slots-c44bbe122e6c
uv run python $E/ebola_ablation.py config \
  --base packages/nowcastbdbv/config/backtest.toml \
  --samples 200 --max-delay 42 --out $OUT/backtest.toml
for v in daily weekly short fresh gap few few_daily late_start small \
  weekday long_delays stops bdbv_like; do
  uv run python $E/ebola_ablation.py sim $v \
    --sims $SIMS/counts-333{00..09}.csv.gz --src data --out $OUT/sim/$v/data
done
uv run python $E/ebola_ablation.py bdbv as_is --src data --out $OUT/bdbv/as_is/data
for d in $OUT/sim/* $OUT/bdbv/*; do
  for m in randomwalknowcast foundation_zeroshot foundation; do
    c=""
    case $m in foundation) c=$CKPT ;; esac
    DEVICE=cpu CHRONOS_BLOCK_LIMIT=1000 BDBV_DATA_DIR=$d/data \
      uv run --group chronos bdbvhub ft-check forecast --model $m \
      --checkpoint "$c" --config $OUT/backtest.toml \
      --data $d/data --out $d/forecasts/$m
  done
  BDBV_DATA_DIR=$d/data uv run python $E/ebola_ablation.py summary $d \
    --out $d/summary.json
done
uv run python $E/ebola_ablation_report.py $OUT/sim/* $OUT/bdbv/* \
  --checkpoint $CKPT --releases $RELEASES --out $OUT/ebola-ablation.html
```

On a CPU each Chronos-2 forecast of a simulated variant takes a few minutes, and of the hub data about 20 minutes.
