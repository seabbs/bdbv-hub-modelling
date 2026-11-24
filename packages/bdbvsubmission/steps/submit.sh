#!/usr/bin/env bash
# Build the PR body and open or update the hub PR (or print it).
#
#   DRY_RUN=true    do everything except push and open the PR (default)
#   FORK            the bot's fork, e.g. seabbs-bot/BDBV-Modeling-Hub
#   HUB_REF, DATA_COMMIT, AS_OF, REFERENCE_DATE
#   METADATA        model metadata YAML to include in the PR (optional)
#   SOURCE_SHA      commit of this repo that produced the submission
# shellcheck source=packages/bdbvsubmission/steps/lib.sh
source "$(dirname "$0")/lib.sh"

: "${HUB_REF:?}" "${DATA_COMMIT:?}" "${REFERENCE_DATE:?}"
DRY_RUN=${DRY_RUN:-true}
FORK=${FORK:-seabbs-bot/BDBV-Modeling-Hub}
gate="${WORK:?}/gate"
sub=$(find "$gate/selected/submission" -name '*.csv' | sort | head -n 1)
plot="$gate/selected/nowcast.png"
# The weekly R_t plot, from select.sh when the file has wk R_t rows.
rt_plot="$gate/selected/rt.png"
rt_plot_url=""

if [ "$DRY_RUN" = "true" ]; then
  plot_url="nowcast.png"
  note "Dry run: the plots are in the run's artefacts, not published."
else
  run=${GITHUB_RUN_ID:-local}
  plot_url=$(sci open-or-update-pr publish-plot --plot "$plot" \
    --repo "$FORK" --dest "plots/$MODEL_ID/$REFERENCE_DATE/$run.png")
  if [ -f "$rt_plot" ]; then
    rt_plot_url=$(sci open-or-update-pr publish-plot --plot "$rt_plot" \
      --repo "$FORK" --dest "plots/$MODEL_ID/$REFERENCE_DATE/$run-rt.png") \
      || echo "::warning::publishing the R_t plot failed"
  fi
fi

body=(--gate "$gate" --reference-date "$REFERENCE_DATE"
  --model-id "$MODEL_ID" --as-of "${AS_OF:-}" --plot-url "$plot_url"
  --rt-plot-url "$rt_plot_url"
  --observed "$gate/time-series.csv" --policy "$POLICY" --out "$gate")
if [ "$DRY_RUN" = "true" ]; then
  body+=(--dry-run)
fi
sci build-pr-body "${body[@]}"

args=(submit --submission "$sub" --model-id "$MODEL_ID"
  --reference-date "$REFERENCE_DATE" --title "$(cat "$gate/pr-title.txt")"
  --body-file "$gate/pr-body.md" --fork "$FORK" --hub "$HUB")
if [ -n "${METADATA:-}" ]; then
  args+=(--metadata "$METADATA")
fi
if [ "$DRY_RUN" = "true" ]; then
  args+=(--dry-run)
fi
sci open-or-update-pr "${args[@]}" | tee "$gate/pr-outcome.txt"
