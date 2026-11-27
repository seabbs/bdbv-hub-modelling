#!/usr/bin/env bash
# Build the baseline candidates from the pinned inputs (prepare.sh):
# random walk first, since every other candidate is checked against it,
# then random walk on a nowcast, then the `direct` models.
#
# The walks are read from the fit store like the fitted candidates
# (FIT_SOURCE=store). Fit store stores their forecasts at each origin,
# so the submission, the backtest and the docs read the same ones. With
# FIT_SOURCE=store this first waits for the walks' stored forecasts at
# AS_OF (await_fits.sh, with its DISPATCH, WAIT_MINUTES, FIT_REF and
# FIT_STORE).
#
#   AS_OF       data vintage
#   FIT_SOURCE  store, or fit (the default) to run the walks here
# shellcheck source=packages/bdbvsubmission/steps/lib.sh
source "$(dirname "$0")/lib.sh"
resolve_model

: "${AS_OF:?}"
sci python -m bdbvsubmission.policy --policy "$POLICY" --stage baseline \
  --model "$MODEL" \
  > "$WORK/matrix-baseline.json"
python3 - "$WORK/matrix-baseline.json" << 'EOF' > "$WORK/legs-baseline.txt"
import json, sys
legs = json.load(open(sys.argv[1]))["include"]
for e in sorted(legs, key=lambda e: e["id"] != "randomwalk"):
    print("\t".join([e["id"], e["kind"], e["config"], e["label"],
                     e["model"] or "-"]))
EOF

if [ "${FIT_SOURCE:-fit}" = store ]; then
  # A walk candidate that names no model runs randomwalk (candidate.sh).
  walks=$(awk -F '\t' '$2 == "randomwalk" {
    print ($5 == "-" ? "randomwalk" : $5) }' "$WORK/legs-baseline.txt" \
    | sort -u | tr '\n' ' ')
  if [ -n "${walks// /}" ]; then
    MODELS=$walks "$(dirname "$0")/await_fits.sh"
  fi
fi

while IFS=$'\t' read -r id kind config label cmodel; do
  [ "$cmodel" = - ] && cmodel=""
  CAND_ID=$id CAND_KIND=$kind CAND_CONFIG=$config CAND_LABEL=$label \
    CAND_MODEL=$cmodel CAND_SAMPLER='{}' "$(dirname "$0")/candidate.sh"
done < "$WORK/legs-baseline.txt"
