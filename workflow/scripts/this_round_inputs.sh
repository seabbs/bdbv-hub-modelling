#!/usr/bin/env bash
# The inputs of the this-round page (`this_round`, rules/report.smk):
# the latest Submit run on main's artefacts, the fit store, and every
# candidate's hub table for the run's round (`fit-store archive`), and
# every hub team's submitted files.
#
#   this_round_inputs.sh OUT_DIR DATA_DIR RUN_FILE
#
# RUN_FILE holds the run id first (workflow/scripts/hub_watch.sh writes
# it). Writes OUT_DIR/run/ (the run's `inputs`, `gate` and
# `baseline-gate` artefacts and run.json), OUT_DIR/store/ and
# OUT_DIR/candidates.csv.gz. A model with no good stored fit at the
# round's origin, such as a failed Chronos-2 fit, is left out.
set -eu
out=$1 data=$2 run_file=$3
repo=${REPO:-seabbs/bdbv-hub-modelling}
models=(renewalbvd renewalsp renewal randomwalknowcast randomwalk
  fm_chronos_daily_ragged fm_chronos_daily_ragged_ft)
run=$(awk '{print $1; exit}' "$run_file")
rm -rf "$out/run" && mkdir -p "$out/run"
for name in inputs gate baseline-gate; do
  gh run download "$run" -R "$repo" -n "$name" -D "$out/run/$name" ||
    echo "no $name artefact in run $run yet"
done
gh run view "$run" -R "$repo" --json databaseId,headSha,status,conclusion \
  --jq '{run: .databaseId, sha: .headSha, mode: (.status + " " + .conclusion)}' \
  > "$out/run/run.json"
ctx="$out/run/inputs/inputs/context.json"
origin=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["as_of"])' "$ctx")
ref=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["reference_date"])' "$ctx")
rm -rf "$out/store"
uv run fit-store-branch fetch "$out/store"
BDBV_DATA_DIR=$data uv run bdbvhub fit-store archive --store "$out/store" \
  --data "$data" --origin "$origin" --reference-date "$ref" \
  --models "${models[@]}" --out "$out/candidates.csv.gz" ||
  echo "some candidates have no good stored fit at $origin"
test -s "$out/candidates.csv.gz"
# Every hub team's files (read-only from the hub repo), for the round's
# submissions beside ours: openly licensed ones in hub/, others in
# hub-raw/ (`bdbvhub.compare.fetch_hub_submissions`).
rm -rf "$out/hub" "$out/hub-raw"
uv run python -c 'import sys
from bdbvhub.compare import fetch_hub_submissions
fetch_hub_submissions(sys.argv[1], sys.argv[2])' "$out/hub" "$out/hub-raw"
