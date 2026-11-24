#!/usr/bin/env bash
# Keep the hub pages current (rules/report.smk): `hub_rounds` when the
# hub's data change, `this_round` when they change, when the latest
# Submit run on main or its status changes, or when the fit store moves.
# Every POLL seconds (default 600) it writes what changed under
# <ROOT>/hub/ (head.txt: the hub's time-series.csv commit; round/run.txt:
# the run id and status; round/store.txt: the fit-store branch commit),
# which reruns the rules that read it, and builds the targets. Run it
# from the workflow's worktree, for example under nohup; it never stops
# by itself.
#
#   POLL       seconds between checks
#   ONCE=true  check and build once, then exit
set -u
root=$(python3 workflow/flowlib/localconfig.py ROOT)
hub=InsightNet-US/BDBV-Modeling-Hub
repo=${REPO:-seabbs/bdbv-hub-modelling}
mkdir -p "$root/hub/round"

# Write VALUE to FILE when it differs; print 1 if it did.
update() {
  local file=$1 value=$2
  [ -n "$value" ] || return 0
  if [ "$value" != "$(cat "$file" 2>/dev/null)" ]; then
    echo "$value" > "$file"
    echo 1
  fi
}

build() {
  GITHUB_TOKEN=${GITHUB_TOKEN:-$(gh auth token 2>/dev/null)} \
    uv run --group paper snakemake -s workflow/Snakefile -c 1 "$@" ||
    echo "$(date -u '+%F %T') $* failed"
}

while :; do
  data=$(update "$root/hub/head.txt" "$(gh api \
    "repos/$hub/commits?path=target-data/time-series.csv&per_page=1" \
    --jq '.[0].sha' 2>/dev/null)")
  run=$(update "$root/hub/round/run.txt" "$(gh run list -R "$repo" \
    -w submit.yml -b main -L 1 --json databaseId,status,conclusion \
    --jq '.[0] | "\(.databaseId) \(.status) \(.conclusion)"' 2>/dev/null)")
  store=$(update "$root/hub/round/store.txt" \
    "$(git ls-remote origin fit-store 2>/dev/null | cut -f1)")
  if [ -n "$data" ]; then
    echo "$(date -u '+%F %T') hub data moved: building hub_rounds"
    build hub_rounds
  fi
  if [ -n "$data$run$store" ]; then
    echo "$(date -u '+%F %T') round inputs moved: building this_round"
    build this_round
  fi
  [ "${ONCE:-false}" = true ] && exit 0
  sleep "${POLL:-600}"
done
