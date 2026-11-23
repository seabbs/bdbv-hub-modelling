#!/usr/bin/env bash
# Rebuild the hub rounds report (`hub_rounds`, rules/hub.smk) when the
# hub's data change. Checks the hub's target-data/time-series.csv commit
# every POLL seconds (default 600); when it differs from
# <ROOT>/hub/head.txt it writes it there, which reruns the report's
# rules, and runs the target. Run it from the workflow's worktree, for
# example under nohup; it never stops by itself.
#
#   POLL       seconds between checks
#   ONCE=true  check and build once, then exit
set -u
root=$(python3 workflow/flowlib/localconfig.py ROOT)
head="$root/hub/head.txt"
hub=InsightNet-US/BDBV-Modeling-Hub
mkdir -p "$root/hub"
while :; do
  sha=$(gh api "repos/$hub/commits?path=target-data/time-series.csv&per_page=1" \
    --jq '.[0].sha' 2>/dev/null)
  if [ -n "$sha" ] && [ "$sha" != "$(cat "$head" 2>/dev/null)" ]; then
    echo "$sha" > "$head"
    echo "$(date -u '+%F %T') hub data $sha: building hub_rounds"
    GITHUB_TOKEN=${GITHUB_TOKEN:-$(gh auth token 2>/dev/null)} \
      uv run --group paper snakemake -s workflow/Snakefile -c 1 hub_rounds ||
      echo "$(date -u '+%F %T') hub_rounds failed for $sha"
  fi
  [ "${ONCE:-false}" = true ] && exit 0
  sleep "${POLL:-600}"
done
