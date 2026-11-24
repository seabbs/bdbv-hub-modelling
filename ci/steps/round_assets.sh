#!/usr/bin/env bash
# Attach a Submit run's artefacts to its round's release, so the models
# a round did not submit are kept for later evaluation: every
# candidate's hub file, checks and run record (`candidate-<id>`), the
# gates' provenance with each candidate's hub validation log (`gate`,
# `baseline-gate`, `chronos-gate`) and the round's inputs (`inputs`).
# The fit directories (`fit-<id>`), the PR body and the fit-store wait
# are left out. The run's artefacts expire (candidates after 7 days), so
# release.sh calls this as each round is released and a dispatch of
# round-assets.yml backfills past rounds while they last.
#
#   round_assets.sh RUN_ID [TAG]
#
# TAG defaults to round-<reference date>, from the run's inputs. The
# pack, round-<ref>-run-<id>.tar.gz, replaces any of that name on the
# release. A round with no release yet gets nothing.
#
#   DRY_RUN            true to only say what it would attach (default true)
#   GITHUB_REPOSITORY  owner/name
# shellcheck source=ci/steps/lib.sh
source "$(dirname "$0")/lib.sh"

run=${1:?usage: round_assets.sh RUN_ID [TAG]}
repo=${GITHUB_REPOSITORY:?}
out="$WORK/round-assets"
pack="$out/run-$run"
rm -rf "$pack" && mkdir -p "$pack"

names=$(gh api "repos/$repo/actions/runs/$run/artifacts" --paginate \
  --jq '.artifacts[] | select(.expired | not) | .name')
kept=()
for name in $names; do
  case "$name" in
    inputs | gate | baseline-gate | chronos-gate | candidate-*) ;;
    *) continue ;;
  esac
  if gh run download "$run" -R "$repo" -n "$name" -D "$pack/$name"; then
    kept+=("$name")
  fi
done
if [ "${#kept[@]}" -eq 0 ]; then
  note "Run $run has no artefacts left to attach."
  exit 0
fi

tag=${2:-}
if [ -z "$tag" ]; then
  ctx="$pack/inputs/inputs/context.json"
  if [ ! -f "$ctx" ]; then
    echo "::error::run $run has no inputs artefact to name its round; give TAG"
    exit 1
  fi
  ref=$(python3 -c 'import json, sys
print(json.load(open(sys.argv[1]))["reference_date"])' "$ctx")
  tag="round-$ref"
fi
file="$out/$tag-run-$run.tar.gz"
tar -czf "$file" -C "$out" "run-$run"
size=$(du -h "$file" | cut -f1)

if ! gh release view "$tag" -R "$repo" > /dev/null 2>&1; then
  note "There is no release $tag yet: nothing attached for run $run."
  exit 0
fi
if [ "${DRY_RUN:-true}" != false ]; then
  note "Dry run: would attach $(basename "$file") ($size: ${kept[*]}) to $tag."
  exit 0
fi
gh release upload "$tag" "$file" -R "$repo" --clobber
note "Attached $(basename "$file") ($size: ${kept[*]}) to $tag."
