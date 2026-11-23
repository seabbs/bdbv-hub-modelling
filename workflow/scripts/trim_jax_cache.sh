#!/usr/bin/env bash
# Empty a JAX compile cache folder once it holds more than CAP_GB, so the
# cache simulation batches share stays bounded.
#
#   workflow/scripts/trim_jax_cache.sh FOLDER CAP_GB
#
# The cache runs with no cap of JAX's own, as JAX's cap scans the whole
# folder on every write. Only JAX's entry files (*-cache, *-atime) are
# removed; a batch reading one as it goes compiles it again. An empty
# FOLDER, or one not there yet, does nothing.
set -euo pipefail
folder=$1
cap_gb=$2
[ -n "$folder" ] && [ -d "$folder" ] || exit 0
used_kb=$(du -sk "$folder" | cut -f1)
cap_kb=$(awk -v gb="$cap_gb" 'BEGIN {print int(gb * 1000000)}')
if [ "$used_kb" -gt "$cap_kb" ]; then
  find "$folder" -maxdepth 1 -type f \( -name '*-cache' -o -name '*-atime' \) -delete
fi
