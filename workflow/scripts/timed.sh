#!/usr/bin/env bash
# Run a command and end its stderr with its peak memory and wall time,
# as `peak_rss_kb <KB> wall_s <s>` (flowlib.manifest reads it), with GNU
# or BSD (macOS) time. Where neither is installed (some cluster nodes),
# the wall time alone, as `wall_s <s>`.
#
#   workflow/scripts/timed.sh COMMAND [ARGS...]
#
# TIMED_TIME names the time program (default /usr/bin/time).
set -uo pipefail
time_bin=${TIMED_TIME:-/usr/bin/time}
if [ ! -x "$time_bin" ]; then
  start=$(date +%s.%N)
  "$@"
  rc=$?
  awk -v a="$start" -v b="$(date +%s.%N)" \
    'BEGIN { printf "wall_s %.2f\n", b - a }' >&2
  exit "$rc"
fi
if "$time_bin" -f "" true 2> /dev/null; then
  exec "$time_bin" -f "peak_rss_kb %M wall_s %e" "$@"
fi
# Its stderr passes through as it is written and is kept for the end.
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
out=$tmp/stderr
mkfifo "$tmp/pipe"
tee "$out" >&2 < "$tmp/pipe" &
tee_pid=$!
"$time_bin" -l "$@" 2> "$tmp/pipe"
rc=$?
wait "$tee_pid"
# BSD time gives the maximum resident set size in bytes.
awk '
  / real / { wall = $1 }
  /maximum resident set size/ { rss = int($1 / 1024) }
  END { if (rss != "") print "peak_rss_kb " rss " wall_s " wall }' "$out" >&2
exit "$rc"
