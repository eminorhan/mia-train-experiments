#!/usr/bin/env bash
# Run a command with `--scratch <dir>` appended, <dir> on this node's own disk with at least <GB> free, deleted
# afterwards. For exact mws on the full task: it spills each streamed volume's edges to --scratch (13 B per edge,
# 551 GB for the zebrafish doublecube), which /nrs (shared, 93% full on 2026-10-23) should not take.
#
#   bash lmd_ssl_v2/local_scratch.sh <GB> <command> [args...]
set -euo pipefail
need=$1; shift
for root in "/scratch/$USER" /tmp; do
  mkdir -p "$root" 2>/dev/null || continue
  free=$(df -P -BG "$root" | awk 'NR == 2 {print $4 + 0}')
  if (( free >= need )); then
    dir=$(mktemp -d "$root/lmd2_${LSB_JOBID:-$$}_XXXX")
    trap 'rm -rf "$dir"' EXIT
    echo "[local_scratch] $dir on $(hostname): ${free} GB free"
    "$@" --scratch "$dir"
    exit
  fi
done
echo "[local_scratch] no local disk on $(hostname) with ${need} GB free" >&2
exit 1
