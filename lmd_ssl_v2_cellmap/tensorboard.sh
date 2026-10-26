#!/usr/bin/env bash
# Serve the lmd_ssl_v2_cellmap runs in one TensorBoard.
#
#   bash lmd_ssl_v2_cellmap/tensorboard.sh [port]
#   bash lmd_ssl_v2_cellmap/tensorboard.sh 6008 --smoke     # smoke runs instead
#   bash lmd_ssl_v2_cellmap/tensorboard.sh 6006 --list      # rebuild the tree, print, do not serve
#
# lmd_ssl_v2/tensorboard.sh with this experiment's runs: each arm in both tasks (<arm> and <arm>_fewshot). The tree
# is `tensorboard/` in this experiment's /nrs directory; smoke runs are linked under `tensorboard/smoke/`.
#
# WHAT TO READ
#
#   train/loss          Cross-entropy (plus dice_loss for 2b), the last step's 8 windows only: noisy; same-seed
#                       runs of a task see the same windows, so their spikes coincide.
#   train/mean_iou      Over the classes present in that step's windows, single-class voxels only.
#   train/pixel_accuracy  A voxel labelled with a composite counts as right when the prediction is one of its parts.
#   train/classes_present  How many classes the step's windows held.
#   grad_norm, lr       Clip 1.0; lr reaches 3e-4 after the warmup (5k large, 2k fewshot), then decays linearly.
set -euo pipefail

EXP=/nrs/scicompsoft/orhane/mia-train-experiments/lmd_ssl_v2_cellmap
RUNS=$EXP/runs
VIEW=$EXP/tensorboard
VENV=/groups/scicompsoft/home/orhane/myvenv
HERE="$(cd "$(dirname "$0")" && pwd)"
mapfile -t ARMS < <(cd "$HERE" && ls *.toml | sed 's/\.toml$//' | sort)

PORT=6006 SMOKE=0 LIST=0
for arg in "$@"; do
  case $arg in
    --smoke) SMOKE=1 ;;
    --list) LIST=1 ;;
    [0-9]*) PORT=$arg ;;
    *) echo "unknown argument $arg" >&2; exit 2 ;;
  esac
done

if [[ $SMOKE -eq 1 ]]; then RUNS=$RUNS/smoke; VIEW=$VIEW/smoke; fi   # smoke artifacts live one level down

port_busy () { ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]$1$"; }
if [[ $LIST -eq 0 ]]; then
  while port_busy "$PORT"; do echo "port $PORT is in use, trying $((PORT + 1))"; PORT=$((PORT + 1)); done
fi

# Replace the links in place rather than removing the directory: a TensorBoard already serving
# it holds files open, and on NFS that leaves the directory undeletable.
mkdir -p "$VIEW"; find "$VIEW" -maxdepth 1 -type l -delete
missing=()
link () {
  local name=$1 dir
  dir=$(ls -dt $2 2>/dev/null | head -1 || true)
  if [[ -n "$dir" && -d "${dir%/}/tensorboard" ]]; then
    ln -sfn "${dir%/}/tensorboard" "$VIEW/$name"; printf '  %-40s -> %s\n' "$name" "$(basename "${dir%/}")"
  else
    missing+=("$name")
  fi
}

echo "runs found:"
# Only the run directory's own _YYYYMMDD_HHMMSS suffix may follow the name: a bare `_*` would let 1e match
# 1e_..._fewshot's directories.
STAMP="[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9]"
for arm in "${ARMS[@]}"; do
  if [[ $SMOKE -eq 1 ]]; then
    link "$arm" "$RUNS/smoke_lmd2cm__${arm}_${STAMP}/"
  else
    link "$arm" "$RUNS/lmd2cm__${arm}_${STAMP}/"
  fi
done
(( ${#missing[@]} )) && echo "  not started: ${missing[*]}"
[[ $LIST -eq 1 ]] && exit 0
echo; echo "http://localhost:$PORT"
exec "$VENV/bin/tensorboard" --logdir "$VIEW" --port "$PORT"
