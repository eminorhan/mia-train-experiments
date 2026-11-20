#!/usr/bin/env bash
# Serve the lmd_ssl_v3 arms in one TensorBoard: the real runs only, smoke runs are never linked.
#
#   bash lmd_ssl_v3/tensorboard.sh [port]     # default 6006; takes the next free port if that one is in use
#   bash lmd_ssl_v3/tensorboard.sh --list     # rebuild the links and print them, do not serve
#
# lmd_ssl_v2/tensorboard.sh with this experiment's arms and without its smoke view. Builds one link per arm, to
# the newest run directory of that arm, under `tensorboard/` in this experiment's /nrs directory. An arm that
# starts while TensorBoard is up appears once the links are rebuilt: run `--list` in another terminal.
#
# THE ARMS (all: SimMIM, DINOv3 ViT-L at patch 8 from LVD-1689M, global batch 32, 250k steps)
#   1a_base          the default v3 data config: 8 nm, 256^3 crops, size exponent 0.4
#   1b_em            EM volumes only
#   1c_exm           ExM volumes only
#   1d_res_4to16nm   voxel size drawn per crop from 4 to 16 nm
#   1e_native        every volume at its stored voxel size
#   1f_p384          384^3 crops, on four nodes with one crop per rank
#   1g_data10pct     10% of every volume's region
#   1h_data1pct      1% of every volume's region
#   1i_swe0.2        size exponent 0.2
#   1j_swe0.8        size exponent 0.8
#
# WHAT TO READ
#   train/loss, val/loss   Masked L1. Not comparable between arms (each draws its own crops) or with lmd_ssl_v2
#                          (v2 divided 8-bit volumes by 255; v3 windows every volume at p0.1-p99.9, clipped).
#   grad_norm              About 1 at step 1k, under 0.3 from 2k, under 0.1 after 5k in lmd_ssl_v2's 1a_b32.
#   samples_per_s          About 16 for the one-node arms (2.0 s per step), about 6.4 for 1f (4.98 s per step).
#   data_wait_frac         Near 0, with a short spike every 31,250 steps: the loader restarts at an epoch's end.
#   lr                     3e-4 at step 5k, then linear to 3e-7 at 250k. Validation every 10k steps.
set -euo pipefail

EXP=/nrs/scicompsoft/orhane/mia-train-experiments/lmd_ssl_v3   # this experiment's home on /nrs
RUNS=$EXP/runs
VIEW=$EXP/tensorboard
VENV=/groups/scicompsoft/home/orhane/myvenv
ARMS=(1a_base 1b_em 1c_exm 1d_res_4to16nm 1e_native 1f_p384 1g_data10pct 1h_data1pct 1i_swe0.2 1j_swe0.8)

PORT=6006 LIST=0
for arg in "$@"; do
  case $arg in
    --list) LIST=1 ;;
    [0-9]*) PORT=$arg ;;
    *) echo "unknown argument $arg" >&2; exit 2 ;;
  esac
done

port_busy () { ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]$1$"; }
if [[ $LIST -eq 0 ]]; then
  while port_busy "$PORT"; do echo "port $PORT is in use, trying $((PORT + 1))"; PORT=$((PORT + 1)); done
fi

# Replace the links in place rather than removing the directory: a TensorBoard already serving it holds files
# open, and on NFS that leaves the directory undeletable.
mkdir -p "$VIEW"; find "$VIEW" -maxdepth 1 -type l -delete
missing=()
echo "arms found:"
# Only the run directory's own _YYYYMMDD_HHMMSS suffix may follow the arm name. Smoke runs live in runs/smoke/
# under other names, so they never match.
STAMP="[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9]"
for arm in "${ARMS[@]}"; do
  dir=$(ls -dt "$RUNS"/lmd3__"$arm"_${STAMP}/ 2>/dev/null | head -1 || true)
  if [[ -n "$dir" && -d "${dir%/}/tensorboard" ]]; then
    ln -sfn "${dir%/}/tensorboard" "$VIEW/$arm"; printf '  %-16s -> %s\n' "$arm" "$(basename "${dir%/}")"
  else
    missing+=("$arm")
  fi
done
(( ${#missing[@]} )) && echo "  not started: ${missing[*]}"
[[ $LIST -eq 1 ]] && exit 0
echo; echo "http://localhost:$PORT"
exec "$VENV/bin/tensorboard" --logdir "$VIEW" --port "$PORT"
