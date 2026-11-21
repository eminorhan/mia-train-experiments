#!/usr/bin/env bash
# attention_variants: throughput of one or more arms on the GPUs of one job, through mia-train's real trainer.
#
# An arm's TOML holds the benchmark's settings (100 steps, no validation, no checkpoint, one node). This script
# changes two lines of it: dp_shard becomes the number of GPUs of the job, and the run's name says so. Code: the
# mia-train snapshot and the miao that lmd_ssl_v3's arms run. Prints seconds per step, the share of time spent
# waiting for data, MFU and peak GPU memory.
#
# Submit (here: three free GPUs of i04u22; 12 slots per GPU):
#   EXP=/nrs/scicompsoft/orhane/mia-train-experiments/attention_variants
#   bsub -P miaai -q gpu_b300 -m i04u22 -gpu "num=3" -n 36 -W 1:00 -J attnvar_throughput \
#        -o $EXP/jobs/throughput/throughput_%J.log \
#        bash /groups/scicompsoft/home/orhane/projects/mia-train-experiments/attention_variants/throughput.sh \
#        p384_block24 p384_block16 p384_block12 p384_block8
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
EXP=/nrs/scicompsoft/orhane/mia-train-experiments/attention_variants
OUT=$EXP/jobs/throughput            # the derived configs, the results and the job's log
RUNS=$EXP/runs/throughput           # the run directories of these short runs
MIA_TRAIN=${MIA_TRAIN:-/nrs/scicompsoft/orhane/probes/mia-train-pinned/8af6a8f}
MIAO=/nrs/scicompsoft/orhane/probes/miao-pinned/28229ac/src
PY=/groups/scicompsoft/home/orhane/myvenv/bin
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TORCHINDUCTOR_LAYOUT_OPTIMIZATION=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$MIA_TRAIN/src:$MIAO
JOB=${LSB_JOBID:-manual}
GPUS=$(tr ',' '\n' <<< "${CUDA_VISIBLE_DEVICES:-}" | grep -c .)

# Python and compile caches in node-local scratch, removed when the job ends.
SCR=/scratch/${USER:-$(id -un)}
mkdir -p "$SCR"
TMP=$(mktemp -d -p "$SCR" "attnvar_${JOB}.XXXXXX")
export PYTHONPYCACHEPREFIX=$TMP/pyc TORCHINDUCTOR_CACHE_DIR=$TMP/inductor
trap 'rm -rf "$TMP"' EXIT TERM
mkdir -p "$OUT/configs" "$OUT/results" "$RUNS"

echo "== $(hostname) $(date): $GPUS GPUs, arms: $*"
(( GPUS > 0 )) || { echo "FATAL: no GPU assigned"; exit 2; }
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
"$PY/python" -c "import models; assert models.__file__.startswith('$MIA_TRAIN/src/'), models.__file__; print('[mia-train]', '$MIA_TRAIN')" || exit 3
"$PY/python" -c "import miao; assert miao.__file__.startswith('$MIAO'), miao.__file__; print('[miao]', miao.__file__)" || exit 3
cd "$MIA_TRAIN" || exit 5

for ARM in "$@"; do
  SRC=$HERE/$ARM.toml
  [ -f "$SRC" ] || { echo "FATAL: no config $SRC"; exit 4; }
  TAG=${ARM}_${GPUS}gpu
  CFG=$OUT/configs/${TAG}_$JOB.toml
  sed -e "s/^experiment_name = .*/experiment_name = \"throughput_$TAG\"/" \
      -e "s/^dp_shard = [0-9]*/dp_shard = $GPUS/" "$SRC" > "$CFG"
  echo
  echo "===== $ARM $(date +%T): what differs from $ARM.toml"
  diff "$SRC" "$CFG" | grep '^[<>]' | cut -c1-160
  "$PY/torchrun" --standalone --nproc_per_node="$GPUS" "$HERE/throughput_probe.py" --config "$CFG" \
    --output-root "$RUNS" --out "$OUT/results/${TAG}_$JOB"
  echo "===== $ARM exit $? $(date +%T)"
done

echo
echo "===== table $(date +%T)"
"$PY/python" "$HERE/throughput_table.py" "$OUT/results" "$JOB"
echo "== done $(date)"
