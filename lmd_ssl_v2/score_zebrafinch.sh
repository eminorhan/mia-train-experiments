#!/usr/bin/env bash
# Score one lmd_ssl_v2 arm's step-100000 checkpoint on mia-evals' zebrafinch_neurite_tracing_11um, on one grid:
#
#   bash lmd_ssl_v2/score_zebrafinch.sh <arm> native|8nm [--stages predict,convert,score] [--after <job id>]... [--dry-run]
#
#   native  predict.py on the store's own 20 x 9 x 9 nm grid, data/zebrafinch_11um_native.yaml, with
#           --allow-resolution-mismatch (the arms trained at 8 nm). An affinity arm is scored by the task's
#           mws_blockwise.toml: one block over the cube, so exact mutex watershed, the neuropil mask inside it. A
#           promptable arm's segment-everything labelling goes through zf_convert.py and identity.toml.
#   8nm     predict.py at the arms' 8 nm, data/zebrafinch_11um_8nm.yaml. zf_convert.py makes the labelling on the
#           store's grid (an affinity arm: exact mutex watershed on the 8 nm lattice first); identity_8nm.toml scores it.
#
# Step 100000 for every arm (the user's choice, 2026-10-27). Every route has one candidate, so nothing is fitted and no
# --val is passed. --stages picks which of the three jobs to submit, each chained on the one before it here; --after
# (repeatable) makes the first of them also wait for other jobs. Predictions go to PREDICT_QUEUE (default gpu_h200).
#
#   predictions   $EXP/eval/<arm>/step100000/zebrafinch_11um/<grid>/{pred,test}/zebrafinch_11um_test.zarr
#                 (test/ is what the scorer reads: predict.py's artifact for a native affinity arm, else zf_convert.py's)
#   scored labellings and scratch   /nrs/scicompsoft/orhane/mia-evals/zebrafinch_neurite_tracing_11um/{scored,scorescratch}/<record>
#   LSF logs      $EXP/jobs/lmd2zf_*
# Records: <run>.step100000.mws_blockwise (native, affinity), .identity (native, promptable), .identity_8nm (8 nm).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MIA_TRAIN=/groups/scicompsoft/home/orhane/projects/mia-train
EVALS=/groups/scicompsoft/home/orhane/projects/mia-evals
VENV=/groups/scicompsoft/home/orhane/myvenv
MIA_EVALS=$VENV/bin/mia-evals
PROJECT=miaai
EXP=/nrs/scicompsoft/orhane/mia-train-experiments/lmd_ssl_v2
RUNS=$EXP/runs
LOGS=$EXP/jobs
MIAO_SRC=${MIAO_SRC:-/nrs/scicompsoft/orhane/probes/miao-pinned/62b3746/src}
TASK=zebrafinch_neurite_tracing_11um
TASK_DIR=/nrs/scicompsoft/orhane/mia-evals/$TASK
VOLUME=zebrafinch_11um_test
STEP=100000
PREDICT_QUEUE=${PREDICT_QUEUE:-gpu_h200}

DRY=0 STAGES=predict,convert,score
AFTER=() ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run) DRY=1 ;;
    --stages) STAGES=$2; shift ;;
    --after) AFTER+=("$2"); shift ;;
    -*) echo "unknown flag $1" >&2; exit 2 ;;
    *) ARGS+=("$1") ;;
  esac
  shift
done
USAGE="usage: score_zebrafinch.sh <arm> native|8nm [--stages predict,convert,score] [--after <job id>]... [--dry-run]"
ARM=${ARGS[0]:?$USAGE}
GRID=${ARGS[1]:?$USAGE}
[[ $GRID == native || $GRID == 8nm ]] || { echo "$USAGE" >&2; exit 2; }
run=$(ls -dt "$RUNS/lmd2__${ARM}_20"*/ 2>/dev/null | head -1 || true); run=${run%/}
[[ -n "$run" && -d "$run/checkpoints/step_$STEP" ]] || { echo "no checkpoint step_$STEP for $ARM under $RUNS" >&2; exit 1; }
RUN_NAME=$(basename "$run")
algorithm=$(awk '/^\[algorithm\]/ {f = 1; next} /^\[/ {f = 0} f && $1 == "name" {gsub(/"/, "", $3); print $3}' "$run/config.toml")

DATA=$HERE/data/zebrafinch_11um_$GRID.yaml
ART=$EXP/eval/$ARM/step$STEP/zebrafinch_11um/$GRID
flags="--no-truth"; [[ $GRID == native ]] && flags+=" --allow-resolution-mismatch"
if [[ $GRID == native && $algorithm != promptable_seg ]]; then
  PRED=$ART/test CONVERT=0 ROUTE=mws_blockwise
else
  PRED=$ART/pred CONVERT=1 ROUTE=identity
  [[ $GRID == 8nm ]] && ROUTE=identity_8nm
fi
REC=$RUN_NAME.step$STEP.$ROUTE
echo "$ARM ($algorithm), $GRID: run $RUN_NAME -> record $REC"
mkdir -p "$LOGS" "$ART" "$TASK_DIR/scored" "$TASK_DIR/scorescratch"

jobid () { grep -o 'Job <[0-9]*>' | head -1 | tr -dc 0-9; }
submit () {      # submit <name> <bsub args...> -- <command>   -> job id or DRYRUN
  local name=$1; shift; local args=(); while [[ $1 != -- ]]; do args+=("$1"); shift; done; shift
  if [[ $DRY -eq 1 ]]; then printf 'bsub -P %s %s -J %s ...\n    %s\n' "$PROJECT" "${args[*]}" "$name" "$1" >&2; echo DRYRUN
  else bsub -P "$PROJECT" "${args[@]}" -J "$name" -o "$LOGS/${name}_%J.log" -e "$LOGS/${name}_%J.err" "$1" | jobid; fi
}
waits=("${AFTER[@]}")
chain () {       # the -w arguments for the next job, one per line: the previous job here, or --after for the first
  local id cond=""
  for id in "${waits[@]}"; do [[ $id == DRYRUN ]] || cond+="${cond:+ && }done($id)"; done
  [[ -n $cond ]] && printf '%s\n%s\n' -w "$cond"
  return 0
}
# A job-local pycache: an edited module on NFS must not meet a stale .pyc on another host.
PYENV="export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONPYCACHEPREFIX=\${TMPDIR:-/tmp}/pyc_\$LSB_JOBID;"
NUMA="'$VENV/bin/python' '$HERE/../gary_comparison/numa_local.py'"

if [[ ,$STAGES, == *,predict,* ]]; then
  [[ -e $PRED/$VOLUME.zarr ]] && { echo "$PRED/$VOLUME.zarr exists; delete it to predict again" >&2; exit 1; }
  if [[ $algorithm == promptable_seg ]]; then
    predict="bash '$HERE/predict_sam.sh' '$run' $STEP '$DATA' $VOLUME '$PRED' $flags"
    wall=8:00; [[ $GRID == 8nm ]] && wall=16:00             # ~12 s a tile on an H200: 1,008 / 2,197 tiles
  else
    predict="cd '$MIA_TRAIN' && $VENV/bin/python src/predict.py '$run' --step $STEP --data-config '$DATA' \
--volume $VOLUME --out '$PRED' $flags"
    wall=3:00; [[ $GRID == 8nm ]] && wall=6:00              # 1.4-2.1 h at 8 nm; 2f hit 3:00 on a shared node
  fi
  cmd="$PYENV export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$MIAO_SRC:$MIA_TRAIN/src; \
echo \"[mia-train] \$(git -C '$MIA_TRAIN' rev-parse HEAD)\"; \
$VENV/bin/python -c \"import miao; assert miao.__file__.startswith('$MIAO_SRC'), miao.__file__\" && $predict"
  readarray -t w < <(chain)
  id=$(submit "lmd2zf_p_${ARM}_$GRID" -q "$PREDICT_QUEUE" -gpu "num=1" -n 12 -W "$wall" "${w[@]}" -- "$cmd")
  echo "  predict  job $id -> $PRED/$VOLUME.zarr"
  waits=("$id")
fi

if [[ ,$STAGES, == *,convert,* && $CONVERT -eq 1 ]]; then
  [[ -e $ART/test/$VOLUME.zarr ]] && { echo "$ART/test/$VOLUME.zarr exists; delete it to convert again" >&2; exit 1; }
  if [[ $algorithm == promptable_seg ]]; then
    args=(-q local -n 16 -W 2:00)                                  # a labelling: gather, mask, relabel
  else
    args=(-q local -n 96 -R "span[hosts=1]" -W 12:00)              # exact mws on 2.4 G lattice voxels: 2a 796 GB
  fi
  cmd="$PYENV mkdir -p '$ART/test' && $NUMA '$VENV/bin/python' '$HERE/zf_convert.py' '$PRED/$VOLUME.zarr' '$ART/test/$VOLUME.zarr'"
  readarray -t w < <(chain)
  id=$(submit "lmd2zf_c_${ARM}_$GRID" "${args[@]}" "${w[@]}" -- "$cmd")
  echo "  convert  job $id -> $ART/test/$VOLUME.zarr"
  waits=("$id")
fi

if [[ ,$STAGES, == *,score,* ]]; then
  if [[ $ROUTE == mws_blockwise ]]; then
    args=(-q local -n 63 -W 8:00)        # exact mws on the 0.78 G-voxel cube, ~0.3-0.45 TB; lmd_ssl_v2's 63 slots
  else
    args=(-q local -n 8 -W 2:00)         # node lookups only
  fi
  cmd="$PYENV cd '$EVALS' && $NUMA '$MIA_EVALS' score configs/$TASK/$ROUTE.toml --test '$ART/test' --run-dir '$run' \
--scored-out '$TASK_DIR/scored/$REC' --scratch '$TASK_DIR/scorescratch/$REC'"
  readarray -t w < <(chain)
  id=$(submit "lmd2zf_s_${ARM}_$GRID" "${args[@]}" "${w[@]}" -- "$cmd")
  echo "  score    job $id -> record $REC"
fi
