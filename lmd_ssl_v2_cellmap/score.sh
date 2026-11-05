#!/usr/bin/env bash
# Predict and score lmd_ssl_v2_cellmap arms on mia-evals' cellmap_organelle_semantic, or with --fewshot on
# cellmap_organelle_semantic_fewshot: one prediction job per checkpoint, covering every test crop (the checkpoint loads
# once), then a CPU scoring job chained on it. The route is identity: predict.py --argmax writes uint8 CellMap ids,
# which mia-evals scores as written. Nothing is fitted, so the fit split is not predicted.
#
#   bash lmd_ssl_v2_cellmap/score.sh 1e_affinity_simmim_lvd                       # its final checkpoint
#   bash lmd_ssl_v2_cellmap/score.sh 1e_affinity_simmim_lvd --steps 100000,75000,50000,25000
#   bash lmd_ssl_v2_cellmap/score.sh 2c_affinity_simmim_lvd_seed1 --steps 100000 --after <training job>
#   bash lmd_ssl_v2_cellmap/score.sh 1e_affinity_simmim_lvd --fewshot --steps 25000
#   bash lmd_ssl_v2_cellmap/score.sh ... --predict-only | --score-only | --dry-run
#   PREDICT_QUEUE=gpu_h100 PREDICT_WALL=12:00 bash lmd_ssl_v2_cellmap/score.sh ...
#   SCORE_AFTER="<job> ..." bash lmd_ssl_v2_cellmap/score.sh ... --score-only   # the scorer also waits for these jobs
#   BOARD=$EXP/eval/board bash lmd_ssl_v2_cellmap/score.sh ...   # records into that board, not mia-evals' leaderboard
#
# With BOARD set, the records and the task's table go to that leaderboard root instead (mia-evals' --leaderboard), no
# scored copies are kept (they only feed the views, whose keys stay with mia-evals' leaderboard) and scratch goes under
# $EXP/eval/scratch/; render the board once after the last scorer: mia-evals leaderboard --leaderboard $BOARD.
#
# Every listed checkpoint is scored on the test crops into mia-evals' leaderboard, each its own record
# <run>.step<N>.identity: the fit split is these models' own training crops, so it cannot choose a checkpoint. The run's
# training resolution (8 nm, or native for 2a) picks data/predict_<task>_<res>.yaml, the boxes make_configs.py wrote.
# Walls are twice the 10-26 pipeline check's ~18 s per tile on an H200.
#
#   predictions   $EXP/eval/<arm>/step<N>/test/<volume>.zarr, and .done once predict.py has written them all
#   scored labellings and scratch   /nrs/scicompsoft/orhane/mia-evals/<task>/{scored,scorescratch}/<record>
#   job scripts   $EXP/jobs/eval/<time>_<arm>/, their logs beside them
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MIA_TRAIN=/groups/scicompsoft/home/orhane/projects/mia-train
EVALS=/groups/scicompsoft/home/orhane/projects/mia-evals
VENV=/groups/scicompsoft/home/orhane/myvenv
PROJECT=miaai
EXP=/nrs/scicompsoft/orhane/mia-train-experiments/lmd_ssl_v2_cellmap
MIAO_SRC=${MIAO_SRC:-/nrs/scicompsoft/orhane/probes/miao-pinned/4ef0d8c/src}   # our fork's master (series b32 trained on it)
PREDICT_QUEUE=${PREDICT_QUEUE:-gpu_h200}
SCORE_AFTER=${SCORE_AFTER:-}
BOARD=${BOARD:-}                # a leaderboard root of its own; empty = mia-evals' leaderboard

DRY=0 PREDICT=1 SCORE=1 STEPS="" AFTER="" FEWSHOT=0
ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run) DRY=1 ;;
    --fewshot) FEWSHOT=1 ;;
    --after) AFTER=$2; shift ;;
    --predict-only) SCORE=0 ;;
    --score-only) PREDICT=0 ;;
    --steps) STEPS=${2//,/ }; shift ;;
    -*) echo "unknown flag $1" >&2; exit 2 ;;
    *) ARGS+=("$1") ;;
  esac
  shift
done
if [[ $FEWSHOT -eq 1 ]]; then SIZE=fewshot TASK=cellmap_organelle_semantic_fewshot SUFFIX=_fewshot
else SIZE=large TASK=cellmap_organelle_semantic SUFFIX=""; fi
STAMP=$(date +%Y%m%d_%H%M%S)

submit () {      # submit <name> <script> <bsub args...>   -> job id, or DRYRUN
  local name=$1 script=$2; shift 2
  if [[ $DRY -eq 1 ]]; then echo "    bsub -P $PROJECT $* -J $name bash $script" >&2; echo DRYRUN; return; fi
  bsub -P "$PROJECT" "$@" -J "$name" -o "${script%.sh}_%J.log" -e "${script%.sh}_%J.err" bash "$script" \
    | grep -o 'Job <[0-9]*>' | tr -dc 0-9
}

ARM=${ARGS[0]:?usage: score.sh <arm> [--fewshot] [--steps N,N,..] [--after <job>] [--predict-only|--score-only] [--dry-run]}
run=$(ls -dt "$EXP/runs/lmd2cm__${ARM}${SUFFIX}_20"*/ 2>/dev/null | head -1 || true); run=${run%/}
[[ -n "$run" ]] || { echo "no run directory lmd2cm__${ARM}${SUFFIX}_20* under $EXP/runs" >&2; exit 1; }
RES=$(grep -o "train_${SIZE}_\(8nm\|native\)\.yaml" "$run/config.toml" | head -1 | sed 's/.*_\(8nm\|native\)\.yaml/\1/')
[[ -n "$RES" ]] || { echo "$run/config.toml names no data/train_${SIZE}_<res>.yaml" >&2; exit 1; }
DATA=$HERE/data/predict_${SIZE}_${RES}.yaml
[[ -n "$STEPS" ]] || STEPS=$(ls -d "$run"/checkpoints/step_* | sed 's|.*step_||' | sort -n | tail -1)
for step in $STEPS; do
  [[ -d "$run/checkpoints/step_$step" || -n "$AFTER" ]] || { echo "no checkpoint step_$step in $run" >&2; exit 1; }
done
case $SIZE/$RES in              # 762, 1,613, 3,273 and 6,936 tiles at ~18 s, doubled
  large/8nm) WALL=8:00 ;;
  large/native) WALL=16:00 ;;
  fewshot/8nm) WALL=36:00 ;;
  fewshot/native) WALL=72:00 ;;
esac
WALL=${PREDICT_WALL:-$WALL}     # e.g. more on a queue whose hosts are slower
RUN_NAME=$(basename "$run")
TASK_DIR=/nrs/scicompsoft/orhane/mia-evals/$TASK
JOBS=$EXP/jobs/eval/${STAMP}_$ARM$SUFFIX
[[ $DRY -eq 1 ]] && JOBS=$(mktemp -d)
mkdir -p "$JOBS"
if [[ $DRY -eq 0 && -n "$BOARD" ]]; then mkdir -p "$BOARD" "$EXP/eval/scratch"
elif [[ $DRY -eq 0 ]]; then mkdir -p "$TASK_DIR/scored" "$TASK_DIR/scorescratch"; fi
echo "run $run ($RES), steps $STEPS"

for step in $STEPS; do
  art=$EXP/eval/$ARM$SUFFIX/step$step/test
  rec=$RUN_NAME.step$step.identity
  pid=""
  if [[ $PREDICT -eq 1 && -e "$art/.done" ]]; then
    echo "  predict $step: done already ($art)"
  elif [[ $PREDICT -eq 1 ]]; then
    cat > "$JOBS/predict_$step.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCHINDUCTOR_LAYOUT_OPTIMIZATION=0
export PYTHONPATH=$MIAO_SRC
export TMPDIR=/scratch/\$USER/lmd2cm_\$LSB_JOBID PYTHONPYCACHEPREFIX=/scratch/\$USER/lmd2cm_\$LSB_JOBID/pyc
mkdir -p "\$TMPDIR"
trap 'rm -rf "\$TMPDIR"' EXIT
$VENV/bin/python -c "import miao; assert miao.__file__.startswith('$MIAO_SRC'), miao.__file__; print('[miao]', miao.__file__)"
cd $MIA_TRAIN
echo "[mia-train] \$(git rev-parse --short HEAD), uncommitted: \$(git status --short --untracked-files=no | tr '\n' ' ')"
mkdir -p $art
$VENV/bin/python src/predict.py $run --step $step --data-config $DATA --out $art --argmax --no-truth --cover-box
touch $art/.done
EOF
    args=(-q "$PREDICT_QUEUE" -gpu "num=1" -n 12 -W "$WALL")
    [[ -n "$AFTER" && ! -d "$run/checkpoints/step_$step" ]] && args+=(-w "done($AFTER)")   # not saved yet
    pid=$(submit "lmd2cm_p_${ARM}${SUFFIX}_$step" "$JOBS/predict_$step.sh" "${args[@]}")
    echo "  predict $step on $PREDICT_QUEUE, wall $WALL: job $pid -> $art"
  fi
  if [[ $SCORE -eq 1 ]]; then
    if [[ -n "$BOARD" ]]; then dest="--leaderboard $BOARD --no-scored --scratch $EXP/eval/scratch/$rec"
    else dest="--scored-out $TASK_DIR/scored/$rec --scratch $TASK_DIR/scorescratch/$rec"; fi
    cat > "$JOBS/score_$step.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
test -e $art/.done
cd $EVALS
$VENV/bin/mia-evals score configs/$TASK/identity.toml --test $art --run-dir $run \\
  $dest
EOF
    args=(-n 8 -W 1:00)
    deps=$SCORE_AFTER; [[ -n "$pid" && $pid != DRYRUN ]] && deps="$deps $pid"
    [[ -z "${deps// /}" ]] || args+=(-w "$(printf 'done(%s) && ' $deps | sed 's/ && $//')")
    sid=$(submit "lmd2cm_s_${ARM}${SUFFIX}_$step" "$JOBS/score_$step.sh" "${args[@]}")
    echo "  score   $step: job $sid -> record $rec"
  fi
done
