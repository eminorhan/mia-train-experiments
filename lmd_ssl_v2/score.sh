#!/usr/bin/env bash
# Score one lmd_ssl_v2 arm on mia-evals' lmd_ssl_v2_neuron_instance, or with --fast on its 512^3-crop sibling
# lmd_ssl_v2_neuron_instance_crop512: predict every volume of both halves, one job per volume, then score them in
# jobs chained on the predictions. An affinity arm is scored through mws_blockwise_size_filter (full) or
# mws_size_filter (fast, exact); a promptable (SAM) arm through identity_size_filter, from predict_sam.sh's
# segment-everything labellings.
#
#   bash lmd_ssl_v2/score.sh 2a_affinity_unetr_lvd                  # its final checkpoint
#   bash lmd_ssl_v2/score.sh 2a_affinity_unetr_lvd --fast           # the crop512 eval, into eval/<arm>/step<N>/crop512/
#   bash lmd_ssl_v2/score.sh 2a_affinity_unetr_lvd --step 50000
#   bash lmd_ssl_v2/score.sh 2a_affinity_unetr_lvd --fast --steps 25000,50000,75000,100000   # the best of these
#   bash lmd_ssl_v2/score.sh 2a_affinity_unetr_lvd --predict-only   # or --score-only once the predictions exist
#   bash lmd_ssl_v2/score.sh 2a_affinity_unetr_lvd --dry-run
#   bash lmd_ssl_v2/score.sh 2f_affinity_unetr_simmim_lvd --fast --step 100000 --after <training job>   # chained
#   bash lmd_ssl_v2/score.sh 2h_... --fast --steps 25000,50000,75000,100000 --after <training job>  # unsaved ones wait
#   PREDICT_QUEUE=gpu_b300 bash lmd_ssl_v2/score.sh ...           # every prediction on one queue, not queue_for's
#
# --steps makes the checkpoint one more hyperparameter of the post-processing (sweep.py): every listed checkpoint
# is predicted on both halves and scored on the fit half under every candidate, one job each; the (checkpoint,
# candidate) pair that ranks first is chosen, and only that checkpoint is scored on the test half, into the one
# record. The choice and every checkpoint's fit scores: $EXP/eval/<arm>/best_step[.crop512].<route>.json.
#
# Data and scoring configs live in mia-evals under configs/lmd_ssl_v2_neuron_instance/, so prediction and scoring
# read the same volumes. `queue_for` only spreads the load: predictions agree across GPU types to bf16 noise
# (pq 0.0503 vs 0.0486 for one checkpoint, H100 vs B300, 2026-09-10), so no comparison depends on where they ran.
# The training miao pin; one job per volume, because the volumes differ ~2,700x in tiles (the zebrafish doublecube
# 2,744, FlyID49 one); each writes its <volume>.gt.zarr beside it, last, and a volume with one is not predicted again.
#
#   predictions   $EXP/eval/<arm>/step<N>[/crop512]/{fit,test}/<volume>.zarr (+ .gt.zarr)
#   scored labellings and scratch   /nrs/scicompsoft/orhane/mia-evals/<task>/{scored,scorescratch}/<record>
#   LSF logs      $EXP/jobs/
# The record is named <run>.step<N>.<route> by mia-evals; nothing here chooses a label.
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
# The scorer's memory is its slots x 15 GB on `local`: its 8 watershed processes hold one 512^3 block each (~40 GB),
# then the labelling, truth and metric of one volume (the zebrafish doublecube: 7.1 G voxels, ~28 GB per uint32 array).
SCORE_SLOTS=${SCORE_SLOTS:-63}
SCORE_WALL=${SCORE_WALL:-48:00}

PREDICT_QUEUE=${PREDICT_QUEUE:-}

DRY=0 PREDICT=1 SCORE=1 STEP="" STEPS="" SWEEP=0 FAST=0 AFTER=""
ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run) DRY=1 ;;
    --fast) FAST=1 ;;
    --after) AFTER=$2; shift ;;
    --predict-only) SCORE=0 ;;
    --score-only) PREDICT=0 ;;
    --step) STEP=$2; shift ;;
    --steps) STEPS=${2//,/ } SWEEP=1; shift ;;
    -*) echo "unknown flag $1" >&2; exit 2 ;;
    *) ARGS+=("$1") ;;
  esac
  shift
done
ARM=${ARGS[0]:?usage: score.sh <arm> [--fast] [--step N | --steps N,N,..] [--predict-only|--score-only] [--dry-run]}

run=$(ls -dt "$RUNS/lmd2__${ARM}_20"*/ 2>/dev/null | head -1 || true); run=${run%/}
[[ -n "$run" ]] || { echo "no run directory matching lmd2__${ARM}_20* under $RUNS" >&2; exit 1; }
if [[ $SWEEP -eq 1 ]]; then
  [[ -z "$STEP" ]] || { echo "--steps takes no --step" >&2; exit 2; }
elif [[ -n "$AFTER" ]]; then                   # the checkpoint appears when job $AFTER finishes
  [[ -n "$STEP" ]] || { echo "--after needs --step: the checkpoint does not exist yet" >&2; exit 2; }
  STEPS=$STEP
else
  [[ -n "$STEP" ]] || STEP=$(ls -d "$run"/checkpoints/step_* | sed 's|.*step_||' | sort -n | tail -1)
  STEPS=$STEP
fi
if [[ -z "$AFTER" ]]; then
  for step in $STEPS; do
    [[ -d "$run/checkpoints/step_$step" ]] || { echo "no checkpoint step_$step in $run" >&2; exit 1; }
  done
fi
algorithm=$(awk '/^\[algorithm\]/ {f = 1; next} /^\[/ {f = 0} f && $1 == "name" {gsub(/"/, "", $3); print $3}' "$run/config.toml")
[[ $algorithm == affinity_seg || $algorithm == promptable_seg ]] || {
  echo "$run is $algorithm; this script scores affinity_seg and promptable_seg arms" >&2; exit 2; }
RUN_NAME=$(basename "$run")
if [[ $FAST -eq 1 ]]; then
  TASK=lmd_ssl_v2_neuron_instance_crop512 ROUTE=mws_size_filter SUB=/crop512
else
  TASK=lmd_ssl_v2_neuron_instance ROUTE=mws_blockwise_size_filter SUB=""
fi
[[ $algorithm == promptable_seg ]] && ROUTE=identity_size_filter
TASK_DIR=/nrs/scicompsoft/orhane/mia-evals/$TASK
art_for () { echo "$EXP/eval/$ARM/step$1$SUB"; }       # one checkpoint's predictions: fit/ and test/
if [[ $SWEEP -eq 1 ]]; then echo "run $run, the best of steps $STEPS -> record $RUN_NAME.step<best>.$ROUTE"
else echo "run $run, step $STEP -> record $RUN_NAME.step$STEP.$ROUTE"; fi
mkdir -p "$LOGS" "$TASK_DIR/scored" "$TASK_DIR/scorescratch"

jobid () { grep -o 'Job <[0-9]*>' | head -1 | tr -dc 0-9; }
submit () {      # submit <name> <bsub args...> -- <command>   -> job id or DRYRUN
  local name=$1; shift; local args=(); while [[ $1 != -- ]]; do args+=("$1"); shift; done; shift
  if [[ $DRY -eq 1 ]]; then printf 'bsub -P %s %s -J %s ...\n    %s\n' "$PROJECT" "${args[*]}" "$name" "$1" >&2; echo DRYRUN
  else bsub -P "$PROJECT" "${args[@]}" -J "$name" -cwd "$MIA_TRAIN" -o "$LOGS/${name}_%J.log" -e "$LOGS/${name}_%J.err" "$1" | jobid; fi
}
waits () {       # waits <job id>...   -> bsub's -w condition on them all, or nothing
  local ids=() id
  for id in "$@"; do [[ $id == DRYRUN ]] || ids+=("$id"); done
  if [[ ${#ids[@]} -gt 0 ]]; then printf 'done(%s) && ' "${ids[@]}" | sed 's/ && $//'; fi
}
queue_for () {   # queue_for <volume> <split>: a default split of the load between the GPU types
  [[ $FAST -eq 1 && $2 == test ]] && { echo gpu_b300; return; }
  case $1 in
    zebrafish_fish2_doublecube1*) echo gpu_b300 ;;                                   # 36% of the tiles
    *) echo gpu_h200 ;;
  esac
}
wall_for () {    # by tile count (probes/eval_sizes/sizes.json); the blend accumulator, ~40 B/voxel, fits 12 slots
  case $1 in
    *_crop512) echo 1:00 ;;                                                          # 45 tiles at most
    zebrafish_fish2_doublecube1) echo 6:00 ;;                                        # 2,744 tiles
    zebrafish_fish2_quadcube1|cns_mito_crop182|cns_mito_crop164) echo 4:00 ;;        # 832-1,575
    hemibrain_ellipsoid_body|liconn_expid82) echo 2:00 ;;                            # 343 / 320
    *) echo 1:00 ;;                                                                  # 27 or fewer
  esac
}
sam_wall_for () {  # segment-everything: 17-32 s per tile on an H200 at patch 8 (probes/sam_eval_timing), doubled
  case $1 in
    *_crop512) echo 2:00 ;;                                                          # 45 or fewer
    zebrafish_fish2_doublecube1) echo 72:00 ;;                                       # 2,744 tiles
    zebrafish_fish2_quadcube1|cns_mito_crop182|cns_mito_crop164) echo 36:00 ;;       # 832-1,575
    hemibrain_ellipsoid_body|liconn_expid82) echo 8:00 ;;                            # 343 / 320
    *) echo 2:00 ;;                                                                  # 27 or fewer
  esac
}

declare -A predicted=()                        # "<step> <split>" -> the prediction jobs submitted for it
if [[ $PREDICT -eq 1 ]]; then
  for step in $STEPS; do
    art=$(art_for "$step"); mkdir -p "$art"
    for split in fit test; do
      config=$EVALS/configs/$TASK/data/$split.yaml
      for volume in $(sed -n 's/^- name: //p' "$config"); do
        if grep -qs '"background_id"' "$art/$split/$volume.gt.zarr/s0/zarr.json"; then
          printf '  predict %-6s %-5s %-36s exists\n' "$step" "$split" "$volume"; continue
        fi
        if [[ $algorithm == promptable_seg ]]; then
          predict="bash '$HERE/predict_sam.sh' '$run' $step '$config' $volume '$art/$split'"; wall=$(sam_wall_for "$volume")
        else
          predict="cd '$MIA_TRAIN' && $VENV/bin/python src/predict.py '$run' --step $step --data-config '$config' \
--volume $volume --out '$art/$split'"; wall=$(wall_for "$volume")
        fi
        cmd="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=$MIAO_SRC:$MIA_TRAIN/src; $VENV/bin/python -c \"import miao; assert miao.__file__.startswith('$MIAO_SRC'), miao.__file__\" \
&& $predict"
        args=(-q "${PREDICT_QUEUE:-$(queue_for "$volume" "$split")}" -gpu "num=1" -n 12 -W "$wall")
        [[ -n "$AFTER" && ! -d "$run/checkpoints/step_$step" ]] && args+=(-w "done($AFTER)")   # not saved yet
        id=$(submit "lmd2_p_${ARM}_${step}_$volume" "${args[@]}" -- "$cmd")
        printf '  predict %-6s %-5s %-36s job %s\n' "$step" "$split" "$volume" "$id"
        predicted["$step $split"]+=" $id"
      done
    done
  done
fi

SCORE_ENV="export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8; cd '$EVALS' &&"
NUMA="'$VENV/bin/python' '$HERE/../gary_comparison/numa_local.py'"
scorer () {      # scorer <step>: mia-evals score on that checkpoint; '{step}' leaves the step to sweep.py best
  local art rec=$RUN_NAME.step$1.$ROUTE
  art=$(art_for "$1")
  echo "$NUMA '$MIA_EVALS' score configs/$TASK/$ROUTE.toml --test '$art/test' --val '$art/fit' --run-dir '$run' \
--scored-out '$TASK_DIR/scored/$rec' --scratch '$TASK_DIR/scorescratch/$rec'"
}
if [[ $SCORE -eq 1 && $SWEEP -eq 0 ]]; then
  args=(-q local -n "$SCORE_SLOTS" -W "$SCORE_WALL")
  w=$(waits ${predicted["$STEP fit"]:-} ${predicted["$STEP test"]:-}); [[ -z "$w" ]] || args+=(-w "$w")
  id=$(submit "lmd2_sc_${ARM}_${STEP}_$ROUTE" "${args[@]}" -- "$SCORE_ENV $(scorer "$STEP")")
  echo "  score   $ROUTE on $SCORE_SLOTS slots: job $id -> record $RUN_NAME.step$STEP.$ROUTE"
fi
if [[ $SCORE -eq 1 && $SWEEP -eq 1 ]]; then
  fits=() before_best=()
  for step in $STEPS; do
    art=$(art_for "$step")
    args=(-q local -n "$SCORE_SLOTS" -W "$SCORE_WALL")
    w=$(waits ${predicted["$step fit"]:-}); [[ -z "$w" ]] || args+=(-w "$w")
    cmd="$SCORE_ENV $NUMA '$VENV/bin/python' '$HERE/sweep.py' fit configs/$TASK/$ROUTE.toml --val '$art/fit' \
--scratch '$TASK_DIR/scorescratch/$RUN_NAME.step$step.$ROUTE' --out '$art/fit_scores.$ROUTE.json'"
    id=$(submit "lmd2_fit_${ARM}_${step}_$ROUTE" "${args[@]}" -- "$cmd")
    echo "  fit     $step on $SCORE_SLOTS slots: job $id -> $art/fit_scores.$ROUTE.json"
    fits+=("'$art/fit_scores.$ROUTE.json'")
    before_best+=("$id" ${predicted["$step test"]:-})
  done
  best=$EXP/eval/$ARM/best_step${SUB/\//.}.$ROUTE.json
  args=(-q local -n "$SCORE_SLOTS" -W "$SCORE_WALL")
  w=$(waits "${before_best[@]}"); [[ -z "$w" ]] || args+=(-w "$w")
  cmd="$SCORE_ENV '$VENV/bin/python' '$HERE/sweep.py' best ${fits[*]} --out '$best' -- $(scorer '{step}')"
  id=$(submit "lmd2_sc_${ARM}_best_$ROUTE" "${args[@]}" -- "$cmd")
  echo "  score   the best of steps $STEPS on $SCORE_SLOTS slots: job $id -> record $RUN_NAME.step<best>.$ROUTE"
fi
