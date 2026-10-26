#!/usr/bin/env bash
# lmd_ssl_v2_cellmap: submit one run to gpu_b300 (one full node), optionally gated on a 20-step smoke run.
#
#   bash lmd_ssl_v2_cellmap/submit.sh 1e_affinity_simmim_lvd             # smoke, then the real run on `done(smoke)`
#   bash lmd_ssl_v2_cellmap/submit.sh --dry-run 1e_affinity_simmim_lvd   # write the job scripts, print the bsub lines
#   bash lmd_ssl_v2_cellmap/submit.sh --smoke 1e_affinity_simmim_lvd     # smoke only
#   bash lmd_ssl_v2_cellmap/submit.sh --no-smoke 1e_affinity_simmim_lvd  # real run only; a resubmission after a
#                                                                        # wall-time kill continues via --resume
#
# lmd_ssl_v2/submit.sh with this experiment's paths, job names and wall times. Everything it writes lands in this
# experiment's directory on /nrs: jobs/ (LSF logs and the generated job scripts) and runs/ (run directories via
# --output-root); smoke artifacts go under jobs/smoke/ and runs/smoke/. If the smoke fails, the chained real job
# stays PEND forever: `bkill` it.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MIA_TRAIN=/groups/scicompsoft/home/orhane/projects/mia-train
VENV=/groups/scicompsoft/home/orhane/myvenv
PROJECT=miaai
EXP=/nrs/scicompsoft/orhane/mia-train-experiments/lmd_ssl_v2_cellmap   # this experiment's home on /nrs
RUNS=$EXP/runs
LOGS=$EXP/jobs                      # LSF logs AND the generated job scripts
SMOKE_LOGS=$LOGS/smoke              # the smoke's config, script and logs
SMOKE_RUNS=$RUNS/smoke              # the smoke's run directories
QUEUE=${QUEUE:-gpu_b300}
GPUS=${GPUS:-8}                     # the real run's GPUs: must equal the config's dp_replicate x dp_shard x tp
SLOTS=$((12 * GPUS))                # 12 slots/GPU on gpu_b300
SMOKE_GPUS=${SMOKE_GPUS:-1}         # the smoke's GPUs; its derived config gets dp_shard = SMOKE_GPUS
# The miao the job imports: a read-only copy of one commit of our fork (github.com/eminorhan/miao), first on
# PYTHONPATH; the job stops unless it is the miao imported. 62b3746 = v0.4.2 + label_fill (which these configs
# need) + nearest-exact label resampling + label_margin.
MIAO_SRC=${MIAO_SRC:-/nrs/scicompsoft/orhane/probes/miao-pinned/62b3746/src}
THREADS="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4"
# inductor's layout optimisation can hand cuDNN SDPA a query with a non-unit last stride under compile, killing the
# run at step 1; every compiled run here sets this.
COMPILE_ENV="export TORCHINDUCTOR_LAYOUT_OPTIMIZATION=0"

DRY=0 SMOKE=1 REAL=1
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run) DRY=1 ;;
    --smoke) REAL=0 ;;
    --no-smoke) SMOKE=0 ;;
    -*) echo "unknown flag $1" >&2; exit 2 ;;
    *) NAME=$1 ;;
  esac
  shift
done
: "${NAME:?usage: submit.sh [--dry-run] [--smoke|--no-smoke] <config name without .toml>}"
CONFIG="$HERE/$NAME.toml"
[[ -f $CONFIG ]] || { echo "no such config: $CONFIG" >&2; exit 2; }
# 100k steps at ~0.6 s/step is ~17 h, 25k (fewshot) ~4 h; both walls tolerate a 2.5x slower step. `-r` plus
# `--resume` turns a node failure into a requeue from the last checkpoint.
if [[ $NAME == *_fewshot ]]; then WALL=${WALL:-12:00}; else WALL=${WALL:-48:00}; fi
mkdir -p "$RUNS" "$LOGS" "$SMOKE_RUNS" "$SMOKE_LOGS"

jobid () { grep -o 'Job <[0-9]*>' | head -1 | tr -dc 0-9; }

# write_cmd <dir> <tag> <config> <nproc> <tail args...>: the script bsub runs, written into <dir>
write_cmd () {
  local dir=$1 tag=$2 cfg=$3 procs=$4; shift 4
  local cmd="$dir/$tag.sh"
  { echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    echo "$THREADS"
    echo "$COMPILE_ENV"
    echo "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    if [[ -n "$MIAO_SRC" ]]; then
      echo "export PYTHONPATH=$MIAO_SRC\${PYTHONPATH:+:\$PYTHONPATH}"
      echo "$VENV/bin/python -c \"import miao; assert miao.__file__.startswith('$MIAO_SRC'), miao.__file__; print('[miao]', miao.__file__)\""
    fi
    printf 'cd %q\n' "$MIA_TRAIN"
    echo 'echo "[mia-train] $(git rev-parse --short HEAD), uncommitted: $(git status --short --untracked-files=no | tr "\n" " ")"'
    printf '%s --standalone --nproc_per_node=%s src/train.py --config %q %s\n' \
      "$VENV/bin/torchrun" "$procs" "$cfg" "$*"
  } > "$cmd"
  chmod +x "$cmd"
  echo "$cmd"
}

# submit <logdir> <tag> <cmd> <nproc> <slots> <wall> [dependency job id]  -> job id (or DRYRUN)
submit () {
  local logdir=$1 tag=$2 cmd=$3 procs=$4 slots=$5 wall=$6 dep=${7:-}
  local args=(-P "$PROJECT" -q "$QUEUE" -gpu "num=$procs" -n "$slots" -W "$wall" -r
              -J "lmd2cm_$tag" -cwd "$MIA_TRAIN" -o "$logdir/lmd2cm_${tag}_%J.log" -e "$logdir/lmd2cm_${tag}_%J.err")
  [[ -n "$dep" && "$dep" != DRYRUN ]] && args+=(-w "done($dep)")
  if [[ $DRY -eq 1 ]]; then
    printf 'bsub %s bash %q\n' "${args[*]}" "$cmd" >&2; echo DRYRUN
  else
    bsub "${args[@]}" "bash '$cmd'" | jobid
  fi
}

SMOKE_ID=""
if [[ $SMOKE -eq 1 ]]; then
  # 20 steps on SMOKE_GPUS GPUs (default 1), straight from the real config, so the smoke exercises what actually
  # runs; SMOKE_GPUS=GPUS also exercises the real run's sharding and checkpointing. val_every stays 0: these runs
  # have no [val_data].
  smoke_cfg="$SMOKE_LOGS/smoke_$NAME.toml"
  sed -e "s/^experiment_name = .*/experiment_name = \"smoke_lmd2cm__$NAME\"/" \
      -e 's/^max_steps = .*/max_steps = 20/'     -e 's/^warmup_steps = .*/warmup_steps = 2/' \
      -e 's/^checkpoint_every = .*/checkpoint_every = 20/' \
      -e 's/^samples_per_epoch = .*/samples_per_epoch = 20/' -e "s/^dp_shard = .*/dp_shard = $SMOKE_GPUS/" \
      -e 's/^num_workers = .*/num_workers = 2/'  -e 's/^log_every = .*/log_every = 5/' \
      "$CONFIG" > "$smoke_cfg"
  cmd=$(write_cmd "$SMOKE_LOGS" "smoke_$NAME" "$smoke_cfg" "$SMOKE_GPUS" --output-root "$SMOKE_RUNS")
  SMOKE_ID=$(submit "$SMOKE_LOGS" "smoke_$NAME" "$cmd" "$SMOKE_GPUS" $((12 * SMOKE_GPUS)) 2:00)
  echo "smoke  $NAME: job $SMOKE_ID  ($cmd)"
fi
if [[ $REAL -eq 1 ]]; then
  cmd=$(write_cmd "$LOGS" "$NAME" "$CONFIG" "$GPUS" --output-root "$RUNS" --resume)
  REAL_ID=$(submit "$LOGS" "$NAME" "$cmd" "$GPUS" "$SLOTS" "$WALL" "$SMOKE_ID")
  echo "real   $NAME: job $REAL_ID  ($cmd)${SMOKE_ID:+, starts after done($SMOKE_ID)}"
fi
