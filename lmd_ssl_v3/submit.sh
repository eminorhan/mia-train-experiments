#!/usr/bin/env bash
# lmd_ssl_v3: submit one arm to B300 nodes, optionally gated on a 20-step smoke run.
#
#   bash lmd_ssl_v3/submit.sh 1a_base               # smoke, then the real run chained on `done(smoke)`
#   bash lmd_ssl_v3/submit.sh --dry-run 1a_base     # write the job scripts, print the bsub lines, submit nothing
#   bash lmd_ssl_v3/submit.sh --smoke 1a_base       # smoke only
#   bash lmd_ssl_v3/submit.sh --no-smoke 1a_base    # real run only; resubmitting after a wall-time kill
#                                                   # continues via --resume
#
# lmd_ssl_v2/submit.sh with this experiment's paths, job names, miao pin and wall time, and three additions:
#   - The node count is read from the config: dp_replicate nodes of dp_shard x tp GPUs each. One node goes to
#     gpu_b300 under `torchrun --standalone`; several go to gpu_b300_parallel under mia-train's
#     deploy/lsf/launch_multinode.sh (arm 1f_p384: four nodes). The smoke always runs on one node.
#   - EXCLUDE_HOSTS="host1 host2" keeps jobs off the named nodes. Default: none.
#   - Each one-node job keeps its Python and compile caches in node-local scratch and deletes them when it ends.
# Everything lands in this experiment's directory on /nrs: jobs/ (LSF logs and the generated job scripts) and
# runs/ (run directories via --output-root); smoke artifacts go under jobs/smoke/ and runs/smoke/. If the smoke
# fails, the chained real job stays PEND forever: `bkill` it.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MIA_TRAIN=/groups/scicompsoft/home/orhane/projects/mia-train
VENV=/groups/scicompsoft/home/orhane/myvenv
PROJECT=miaai
EXP=/nrs/scicompsoft/orhane/mia-train-experiments/lmd_ssl_v3   # this experiment's home on /nrs
RUNS=$EXP/runs
LOGS=$EXP/jobs                      # LSF logs AND the generated job scripts
SMOKE_LOGS=$LOGS/smoke              # the smoke's config, script and logs
SMOKE_RUNS=$RUNS/smoke              # the smoke's run directories
QUEUE=${QUEUE:-gpu_b300}                            # one node
PARALLEL_QUEUE=${PARALLEL_QUEUE:-gpu_b300_parallel}  # several whole nodes
SLOTS_PER_GPU=12
SMOKE_GPUS=${SMOKE_GPUS:-1}         # the smoke's GPUs; its derived config gets dp_replicate = 1, dp_shard = SMOKE_GPUS
EXCLUDE_HOSTS=${EXCLUDE_HOSTS:-}
# The miao the job imports: a read-only copy of one commit of our fork (github.com/eminorhan/miao), first on
# PYTHONPATH; the job stops unless it is the miao imported. 28229ac = 0.4.2+fork.5, the first that lets
# mia-train load the N5 and precomputed volumes of this experiment's data configs.
MIAO_SRC=${MIAO_SRC:-/nrs/scicompsoft/orhane/probes/miao-pinned/28229ac/src}
THREADS="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4"
# inductor's layout optimisation can hand cuDNN SDPA a query with a non-unit last stride under compile, killing
# the run at step 1; every compiled run of ours sets this. A multi-node run does not get it (the launcher starts a
# fresh shell on each node), and SimMIM does not need it: the compiler leaves the layout alone for a model with one
# convolution (measured 2026-11-17, probes/layout_opt on /nrs).
COMPILE_ENV="export TORCHINDUCTOR_LAYOUT_OPTIMIZATION=0"
LAUNCHER=$MIA_TRAIN/deploy/lsf/launch_multinode.sh

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
mkdir -p "$RUNS" "$LOGS" "$SMOKE_RUNS" "$SMOKE_LOGS"

# The run's layout, from its [parallelism] section.
dim () { sed -n "s/^$1 = \([0-9][0-9]*\).*/\1/p" "$CONFIG" | head -1; }
NODES=$(dim dp_replicate); GPUS=$(( $(dim dp_shard) * $(dim tp) ))   # GPUs per node
if (( NODES > 1 && GPUS != 8 )); then
  echo "$NAME: $NODES nodes of $GPUS GPUs; a multi-node run takes whole nodes (dp_shard x tp = 8)" >&2; exit 2
fi
# 250k steps at 2.0 s (measured for 1a on 2026-11-17) is 5.8 days; 8 days leaves room for validation and
# checkpoints. Arm 1f measured 4.84 s per step, 14.0 days, which is LSF's limit: it resumes once if it must.
if (( NODES == 1 )); then WALL=${WALL:-192:00}; else WALL=${WALL:-336:00}; fi
# `-r` plus `--resume` turns a node failure into a requeue from the last checkpoint (every 25k steps).

SELECT=""
if [[ -n "$EXCLUDE_HOSTS" ]]; then
  for h in $EXCLUDE_HOSTS; do SELECT="${SELECT:+$SELECT && }hname!=$h"; done
  SELECT="select[$SELECT]"
fi

jobid () { grep -o 'Job <[0-9]*>' | head -1 | tr -dc 0-9; }

# write_cmd <dir> <tag> <config> <gpus per node> <nodes> <tail args...>: the script bsub runs, written into <dir>
write_cmd () {
  local dir=$1 tag=$2 cfg=$3 procs=$4 nodes=$5; shift 5
  local cmd="$dir/$tag.sh"
  { echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    echo "$THREADS"
    echo "$COMPILE_ENV"
    echo "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    if (( nodes == 1 )); then
      # Node-local scratch for the caches, removed when the job ends. A multi-node run keeps PyTorch's defaults:
      # the launcher starts a fresh shell on each node and passes on PYTHONPATH only.
      echo 'SCRATCH=/scratch/${USER:-$(id -un)}/lmd3_${LSB_JOBID:-manual}'
      echo 'mkdir -p "$SCRATCH"'
      echo "trap 'rm -rf \"\$SCRATCH\"' EXIT TERM"
      echo 'export PYTHONPYCACHEPREFIX=$SCRATCH/pyc TORCHINDUCTOR_CACHE_DIR=$SCRATCH/inductor'
    fi
    echo "export PYTHONPATH=$MIAO_SRC\${PYTHONPATH:+:\$PYTHONPATH}"
    echo "$VENV/bin/python -c \"import miao; assert miao.__file__.startswith('$MIAO_SRC'), miao.__file__; print('[miao]', miao.__file__)\""
    printf 'cd %q\n' "$MIA_TRAIN"
    echo 'echo "[mia-train] $(git rev-parse --short HEAD), uncommitted: $(git status --short --untracked-files=no | tr "\n" " ")"'
    if (( nodes == 1 )); then
      printf '%s --standalone --nproc_per_node=%s src/train.py --config %q %s\n' \
        "$VENV/bin/torchrun" "$procs" "$cfg" "$*"
    else
      # One torchrun per node under a c10d rendezvous; the launcher reads LSF's host list.
      printf 'MIA_TRAIN=%q VENV=%q %q %s %q %s\n' "$MIA_TRAIN" "$VENV" "$LAUNCHER" "$procs" "$cfg" "$*"
    fi
  } > "$cmd"
  chmod +x "$cmd"
  echo "$cmd"
}

# submit <logdir> <tag> <cmd> <gpus per node> <nodes> <wall> [dependency job id]  -> job id (or DRYRUN)
submit () {
  local logdir=$1 tag=$2 cmd=$3 procs=$4 nodes=$5 wall=$6 dep=${7:-}
  local args=(-P "$PROJECT" -W "$wall" -r -J "lmd3_$tag" -cwd "$MIA_TRAIN"
              -o "$logdir/lmd3_${tag}_%J.log" -e "$logdir/lmd3_${tag}_%J.err")
  if (( nodes == 1 )); then
    args+=(-q "$QUEUE" -gpu "num=$procs" -n "$((SLOTS_PER_GPU * procs))")
  else
    # `*_parallel` allocates whole nodes; `mode=shared` so one torchrun per node sees all its GPUs.
    args+=(-q "$PARALLEL_QUEUE" -app "parallel-$((SLOTS_PER_GPU * procs))" -gpu "num=$procs:mode=shared"
           -n "$((SLOTS_PER_GPU * procs * nodes))")
  fi
  [[ -n "$SELECT" ]] && args+=(-R "$SELECT")
  [[ -n "$dep" && "$dep" != DRYRUN ]] && args+=(-w "done($dep)")
  if [[ $DRY -eq 1 ]]; then
    { printf 'bsub'; printf ' %q' "${args[@]}"; printf ' %q\n' "bash '$cmd'"; } >&2; echo DRYRUN
  else
    bsub "${args[@]}" "bash '$cmd'" | jobid
  fi
}

SMOKE_ID=""
if [[ $SMOKE -eq 1 ]]; then
  # 20 steps on SMOKE_GPUS GPUs of one node, straight from the real config, with a validation and a checkpoint,
  # so the smoke exercises what actually runs (the data config, the model, the batch per rank).
  smoke_cfg="$SMOKE_LOGS/smoke_$NAME.toml"
  sed -e "s/^experiment_name = .*/experiment_name = \"smoke_lmd3__$NAME\"/" \
      -e 's/^max_steps = .*/max_steps = 20/'     -e 's/^warmup_steps = .*/warmup_steps = 2/' \
      -e 's/^val_every = .*/val_every = 10/'     -e 's/^checkpoint_every = .*/checkpoint_every = 20/' \
      -e 's/^samples_per_epoch = .*/samples_per_epoch = 20/' \
      -e 's/^dp_replicate = .*/dp_replicate = 1/' -e "s/^dp_shard = .*/dp_shard = $SMOKE_GPUS/" \
      -e 's/^num_workers = .*/num_workers = 2/'  -e 's/^log_every = .*/log_every = 5/' \
      "$CONFIG" > "$smoke_cfg"
  cmd=$(write_cmd "$SMOKE_LOGS" "smoke_$NAME" "$smoke_cfg" "$SMOKE_GPUS" 1 --output-root "$SMOKE_RUNS")
  SMOKE_ID=$(submit "$SMOKE_LOGS" "smoke_$NAME" "$cmd" "$SMOKE_GPUS" 1 2:00)
  echo "smoke  $NAME: job $SMOKE_ID  ($cmd)"
fi
if [[ $REAL -eq 1 ]]; then
  cmd=$(write_cmd "$LOGS" "$NAME" "$CONFIG" "$GPUS" "$NODES" --output-root "$RUNS" --resume)
  REAL_ID=$(submit "$LOGS" "$NAME" "$cmd" "$GPUS" "$NODES" "$WALL" "$SMOKE_ID")
  echo "real   $NAME: job $REAL_ID, $NODES node(s) x $GPUS GPU(s), wall $WALL  ($cmd)${SMOKE_ID:+, starts after done($SMOKE_ID)}"
fi
