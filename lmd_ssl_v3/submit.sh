#!/usr/bin/env bash
# lmd_ssl_v3: submit one arm to B300 nodes, optionally gated on a 20-step smoke run.
#
#   bash lmd_ssl_v3/submit.sh 1a_base               # smoke, then the real run chained on `done(smoke)`
#   bash lmd_ssl_v3/submit.sh --dry-run 1a_base     # write the job scripts, print the bsub lines, submit nothing
#   bash lmd_ssl_v3/submit.sh --smoke 1a_base       # smoke only
#   bash lmd_ssl_v3/submit.sh --no-smoke 1a_base    # real run only; resubmitting after a wall-time kill
#                                                   # continues via --resume
#
# lmd_ssl_v2/submit.sh with this experiment's paths, job names, pins (mia-train, miao) and wall time, and these
# additions:
#   - The node count is read from the config: dp_replicate nodes of dp_shard x tp GPUs each. One node goes to
#     gpu_b300 under `torchrun --standalone`; several go to gpu_b300_parallel under mia-train's
#     deploy/lsf/launch_multinode.sh (arm 1f_p384: four nodes). The smoke always runs on one node.
#   - EXCLUDE_HOSTS="host1 host2" keeps jobs off the named nodes. Default: none.
#   - Each one-node job keeps its Python and compile caches in node-local scratch and deletes them when it ends.
#   - An arm's configs are frozen at its first real submission: its TOML and the data config it names are copied
#     to jobs/frozen/<arm>/, and the arm's jobs read the copies, also after a requeue or a resubmission. Edits in
#     the checkout no longer reach that arm. To launch it again with changed configs, remove jobs/frozen/<arm>/.
#   - Each job starts by checking that every store of its data config exists and that no link of our wrappers
#     dangles: a store that its owner moved would otherwise read as zeros, with no error.
# Everything lands in this experiment's directory on /nrs: jobs/ (LSF logs, the generated job scripts and the
# frozen configs) and runs/ (run directories via --output-root); smoke artifacts go under jobs/smoke/ and
# runs/smoke/. A dry run writes its scripts and config copies under jobs/dryrun/, never over a real job's files.
# If the smoke fails, the chained real job stays PEND forever: `bkill` it.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# The mia-train the job runs: a read-only copy of one commit (git archive), so that work in the checkout cannot
# reach an arm that starts, requeues or resumes later. The job stops unless it is the mia-train imported.
# 8af6a8f = master on 2026-11-18.
MIA_TRAIN=${MIA_TRAIN:-/nrs/scicompsoft/orhane/probes/mia-train-pinned/8af6a8f}
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
WRAPPERS=/groups/miaai/miaai/lmd-external-wrappers   # our wrappers around volumes that stay in their owners' folders

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
GEN=$LOGS; if (( DRY )); then GEN=$LOGS/dryrun; fi   # where this call writes job scripts and config copies
mkdir -p "$RUNS" "$LOGS" "$SMOKE_RUNS" "$SMOKE_LOGS" "$GEN/smoke"

# The run's layout, from its [parallelism] section.
dim () { sed -n "s/^$1 = \([0-9][0-9]*\).*/\1/p" "$CONFIG" | head -1; }
NODES=$(dim dp_replicate); GPUS=$(( $(dim dp_shard) * $(dim tp) ))   # GPUs per node
if (( NODES > 1 && GPUS != 8 )); then
  echo "$NAME: $NODES nodes of $GPUS GPUs; a multi-node run takes whole nodes (dp_shard x tp = 8)" >&2; exit 2
fi
# 250k steps at 2.0 s (measured for 1a on 2026-11-17) is 5.8 days; 8 days leaves room for validation and
# checkpoints. Arm 1f measured 4.84 to 4.86 s per step on one node: 14.0 to 14.1 days, against LSF's limit of 14.
# It is watched by hand: if it will not finish, `bkill` it after the step-225k checkpoint and resubmit (--no-smoke).
if (( NODES == 1 )); then WALL=${WALL:-192:00}; else WALL=${WALL:-336:00}; fi
# `-r` plus `--resume` turns a node failure into a requeue from the last checkpoint (every 25k steps).

SELECT=""
if [[ -n "$EXCLUDE_HOSTS" ]]; then
  for h in $EXCLUDE_HOSTS; do SELECT="${SELECT:+$SELECT && }hname!=$h"; done
  SELECT="select[$SELECT]"
fi

jobid () { grep -o 'Job <[0-9]*>' | head -1 | tr -dc 0-9; }

# data_config <toml>: the data config a run config names (the same file under [data] and [val_data])
data_config () {
  local yaml
  yaml=$(sed -n 's/^config_path = "\(.*\)"$/\1/p' "$1" | sort -u)
  [[ -f $yaml ]] || { echo "$1: expected one existing data config under [data] and [val_data], found: $yaml" >&2; exit 2; }
  echo "$yaml"
}

# freeze <dir> <keep>: copy the arm's TOML and its data config into <dir>, point the copied TOML at the copied
# data config, and print the copied TOML. With keep = 1 an existing copy is left as it is, and its files are
# read-only; with keep = 0 (a dry run) the copy is written again on every call.
freeze () {
  local dir=$1 keep=$2 toml=$1/$NAME.toml yaml copy
  yaml=$(data_config "$CONFIG") || exit 2      # explicit: `set -e` does not reach into a $(...) call
  copy=$dir/$(basename "$yaml")
  if (( keep )) && [[ -f $toml ]]; then
    if ! { cmp -s "$yaml" "$copy" && sed "s|^config_path = .*|config_path = \"$copy\"|" "$CONFIG" | cmp -s - "$toml"; }; then
      echo "NOTE: $NAME.toml or its data config changed after this arm was frozen; it keeps the copies in $dir" >&2
    fi
  else
    { mkdir -p "$dir" && cp -f "$yaml" "$copy" &&
      sed "s|^config_path = .*|config_path = \"$copy\"|" "$CONFIG" > "$toml.tmp" && mv -f "$toml.tmp" "$toml"; } || exit 2
    if (( keep )); then chmod a-w "$toml" "$copy" || exit 2; fi
  fi
  echo "$toml"
}

# write_cmd <dir> <tag> <config> <gpus per node> <nodes> <tail args...>: the script bsub runs, written into <dir>
write_cmd () {
  local dir=$1 tag=$2 cfg=$3 procs=$4 nodes=$5; shift 5
  local cmd="$dir/$tag.sh" yaml
  yaml=$(data_config "$cfg") || exit 2
  { echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    # First, a few seconds: every store of the data config must exist, and a store that is one of our wrappers
    # must hold no dangling link. Looked at twice before giving up: a share not mounted yet can look absent once.
    echo "[[ -r '$yaml' ]] || { echo '[data] cannot read $yaml'; exit 1; }"
    echo "broken () { sed -n 's/^ *path: *//p' '$yaml' | sort -u | while read -r p; do if [[ ! -e \$p ]]; then echo \"\$p\"; elif [[ \$p == '$WRAPPERS'/* ]]; then find \"\$p\" -xtype l 2>&1; fi; done || true; }"
    echo 'bad=$(broken); if [[ -n $bad ]]; then sleep 30; bad=$(broken); fi'
    echo 'if [[ -n $bad ]]; then echo "[data] not starting: these stores are missing or hold a dangling link"; echo "$bad"; exit 1; fi'
    echo "echo \"[data] the \$(grep -c '^ *path: ' '$yaml') stores of $(basename "$yaml") exist, no wrapper link dangles\""
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
    echo "export PYTHONPATH=$MIA_TRAIN/src:$MIAO_SRC\${PYTHONPATH:+:\$PYTHONPATH}"
    echo "$VENV/bin/python -c \"import models; assert models.__file__.startswith('$MIA_TRAIN/src/'), models.__file__; print('[mia-train]', '$MIA_TRAIN')\""
    echo "$VENV/bin/python -c \"import miao; assert miao.__file__.startswith('$MIAO_SRC'), miao.__file__; print('[miao]', miao.__file__)\""
    printf 'cd %q\n' "$MIA_TRAIN"
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

# The config the jobs read. A real run reads the arm's frozen copy (made now if there is none; a dry run makes a
# throwaway one). A smoke on its own reads the checkout's file.
USED=$CONFIG
if [[ $REAL -eq 1 ]]; then USED=$(freeze "$GEN/frozen/$NAME" $(( 1 - DRY ))); fi

SMOKE_ID=""
if [[ $SMOKE -eq 1 ]]; then
  # 20 steps on SMOKE_GPUS GPUs of one node, straight from the real config, with a validation and a checkpoint,
  # so the smoke exercises what actually runs (the data config, the model, the batch per rank).
  smoke_cfg="$GEN/smoke/smoke_$NAME.toml"
  sed -e "s/^experiment_name = .*/experiment_name = \"smoke_lmd3__$NAME\"/" \
      -e 's/^max_steps = .*/max_steps = 20/'     -e 's/^warmup_steps = .*/warmup_steps = 2/' \
      -e 's/^val_every = .*/val_every = 10/'     -e 's/^checkpoint_every = .*/checkpoint_every = 20/' \
      -e 's/^samples_per_epoch = .*/samples_per_epoch = 20/' \
      -e 's/^dp_replicate = .*/dp_replicate = 1/' -e "s/^dp_shard = .*/dp_shard = $SMOKE_GPUS/" \
      -e 's/^num_workers = .*/num_workers = 2/'  -e 's/^log_every = .*/log_every = 5/' \
      "$USED" > "$smoke_cfg"
  cmd=$(write_cmd "$GEN/smoke" "smoke_$NAME" "$smoke_cfg" "$SMOKE_GPUS" 1 --output-root "$SMOKE_RUNS")
  SMOKE_ID=$(submit "$SMOKE_LOGS" "smoke_$NAME" "$cmd" "$SMOKE_GPUS" 1 2:00)
  echo "smoke  $NAME: job $SMOKE_ID  ($cmd)"
fi
if [[ $REAL -eq 1 ]]; then
  cmd=$(write_cmd "$GEN" "$NAME" "$USED" "$GPUS" "$NODES" --output-root "$RUNS" --resume)
  REAL_ID=$(submit "$LOGS" "$NAME" "$cmd" "$GPUS" "$NODES" "$WALL" "$SMOKE_ID")
  echo "real   $NAME: job $REAL_ID, $NODES node(s) x $GPUS GPU(s), wall $WALL  ($cmd)${SMOKE_ID:+, starts after done($SMOKE_ID)}"
fi
