#!/usr/bin/env bash
# Serve the lmd_ssl_v2 arms in one TensorBoard.
#
#   bash lmd_ssl_v2/tensorboard.sh [port]
#   bash lmd_ssl_v2/tensorboard.sh 6008 --smoke     # smoke runs instead
#   bash lmd_ssl_v2/tensorboard.sh 6006 --list      # rebuild the tree, print, do not serve
#
# gary_comparison/tensorboard.sh with this experiment's arms. Rebuilds the symlink tree each time, so an arm appears
# as soon as its run directory exists (the newest run of an arm wins if it was rerun). The tree is `tensorboard/` in
# this experiment's /nrs directory; smoke runs are linked under `tensorboard/smoke/` and served from there.
#
# THE ARMS
#
#   1a_simmim_lvd      SimMIM on the lmd corpus + j0126, warm-started from the released DINOv3 LVD-1689M ViT-L;
#                      patch 8, mask 0.9 in 32-voxel blocks, 100k steps, global batch 8, one B300 node
#   1b_simmim_scratch  arm 1a from random init; 1a vs 1b is the value of the natural-image checkpoint
#   1a_simmim_lvd_b32      arm 1a at batch 4 per rank (global 32), nothing else; 4x the samples in the same 100k steps
#   1b_simmim_scratch_b32  arm 1b at batch 4 per rank (global 32), nothing else
#                      Each logged train point averages 32 crops instead of 8, so expect about half the scatter.
#                      Compare with 1a/1b at equal samples (b32 step S = b8 step 4S) as well as equal steps.
#   1c_simmim_lvd_b32_mask60_g2  arm 1a_b32 with mask ratio 0.6 and 16-voxel masked blocks (1a: 0.9, 32 voxels)
#                      Its loss is not comparable with 1a's: an easier, more local prediction task.
#   2a_affinity_unetr_lvd      affinity_seg + UNETR on the v2 finetune half + 33 zebrafinch cubes, LVD start
#   2b_affinity_unetr_scratch  arm 2a from random init
#                      Early check (cold head): train/affinity_accuracy should leave train/target_positive_rate by ~3k.
#   2c_mtlsd_unetr_lvd         arm 2a plus the LSD auxiliary target (sigma 80 nm), labels read 30 voxels past the window
#                      train/loss = loss_affinity + loss_lsd; compare 2a on train/loss_affinity, not train/loss.
#   2e_mtlsd_unetr_lvd_w10     arm 2c with lsd_weight 10 (2c: 1); train/loss = loss_affinity + 10 x loss_lsd
#   2f_affinity_unetr_simmim_lvd           arm 2a from 1a_simmim_lvd_b32's final encoder instead of LVD's
#   2g_affinity_unetr_simmim_scratch       arm 2a from 1b_simmim_scratch_b32's final encoder
#   2h_affinity_unetr_simmim_lvd_mask60_g2 arm 2a from 1c_simmim_lvd_b32_mask60_g2's final encoder
#   2d_sam_musam128_lvd        promptable segmentation (sam_lmd_v1 arm 8's recipe) on 2a's encoder, init and data
#                      first_iou / final_iou: mask IoU after the first prompt and after two correction rounds.
#
# WHAT TO READ
#
#   train/loss         Masked L1 on the 90% of 32-voxel blocks the encoder could not see. In every earlier SimMIM
#                      run here the smoothed masked loss went flat by step 10-25k, so a flat curve after that is
#                      expected and is not the readout; only a fine-tune ranks the arms. Expect 1a below 1b early.
#   train/visible_l1   Error on the patches the encoder could see (not in the loss). No single healthy value: it
#                      depends on the init. From LVD (1a): close to the masked loss and a little above it, in every
#                      earlier LVD-init SimMIM run here (new_ssl_recipe 2a/3a/4a, simmim_vs_direct A1; e.g. 0.106 vs
#                      0.116 at 25k). From scratch (1b): far above at the start (0.6-2.3), then falling by an amount
#                      that varies (gary 2c 0.58 -> 0.13 by 100k; 2a/2b stayed at 0.5-0.8; init_comparison 3a ~1.2).
#                      So visible_l1 near the loss is NOT a collapse sign for 1a. A collapsed model's masked loss sits
#                      at the error of predicting each crop's median.
#   val/loss           128 un-augmented crops drawn fresh from the corpus every 10k steps: different crops each
#                      time, so read the trend over several points, not single values.
#   train/masked_fraction  0.9004 = 461 of 512 blocks.
#   samples_per_s      ~15 for 1a/1b (0.52 s/step; their 48 h wall holds down to ~5 samples/s) and ~16 for the b32
#                      arms (~2 s/step, ~55 h; 96 h wall). Batch does not change throughput per sample here.
#   data_wait_frac     Near 0. 1a/1b never reach an epoch boundary (an epoch is 125k steps); the b32 arms cross
#                      three (every 31,250 steps), each a few-second loader respawn.
#   grad_norm, lr      Clip 1.0; lr reaches 3e-4 at step 5k and decays linearly to 3e-7 at 100k.
set -euo pipefail

EXP=/nrs/scicompsoft/orhane/mia-train-experiments/lmd_ssl_v2   # this experiment's home on /nrs: runs/ jobs/ probes/
RUNS=$EXP/runs
VIEW=$EXP/tensorboard
VENV=/groups/scicompsoft/home/orhane/myvenv
ARMS=(1a_simmim_lvd 1b_simmim_scratch 1a_simmim_lvd_b32 1b_simmim_scratch_b32 1c_simmim_lvd_b32_mask60_g2
      2a_affinity_unetr_lvd 2b_affinity_unetr_scratch
      2c_mtlsd_unetr_lvd 2d_sam_musam128_lvd 2e_mtlsd_unetr_lvd_w10
      2f_affinity_unetr_simmim_lvd 2g_affinity_unetr_simmim_scratch 2h_affinity_unetr_simmim_lvd_mask60_g2)

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

echo "arms found:"
# Only the run directory's own _YYYYMMDD_HHMMSS suffix may follow the arm name: a bare `_*` would let arm 1a_simmim_lvd
# match 1a_simmim_lvd_b32's directories and, as the newest, take its link.
STAMP="[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9]"
for arm in "${ARMS[@]}"; do
  if [[ $SMOKE -eq 1 ]]; then
    link "$arm" "$RUNS/smoke_lmd2__${arm}_${STAMP}/"
  else
    link "$arm" "$RUNS/lmd2__${arm}_${STAMP}/"
  fi
done
(( ${#missing[@]} )) && echo "  not started: ${missing[*]}"
[[ $LIST -eq 1 ]] && exit 0
echo; echo "http://localhost:$PORT"
exec "$VENV/bin/tensorboard" --logdir "$VIEW" --port "$PORT"
