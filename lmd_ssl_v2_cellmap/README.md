# Which lmd_ssl_v2 encoder transfers best to organelle semantic segmentation?

Each arm fine-tunes one encoder starting point with `semantic_seg` on the fit crops of mia-evals' two CellMap
organelle tasks and is scored on their test crops (mean IoU over 48 classes):
`cellmap_organelle_semantic` (228 fit crops, 43 test) and `cellmap_organelle_semantic_fewshot` (17 fit crops, one
per dataset; 193 test). Series 1 varies the starting point; series 2 changes one thing each about 1e, the encoder
lmd_ssl_v2 found best for affinities; series 3 changes 1c, the best 8 nm CellMap arm: 3a its resolution, 3b its
loss as well.

## Arms

Every arm has a large-task run (`<arm>.toml`) and a fewshot run (`<arm>_fewshot.toml`).

| arm | encoder starts from | differs from |
| --- | --- | --- |
| `1a_scratch` | random | — |
| `1b_lvd` | the released DINOv3 LVD-1689M ViT-L | 1a: init |
| `1c_simmim_lvd` | lmd_ssl_v2 `1a_simmim_lvd_b32`, step 100k (LVD, then SimMIM on lmd) | 1a: init |
| `1d_simmim_scratch` | lmd_ssl_v2 `1b_simmim_scratch_b32`, step 100k (SimMIM on lmd from scratch) | 1a: init |
| `1e_affinity_simmim_lvd` | lmd_ssl_v2 `2f`, step 100k (LVD, SimMIM, then affinities) | 1a: init |
| `2a_affinity_simmim_lvd_native` | as 1e, trained and predicted at each dataset's native voxel size | 1e: resolution |
| `2b_affinity_simmim_lvd_dice` | as 1e, cross-entropy + Dice 1:1 | 1e: loss |
| `2c_affinity_simmim_lvd_seed1` | as 1e, seed 1: run-to-run spread | 1e: seed |
| `3a_simmim_lvd_native` | as 1c, trained and predicted at each dataset's native voxel size (series 25k only) | 1c: resolution |
| `3b_simmim_lvd_native_dice` | as 3a, cross-entropy + Dice 1:1 (series 25k only) | 3a: loss |

All take the encoder only (`[init] prefix = "model."` for lmd_ssl_v2 checkpoints); the head starts fresh.

    bash lmd_ssl_v2_cellmap/submit.sh <run>              # a 20-step smoke, then the real run after it
    bash lmd_ssl_v2_cellmap/submit.sh --no-smoke <run>   # the real run only (also: resume after a wall kill)
    bash lmd_ssl_v2_cellmap/tensorboard.sh [port]        # every run in one TensorBoard

Runs, job logs, predictions and probes land in `/nrs/scicompsoft/orhane/mia-train-experiments/lmd_ssl_v2_cellmap/`.

## Recipe

- **Model**: lmd_ssl_v2's 3D DINOv3 ViT-L (256^3 windows, patch 8, axial RoPE, `attn_max_score` 1e5) with lmd_ssl_v2
  2a's UNETR head (skips after blocks 8/16/24, widths 16/32/64, raw-image skip), 75 outputs: CellMap ids 0-74.
- **Loss**: cross-entropy (2b, 3b: + Dice). 0, unannotated, is ignored. A composite id (a whole organelle painted
  without its parts, e.g. 50 = mitochondrion) is supervised as the set of its parts (`label_sets`: minus the log
  of the probability on the set), from `classes.csv`.
- **Data**: the task's fit crops at 8 nm (2a, 3a, 3b: native), without bounding boxes; a window is centred on a crop and
  its labels outside the crop are -1 (`label_fill`), which the loss ignores. Crops holding only composite ids
  (60 of the large task's: whole nuclei, mitochondria, peroxisomes, lipid droplets, mostly on 32 nm grids) are
  drawn at weight 0.25, about 9% of windows. Crops miao cannot centre a 256^3 window on inside the image are
  dropped: large task 206 of 228 at 8 nm, 226 at native; fewshot 17 of 17.
- **Training**: lmd_ssl_v2 2a's optimiser and augmentation (lr 3e-4, linear decay to 0.001x, weight decay 0.05,
  clip 1.0, BANIS augmentation with in-plane rotation); large task 100k steps (5k warmup, a checkpoint every
  25k), fewshot 25k (2k warmup, every 5k); one B300 node at batch 1 per GPU (global 8); no validation. Runs of a
  task with the same seed draw the same windows in the same order.
- **Prediction and scoring**: the final checkpoint only. `src/predict.py --argmax --no-truth --cover-box` on
  `data/predict_<task>_<res>.yaml`; each test crop's box adds half a window of context, shifted inward at the image
  faces, so the lattice covers every crop voxel. At 8 nm, 2-5 nm stores are read at pyramid level 1 or 2
  (mia-train's VolumeGrid). Scored with mia-evals `configs/<task>/identity.toml`.
- **miao**: our fork pinned at 62b3746 (`label_fill`), first on `PYTHONPATH` in every job.

## Files

- `make_configs.py` (run with `make_configs.sh` in a bsub job): writes every data and run config; edit there.
- `data/train_<task>_<res>.yaml`, `data/predict_<task>_<res>.yaml`: `<task>` large or fewshot, `<res>` 8nm or native.
- `submit.sh`, `tensorboard.sh`.
- `score.sh`: predict an arm's checkpoints on a task's test crops and score them into mia-evals' leaderboard
  (identity route), one prediction job per checkpoint.
- `<arm>_b32.toml`: series b32, the large task again for 15k steps (warmup 1k, a checkpoint every 1k) at 4 crops per
  GPU, a global batch of 32; otherwise the arm's first-round config.
- `<arm>_25k.toml`: series 25k, the arm's first-round large-task config for 25k steps (warmup 3k, a checkpoint every
  5k), one crop per GPU as there. `3a_simmim_lvd_native_25k.toml` is `1c_simmim_lvd_25k.toml` on the native data;
  `3b_simmim_lvd_native_dice_25k.toml` is 3a's with `dice_weight = 1.0`.

## Status

2026-10-26: all 16 runs launched. 2026-10-31: 15 done (2c large past 75k); every large-task checkpoint (8 arms x 4)
being predicted (H100s) and scored with `score.sh`, all 32 into mia-evals' leaderboard.
2026-11-04: the first round's checkpoints and predictions deleted (420 GB; configs, logs and records kept). Series
b32 launched (8 arms, large task), on mia-train 4e71956, whose semantic_seg cross-entropy is the mean of its windows'
means (at 4 crops per GPU, pooling voxels would weigh windows by their labelled fraction), and miao 4ef0d8c.
2026-11-05: all 120 b32 checkpoints scored into this experiment's board (`eval/board`, not mia-evals' leaderboard);
then b32's checkpoints and predictions deleted. Series 25k launched, each run's 5 checkpoints predicted and scored
into `eval/board_25k` by jobs chained on its training job (`score.sh --after`).
2026-11-06: that launch (warmup 2k) cancelled at steps 4k-6k, 2b's loss NaN from step 4k; its run folders and
checkpoints deleted, nothing scored. Relaunched at warmup 3k, with semantic_seg's logged metrics pooled over ranks
(mia-train 4e71956 plus that change). All 40 checkpoints scored into `eval/board_25k`, then put on mia-evals'
leaderboard in place of round 1's 32 rows (re-scored there to keep the scored copies its views show). Best: 2a at 15k
0.453, 2b at 10k 0.441, 1c at 20k 0.424. 2c diverged: loss NaN from step 14k (grad norm 5.8 at 13k), so its 15k-25k
checkpoints score 0. The experiment's three divergences (1e_b32, 2b's first 25k launch, 2c here) all start from the
2f encoder.
2026-11-07: 3a (1c at native voxel size) launched on series 25k's recipe and code, its 5 checkpoints chained into
`eval/board_25k` (`jobs/launch_3a_*.log`). The diverged 2c deleted (run, predictions, records on both boards) and
rerun unchanged (seed 1), scored straight into mia-evals' leaderboard behind a gate that stops the chain if its
training logged a NaN loss (`jobs/eval/gate_finite_training.sh`, `jobs/launch_2c_rerun_*.log`).
2026-11-08: 3a scored 0.311, 0.403, 0.461, 0.461, 0.444 at 5k-25k (1c: 0.318, 0.415, 0.409, 0.424, 0.404), the
board's best row; its 5 rows are on mia-evals' leaderboard, and its 20k checkpoint replaced 1c's on Hugging Face
(`ai-hhmi/lmd-vitl-p8-cellmap-lvd`). The 2c rerun stayed finite: 0.116, 0.236, 0.257, 0.285, 0.280.
2026-11-10: 3b (3a + Dice) launched on the same code (mia-train 73d9226 = 4e71956 plus the metrics change), behind
the NaN gate, its 5 checkpoints scored straight into mia-evals' leaderboard (`jobs/launch_3b_*.log`).

## Next iteration

Results came out worse than expected (2026-10-31). To fix next time:

1. **Intensity windows.** Every CellMap crop is uint8 / 255 here; several stores use a narrow band of grey levels
   (e.g. macrophage-2 210-249, jurkat-1 212-255). Take each store's p0.1-99.9 (or p1-99) window from lmd-configs'
   `corpus/windows.yaml` into `make_configs.py` (train + predict) and mia-evals' CellMap task data.
2. **NaN metrics in the training logs** (the loss is fine): `mean_iou` is NaN on ~69% of large-task log steps
   (composite-only windows) and `pixel_accuracy` on ~25% (windows with no annotated voxel; find those crops). Fixed
   2026-11-06 from the 25k relaunch on: semantic_seg sums the metrics' counts over ranks before taking ratios, so one
   rank's empty window no longer turns the logged value into NaN. Still open: which crops give those windows.
3. **Overfitting.** Test mean IoU falls with training in every pretrained arm (1d holds level 25k-50k, then falls);
   only scratch (1a) improves. Best: 1c at 25k, 0.415. Train ~25k steps.

   | arm | 25k | 50k | 75k | 100k |
   |---|---|---|---|---|
   | 1a scratch | 0.072 | 0.114 | 0.155 | 0.179 |
   | 1b lvd | 0.369 | 0.300 | 0.217 | 0.182 |
   | 1c simmim_lvd | 0.415 | 0.398 | 0.327 | 0.284 |
   | 1d simmim_scratch | 0.323 | 0.324 | 0.267 | 0.247 |
   | 1e affinity_simmim_lvd | 0.335 | 0.232 | 0.135 | 0.145 |
   | 2a ... native | 0.366 | 0.332 | 0.310 | 0.251 |
   | 2b ... dice | 0.328 | 0.226 | 0.182 | 0.141 |
   | 2c ... seed1 | 0.390 | 0.321 | 0.224 | 0.178 |

   1e and 2c differ only in seed and are 0.03-0.09 apart, so single-run gaps of a few hundredths are noise.
4. **Larger batch** than the global 8 used here (8 GPUs x 1); at most ~4 per B300 at 256^3, so more nodes beyond 32.
5. **A native version of the best arm (1c),** trained on native-resolution data as 2a is to 1e. Here 2a (native)
   beat 1e (8 nm) at every checkpoint, by more than the seed spread from 50k on. Running as 3a since 2026-11-07.
6. **Composite-only crops supervise positives only** (found 2026-11-10, `probes/class_balance/` on /nrs). In such a
   window the organelle is all that is supervised and nothing says where it is not (0 is ignored), which multiplies
   the organelle's share of the loss: peroxisomes 8.1x, lipid droplets 2.0x, nuclei 1.9x, mitochondria 1.16x. On
   test, 3a at 20k predicts 1.80, 1.13, 1.39 and 1.08 times the labelled volume of these four, and 0.48-0.98 times
   that of the seven organelles with no composite-only crops. To test: train without the 60 crops; to fix: label
   their background as "any class but this organelle's" (a label set).
7. **mito_mem over-predicted 1.24x** by 3a at 20k (1.19-1.26x in the other arms; 2b with Dice 1.05x in series 25k,
   1.15x in b32). Eight crops carry it (mouse liver and kidney, ut21, sum159-1: 2.2x, precision 0.40; liver-zon-1
   crops 313 and 267 hold 57% of the excess), where it is cytosol (34% of the predicted voxels) and ER (12%) called
   mito_mem; the other 16 crops with mito_mem are at 1.04x. Cause open.
8. **Where 3a at 20k fails** (2026-11-10, `probes/class_balance/report_failures.txt`): two datasets hold the score
   down -- ut21 (4 crops, 21% of all wrong voxels; extracellular space called nucleoplasm on 13% of its voxels, the
   board's largest confusion; mean IoU 0.488 without it) and liver-zon-1 (5 crops, 18%; cytosol called mito_mem;
   0.481 without it). About 70% of the nucleus over-prediction in item 6 is that one ut21 confusion, so item 6's
   evidence for nuclei is weaker than its table suggests. Elsewhere the wrong voxels are ER (24%) and mitochondrial
   (18%) membrane-lumen-cytosol boundaries and plasma membrane (9%; fly-vnc-1 worst).
9. **The mito_mem flooding of item 7 is late-training overfitting** (2026-11-10, `probes/class_balance/report_across.txt`,
   `report_by_step.txt`, pictures in `liver_mito/`). 3a's cytosol called mito_mem is 0.44, 0.47, 0.38, 0.77, 1.17% of
   the test voxels at 5k-25k (mito_mem IoU 0.627, 0.648, 0.665, 0.595, 0.547); crop 267 goes 0.91, 1.80, 1.38, 2.54,
   4.43x while the well-predicted liver crops stay at 1.0-1.2x at every step; 1c does the same. Not a halo (over half
   of the predicted mito_mem lies outside every labelled mitochondrion, median 47-71 nm away), not thinner labels
   (20-25 nm in those crops, 24 nm median in the dataset's training crops), no consistent grey-level signature. At 15k
   3a has the same board score as at 20k (0.4606 vs 0.4608) and half the flooding. The recipe trains the whole ViT-L
   at lr 3e-4 with `layerwise_lr_decay = 1.0` and `drop_path_rate = 0.0`, and has no held-out split to stop on.

2026-11-11: 3b (3a + Dice) scored 0.312, 0.460, 0.506, 0.472, 0.458 at 5k-25k against 3a's 0.311, 0.403, 0.461, 0.461,
0.444: the board's best row (3b at 15k), on mia-evals' leaderboard (50 rows). At 15k Dice raises 37 of the 48
classes and brings the under-predicted leaf classes from 0.43 to 0.80 of their labelled volume; it does not stop the
late mito_mem flooding (cytosol called mito_mem 0.47% of voxels at 15k, 0.72% at 20k; 3a 0.38%, 0.77%). The Hugging
Face model is 3a at 15k since 2026-11-10 (it replaced 3a at 20k).
