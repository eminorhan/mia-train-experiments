# Which lmd_ssl_v2 encoder transfers best to organelle semantic segmentation?

Each arm fine-tunes one encoder starting point with `semantic_seg` on the fit crops of mia-evals' two CellMap
organelle tasks and is scored on their test crops (mean IoU over 48 classes):
`cellmap_organelle_semantic` (228 fit crops, 43 test) and `cellmap_organelle_semantic_fewshot` (17 fit crops, one
per dataset; 193 test). Series 1 varies the starting point; series 2 changes one thing each about 1e, the encoder
lmd_ssl_v2 found best for affinities.

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

All take the encoder only (`[init] prefix = "model."` for lmd_ssl_v2 checkpoints); the head starts fresh.

    bash lmd_ssl_v2_cellmap/submit.sh <run>              # a 20-step smoke, then the real run after it
    bash lmd_ssl_v2_cellmap/submit.sh --no-smoke <run>   # the real run only (also: resume after a wall kill)
    bash lmd_ssl_v2_cellmap/tensorboard.sh [port]        # every run in one TensorBoard

Runs, job logs, predictions and probes land in `/nrs/scicompsoft/orhane/mia-train-experiments/lmd_ssl_v2_cellmap/`.

## Recipe

- **Model**: lmd_ssl_v2's 3D DINOv3 ViT-L (256^3 windows, patch 8, axial RoPE, `attn_max_score` 1e5) with lmd_ssl_v2
  2a's UNETR head (skips after blocks 8/16/24, widths 16/32/64, raw-image skip), 75 outputs: CellMap ids 0-74.
- **Loss**: cross-entropy (2b: + Dice). 0, unannotated, is ignored. A composite id (a whole organelle painted
  without its parts, e.g. 50 = mitochondrion) is supervised as the set of its parts (`label_sets`: minus the log
  of the probability on the set), from `classes.csv`.
- **Data**: the task's fit crops at 8 nm (2a: native), without bounding boxes; a window is centred on a crop and
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

## Status

2026-10-26: configs built and checked; nothing launched.
