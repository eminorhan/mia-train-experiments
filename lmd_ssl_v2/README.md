# Does SimMIM on the lmd corpus beat the LVD checkpoint for affinity prediction, and do LSDs or a SAM model help?

SimMIM pretraining on the 2026-10 lmd corpus (arms 1x), then affinity prediction on its labelled
volumes (arms 2x), from the SimMIM encoders, from the released DINOv3 LVD-1689M checkpoint, and from
scratch. Two variants of the affinity arm test local shape descriptors (MTLSD) and a Segment-Anything
model.

Every arm: 3D DINOv3 ViT-L (`dinov3_vit3d`), patch 8, 256^3 crops at 8 nm (32,768 tokens), axial 3D
RoPE, 100k steps, 5k warmup, lr 3e-4 linear, one B300 node (`dp_shard = 8`, batch 1 per rank,
global batch 8) unless noted. Arms with the same data, batch size and seed draw the same crops at
every step, so their curves compare step by step.

## Arms

| arm | what | differs from |
| --- | --- | --- |
| `1a_simmim_lvd` | SimMIM from LVD-1689M; mask ratio 0.9 in 32-voxel blocks | — |
| `1b_simmim_scratch` | SimMIM from random init | 1a: init |
| `1a_simmim_lvd_b32` | batch 4 per rank (global 32) | 1a: batch |
| `1b_simmim_scratch_b32` | batch 4 per rank (global 32) | 1b: batch |
| `1c_simmim_lvd_b32_mask60_g2` | mask ratio 0.6 in 16-voxel blocks: a more local task | 1a_b32: masking |
| `2a_affinity_unetr_lvd` | affinities (`affinity_seg`, UNETR head) from LVD-1689M | — |
| `2b_affinity_unetr_scratch` | from random init | 2a: init |
| `2c_mtlsd_unetr_lvd` | plus local shape descriptors, sigma 80 nm, weight 1 | 2a: LSDs |
| `2d_sam_musam128_lvd` | promptable segmentation, `sam_lmd_v1` arm 8's recipe | 2a: algorithm |
| `2e_mtlsd_unetr_lvd_w10` | LSD weight 10 | 2c: weight |
| `2f_affinity_unetr_simmim_lvd` | from `1a_simmim_lvd_b32`'s final encoder | 2a: init |
| `2g_affinity_unetr_simmim_scratch` | from `1b_simmim_scratch_b32`'s final encoder | 2a: init |
| `2h_affinity_unetr_simmim_lvd_mask60_g2` | from `1c_simmim_lvd_b32_mask60_g2`'s final encoder | 2a: init |

    bash lmd_ssl_v2/submit.sh <arm>              # a 20-step smoke, then the real run after it
    bash lmd_ssl_v2/submit.sh --no-smoke <arm>   # the real run only (also: resume after a wall kill)
    bash lmd_ssl_v2/submit.sh --dry-run <arm>    # write the job script, print the bsub line
    bash lmd_ssl_v2/tensorboard.sh [port]        # every arm in one TensorBoard

Runs, job logs and probes land in `/nrs/scicompsoft/orhane/mia-train-experiments/lmd_ssl_v2/`.

## Data

- **SSL corpus** (`data/8nm_p256_swe0.4_min1e8_up8_census20261015.yaml`): lmd-configs' census-20261015
  single-scale unlabeled config (242 volumes, 8 nm), plus the zebra finch j0126 volume trimmed to its
  tissue by a bounding box.
- **Labelled data**: lmd-configs' v2 instance-segmentation config, 16 volumes, split 8/8 by volume.
  The test half (`..._test.yaml`) is held out for mia-evals; the affinity arms train and validate on
  the finetune half (`..._finetune.yaml`). That file also holds the 33 j0126 zebra finch training
  cubes (weight 1/33 each, 1/9 of the draws), read with `label_fill = -1` so windows can reach past
  a cube.

  | finetune | test |
  | --- | --- |
  | `kasthuri15_ac3` | `kasthuri15_ac4` |
  | `zebrafish_fish2_quadcube1` | `zebrafish_fish2_doublecube1` |
  | `liconn_mouse_dg` | `liconn_mouse_hippocampus` |
  | `hemibrain_ellipsoid_body` | `liconn_expid82` |
  | `flyid49_40xw005` | `flyid49_40xw006` |
  | `cns_mito_crop182` | `cns_mito_crop164` |
  | `cremi_sample_a`, `cremi_sample_b` | `cremi_sample_c`, `jiefu_cerebellum` |

- **miao**: our fork (github.com/eminorhan/miao), pinned at commit 62b3746 and put first on
  `PYTHONPATH` by `submit.sh`. The labelled arms need three of its changes: `label_fill`, labels
  resampled at voxel centres (nearest-exact), and `label_margin`.

## Notes

- **Labels past the window.** The affinity arms read labels 10 voxels past each window face
  (`label_margin`), so affinities at the faces are supervised rather than masked; the LSD arms read
  30, the descriptors' reach (3 sigma). Both targets are then the reference's (Sheridan et al.
  2023, and gunpowder's `AddAffinities`) at every voxel, faces included. One difference remains:
  the reference's input exceeds its output by that reach, so it never trains a voxel whose target
  depends on labels it cannot see; these arms train the whole window, faces included.
- **The LSD term is weak at weight 1.** In 2c the LSD loss is ~6% of the total (BCE affinities
  against MSE on sigmoid descriptors); 2e raises the weight to 10.
- **2d** draws its prompts in the dataloader workers, so it runs with `defer_image_ops = false`:
  deferred augmentation would rotate the image after the prompts were placed. It keeps arm 8's
  32 nm mask cells at patch 8 with `mask_upscale = 2`, and needs activation checkpointing
  (234 GB of 268 per GPU).
- **SimMIM's loss is not the readout.** Rank the SSL arms by their fine-tunes (2f-2h against 2a/2b),
  scored with mia-evals on the test half; `val/boundary_accuracy` is a proxy.
