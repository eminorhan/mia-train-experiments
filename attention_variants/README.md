# attention_variants: windowed and sliding-window attention in place of global attention

Global attention scores every pair of tokens, so its cost grows with the square of the crop's volume.
`lmd_ssl_v3`'s arm 1f trains a DINOv3 ViT-L with SimMIM on 384³ crops at patch 8: 48³ = 110,592 tokens per crop.
Attention is 94% of its 4.8 s step (`lmd_ssl_v3/probes/profile_1f` on /nrs).

mia-train has cheaper attention patterns for `dinov3_vit3d` and `vit3d` (`layers/common/window_attention.py`).
They were built and timed on a segmentation model in [`large_inputs`](../large_inputs/README.md), phase 2:

- **block windows** (`attn_window = N`): each token attends within its own N³-token box of the grid;
- **sliding windows** (`attn_window_mode = "sliding"`): the window moves with the token, tile by tile, so there
  are no fixed seams;
- **global blocks** (`attn_global_blocks`) among the windowed ones, optionally over pooled keys and values
  (`attn_global_kv_pool`).

This experiment works on these variants in 1f's setting.

## 1. Throughput of block windows (2026-11-21)

**Question.** How fast does 1f's training step run with block windows instead of global attention?

**Arms.** Each arm is 1f's model and data plus `attn_window = N`, with the run settings of a
benchmark (see Method). A windowed block attends within block windows, and the CLS and storage tokens attend
globally. The first four arms have no global blocks, so nothing but those five tokens carries information between
windows: they measure cost, they are not proposed models. The fifth adds `attn_global_blocks`: full global
attention in four blocks.

| arm | window | windows per crop |
|---|---|---|
| `p384_block24` | 24³ tokens = 192³ voxels | 8 |
| `p384_block16` | 16³ tokens = 128³ voxels | 27 |
| `p384_block12` | 12³ tokens = 96³ voxels | 64 |
| `p384_block8` | 8³ tokens = 64³ voxels | 216 |
| `p384_block24_g4` | 24³ tokens in 20 blocks; full global attention in blocks 5, 11, 17 and 23 | 8 |

**Method.** `throughput.sh` runs each arm's config through the real trainer on real data: compiled, bf16, one crop
per GPU, with the mia-train snapshot (8af6a8f) and the miao (28229ac) that 1f runs. The arm files hold the
benchmark's settings (100 steps, a log line every 20, no validation, no checkpoint); the script sets only the
number of ranks, to the GPUs of its job. Every figure is the median over the log intervals that carry a rate: all
but the first, which holds compilation.

The numbers below come from job 155664392 on three B300 GPUs of i04u22, which ran 300 steps per arm. Its first
100 steps give the same step times within 0.5%, so the arm files now say 100. `p384_block24_g4` ran 100 steps
in job 155664503 on the same GPUs.

The global-attention baseline was not rerun. 1f measured 4.80 s per step on one GPU (2026-11-19), 4.84 s on two
(2026-11-17, the same method) and 4.86 s on eight (2026-11-02); its live run on 32 GPUs logs 4.96 s.

**Predicted before the run.** 1f's profile splits a step into 4.52 s of attention and 0.29 s of everything else.
If attention time scales with the number of scored pairs, the arms take 0.85 / 0.45 / 0.36 / 0.31 s per step
(windows of 24 / 16 / 12 / 8). A slower step would point to the loader, which must then supply up to 3 crops per
second per GPU instead of 0.2, or to the attention kernel at small windows.

**Results.** `jobs/throughput/table_155664392.txt` and `throughput_155664503.log` on /nrs.

| arm | s per step | speed-up over 1f | crops per s per GPU | data wait, worst rank | MFU | peak GPU memory |
|---|---|---|---|---|---|---|
| 1f, global attention (two GPUs, 2026-11-17) | 4.84 | 1 | 0.21 | 0.01% | 0.40 | 153.7 GiB |
| `p384_block24` | 0.959 | 5.0x | 1.04 | 0.07% | 0.34 | 168.7 GiB |
| `p384_block16` | 0.571 | 8.5x | 1.75 | 0.12% | 0.28 | 168.7 GiB |
| `p384_block12` | 0.476 | 10.2x | 2.10 | 0.14% | 0.25 | 168.7 GiB |
| `p384_block8` | 0.432 | 11.2x | 2.31 | 0.15% | 0.23 | 168.8 GiB |
| `p384_block24_g4` | 1.595 | 3.0x | 0.63 | 0.04% | 0.37 | 166.2 GiB |

* **Block windows alone cut the step 5.0x to 11.2x; with four full global blocks at window 24, 3.0x.**
* **Four full global blocks add 0.64 s per step** at window 24 (0.959 to 1.595 s; the fit below predicted
  1.60 s). A global block costs 0.18 s, a block of 24-token windows 0.02 s.
* **A step is a fixed 0.41 s plus a cost per scored pair.** With the pairs of one layer (the mean over the 24
  layers), `0.41 s + 3.58e-10 s x
  pairs` reproduces the four windows-only arms within 1% and gives 4.79 s for global attention (measured:
  4.84 s). So the
  attention kernel costs the same per pair from 517-key windows up to the whole crop, and the fixed part caps the
  speed-up at 11.8x. Windows of 8 tokens are already within 5% of that cap.
* **Against the prediction:** each windows-only arm is 0.11 to 0.13 s slower. The scaling with pairs held. The fixed part did
  not: it is 0.41 s, not the 0.29 s that 1f's one-GPU profile measured outside attention. Where the other 0.12 s
  goes is not identified (1f itself takes 0.04 s longer on two GPUs than on one).
* **The loader keeps up** at up to 2.3 crops per second per GPU with 8 workers per rank.
* **Memory:** 168.7 GiB whatever the window and 166.2 GiB with the four global blocks, 13 to 15 GiB above 1f.
* **Not measured: global blocks over pooled keys and values.** By the fit, one over 2³-pooled keys and values
  would cost about 0.02 s per step.

## Where things are

`/nrs/scicompsoft/orhane/mia-train-experiments/attention_variants/`: `jobs/throughput/` (job logs, the derived
configs, `results/`), `runs/throughput/` (the run directories of the short runs), `probes/`.
