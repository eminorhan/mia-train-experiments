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

## 2. Do the window-24 arms hold a 768³ crop? (2026-11-21)

**Question.** Does one B300 GPU hold a 768³ crop (96³ = 884,736 tokens, 64 windows) with the two window-24
designs, and how fast is the step?

**Arms.** The 384³ arms with `img_size = 768` and three more changes:

| arm | attention | activation checkpointing |
|---|---|---|
| `p768_block24` | 24³-token windows in all 24 blocks | on |
| `p768_block24_g4` | the same, with full global attention in blocks 5, 11, 17 and 23 | on |
| `p768_block24_noac` | as `p768_block24` | off |

- **Activation checkpointing.** Without it the 384³ arms hold 169 GiB, and 768³ has 8x the tokens.
  `p768_block24_noac` tests that directly.
- **Data: one volume.** 1f's data config is built for 384³ crops and holds volumes too small for 768³, so these
  arms read `data_configs/opticlobe_box1_8nm_p768.yaml`: the FlyEM optic lobe, stored at 8 nm, one box of
  21,504 x 28,800 x 13,312 voxels.
- **No start-up FLOP probe** (`measure_mfu = false`), so no MFU. The trainer's probe runs one uncompiled step, and
  at 768³ that step does not fit: both arms died in it at 265.5 GiB, with checkpointing on, before any training
  step (job 155665615). The step time is the wall-clock time between two log lines.

**Results.** Three B300 GPUs of i04u22, compiled, one crop per GPU; jobs 155665637 and 155665653.

| arm | holds a 768³ crop | peak GPU memory | s per step | Mvoxel per s per GPU |
|---|---|---|---|---|
| `p768_block24` | yes | 150.5 GiB | 9.60 | 47.2 |
| `p768_block24_g4` | yes | 149.0 GiB | 74.6 | 6.1 |
| `p768_block24_noac` | no: out of memory in the first step's forward pass, at 264.7 GiB | - | - | - |
| for reference: 1f at 384³ | | 153.7 GiB | 4.84 | 11.7 |
| for reference: `p384_block24` | | 168.7 GiB | 0.959 | 59.0 |

* **Both designs hold a 768³ crop with activation checkpointing, and neither does without it.**
* **Windows alone: 9.60 s per step,** 4.0x 1f's voxels per second. The fit of section 1, with a quarter added for
  the recomputed forward pass, gave 9.7 s.
* **With four full global blocks: 74.6 s per step,** half of 1f's voxels per second. The four global blocks add
  65 s to the step: each scores 64x the pairs it does at 384³. The same estimate gave 67 s, 11% too low.
* **The loader keeps up** (data wait 0.01% or less).
* **A run with the default `measure_mfu = true` dies at start-up at this size,** although its training step fits.

## 3. Sliding windows under torch.compile: where does the extra cost go? (2026-11-21)

**Question.** `large_inputs` measured sliding windows at +14% over block windows uncompiled and +53% compiled. It
put the difference down to the block mask (the table of which key tiles each query tile attends to) being rebuilt
in every layer under `torch.compile`. That cause was never measured. Is it the cause?

**Method.** `probes/sliding_compile/` on /nrs times one attention layer, forward plus backward, at 1f's shapes (5
prefix tokens, 16 heads of 64, bf16) with 16-token windows, on one B300 GPU, one process per grid size (job
155665924). Three variants of the sliding layer: compiled as a compiled model runs it today (the mask is rebuilt
in the graph); compiled with the mask built beforehand and handed in; uncompiled as an uncompiled model runs it
(cached mask, FlexAttention compiled on its own).

**Results.** Milliseconds per forward plus backward, with FlexAttention's FLASH kernel (its Triton kernel in
brackets). All sliding variants give identical outputs.

| token grid | block, compiled | sliding, uncompiled | sliding, compiled as today | sliding, compiled, mask handed in | building the mask once |
|---|---|---|---|---|---|
| 32³ | 4.52 | 12.01 (18.40) | 12.65 (18.22) | 10.25 (16.88) | 1.4 |
| 48³ (1f) | 12.89 | 33.42 (62.97) | 30.03 (58.62) | 28.84 (58.18) | 1.4 |
| 64³ | 30.17 | 79.66 (137.93) | 68.96 (127.74) | 68.09 (126.03) | 1.4 |

* **The layer has no compile penalty.** Compiled as today, the sliding layer is 10% and 13% faster than uncompiled
  at 48³ and 64³, and 5% slower at 32³.
* **The mask rebuild is real but small:** 1.4 ms per layer and step at every grid. Handing the mask in saves 0.9 to
  2.4 ms: 19% of the layer at 32³, 4% at 48³, 1% at 64³.
* **So the +53% is not reproduced in the layer, and the rebuild is not its cause.** It belongs to the model-level
  setting of that run (compile with activation checkpointing, the library versions of September), which this
  probe does not cover.
* **One process must see one crop size.** In the first job (155665890) one process timed three grids. At the second
  grid `torch.compile` recompiled the layer for variable sizes; there the FLASH kernel refuses to run and the Triton
  kernel takes 2.2 to 2.4 times as long as uncompiled. A compiled model with sliding windows that meets a second
  crop size would fail or slow down in the same way.
* **Sliding against block windows, per layer, compiled:** 2.8x / 2.3x / 2.3x the time at 32³ / 48³ / 64³, for
  1.8x / 2.1x / 2.2x the pairs.
* **FLASH against Triton:** 1.4x / 2.0x / 1.9x faster at the three grids.

**In the trainer.** `p384_slide24` and `p384_slide16` are `p384_block24` and `p384_block16` with
`attn_window_mode = "sliding"`: every token sees at least the window centred on it, in all 24 blocks, and the
windows move in tiles of 8 x 8 x 4 tokens whatever the window. Compiled, 100 steps, three B300 GPUs, no start-up
FLOP probe (jobs 155665990 and 155666031):

| arm | s per step | speed-up over 1f | peak GPU memory |
|---|---|---|---|
| `p384_slide24` | 1.877 | 2.6x | 148.9 GiB |
| `p384_slide16` | 1.073 | 4.5x | 148.9 GiB |
| `p384_slide16_g4`: `p384_slide16` with full global attention in blocks 5, 11, 17 and 23 (job 155666051) | 1.693 | 2.9x | 146.4 GiB |
| `p384_block24` (section 1) | 0.959 | 5.0x | 168.7 GiB |
| `p384_block16` (section 1) | 0.571 | 8.5x | 168.7 GiB |
| `p384_block24_g4` (section 1) | 1.595 | 3.0x | 166.2 GiB |

* **Sliding windows take about twice the time of block windows of the same size:** 1.96x at 24 tokens for 1.9x
  the pairs, 1.88x at 16 tokens for 2.1x the pairs.
* **Four full global blocks add 0.62 s to the sliding arm** (1.073 to 1.693 s), as they add 0.64 s to the
  block-window arm. The estimate beforehand was 1.69 s.
* **No compile penalty shows at the model level either** in this setting (no activation checkpointing): the layer
  timings above gave estimates of 1.7 s and 0.98 s beforehand, each 9% under its measurement.
* **Memory:** 148.9 GiB with 24 sliding blocks and 146.4 GiB with four of them global. A sliding block holds
  0.625 GiB more than a global one, the same difference as for a block-window block (section 1). The attention
  kernels never store the scores, so a global block costs time, not memory.
* **These memory figures cannot be compared with section 1's.** The sliding arms ran without the start-up FLOP
  probe, and that probe's uncompiled step sets the peak in the arms of section 1. `p384_block24_noprobe`, which is
  `p384_block24` with `measure_mfu = false` (job 155666108), measured 142.0 GiB at 0.955 s per step, against
  168.7 GiB with the probe. So the compiled training step needs 142.0 GiB with block windows of 24 and 148.9 GiB
  with sliding windows: sliding windows hold 7 GiB more. Section 1's memory figures, 1f's included, are peaks of
  the probe step.

## 4. Sliding windows at 768³, and why they are slow there (2026-11-22)

**Arms.** `p768_slide16` is `p768_block24` (section 2: activation checkpointing on, one large volume, no start-up
FLOP probe) with sliding windows of 16 tokens.

| arm at 768³ | s per step | peak GPU memory over the run | job |
|---|---|---|---|
| `p768_block24` (section 2) | 9.60 | 150.5 GiB | 155665637 |
| `p768_slide16` | 19.08 | 183.4 GiB | 155666108 |
| `p768_block24_g4` (section 2) | 74.6 | 149.0 GiB | 155665637 |
| `p768_slide16_g4`: `p768_slide16` with full global attention in blocks 5, 11, 17 and 23 | 83.9 | 157.9 GiB | 155666108 |

At 384³, without checkpointing, `p384_slide16` takes 1.12x the time of `p384_block24`. Here the ratio is 1.99x.

**Where the time goes** (`probes/slide768/` on /nrs: the trainer's profiler on the real run, and one layer at the
96³ grid; one B300 GPU each, jobs 155666186 to 155666189).

* **With activation checkpointing on, the sliding blocks are not compiled.** They run one operation at a time;
  only the FlexAttention kernel is compiled. Of 19.2 s of GPU time per step, the FlexAttention kernels take 7.2 s
  (48 forward and 24 backward calls), the prefix tokens' global attention 1.3 s, and 10.7 s is ordinary tensor
  work launched by single uncompiled operators: concatenations 2.4 s, copies 2.0 s, multiplications 1.8 s, linear
  layers 1.2 s, layer norms 0.9 s. In 1f's profile (compiled, no checkpointing) all of that ran inside compiled
  graphs.
* **The layer probe shows the same.** Inside the checkpoint wrapper the sliding layer takes 437.5 ms, exactly the
  uncompiled path (307.6 ms forward plus backward, and 130.8 ms for the extra forward pass); compiled and without
  the wrapper it takes 272.4 ms. A variant that hands the mask in crashed inside the wrapper because it ran
  FlexAttention's uncompiled reference code, which only happens when the compiler skips the block.
* **Block windows stay compiled under checkpointing,** by their timings: `p768_block24`'s 9.60 s is 1.25x its 384³
  step scaled by 8, which is the recomputed forward pass and nothing more. That arm was not profiled.
* **It is not the mask rebuild:** 3 ms per build at this grid, and handing the mask in changed the step from
  19.10 to 18.71 s.
* **It is not the grid size as such:** one compiled sliding layer without checkpointing takes 272 ms at 96³, in
  line with the smaller grids of section 3.
* **Why the compiler skips the sliding block: one line.** `layers/common/window_attention.py:334`, in
  `prefix_copies`, calls `math.prod` on a generator expression, which `torch.compile` cannot trace ("Failed to
  convert args/kwargs to proxy", `LocalGeneratorObjectVariable`). It breaks the graph there. Without
  checkpointing that only splits the layer into three compiled pieces. A checkpointed block must be captured as
  one unit, so with a break inside it the compiler runs the whole block uncompiled (`why_skipped.py`, job
  155666229: all 24 graph breaks are this line; block windows never call it).
* **Replacing the line is necessary but not sufficient.** With `math.prod` given a list, the compiler captures
  every variant as one graph with no breaks (job 155666237). But the real run then fails in its first backward
  pass: inside the single graph, FlashAttention-4's backward kernel gets one of the mask's index tensors in a
  non-contiguous form and refuses it ("Expected strides[leading_dim] == 1, but got 3520", job 155666238). The
  break had hidden this by building the mask in a separate piece.
* **Both changes together work.** With the line replaced and the mask built once, before training, and handed in
  (both applied at run time by `run_arm.py`; mia-train is not edited), `p768_slide16` on one GPU runs at 11.57 s
  per step and 150.4 GiB, start-up included (job 155666241), against 19.10 s and 158.5 GiB as mia-train runs it
  today. That is 1.20x the time of `p768_block24` (9.61 s on one GPU); at 384³ without checkpointing the ratio is
  1.12x.
* The +53% that `large_inputs` measured for compiled sliding windows also ran with checkpointing, so it very
  probably has this cause; that run was not repeated.

**Where the memory goes.** In steady state a step of `p768_slide16` peaks at 158.5 GiB, against 150.1 GiB for
`p768_block24` (one GPU each): 8.4 GiB more. The 183.4 GiB above is a start-up peak: on one GPU the sliding arm
reaches 178.6 GiB during its first two steps and stays at 158.5 GiB afterwards.

**After the fix in mia-train (2026-11-22).** Both changes are now in `layers/common/window_attention.py` of the
mia-train checkout, not yet committed: `math.prod` takes a list in `prefix_copies`, and the block mask is built
outside the compiled graph and kept in `_BLOCK_MASKS`. The same arms, run from that checkout on three B300 GPUs
(jobs 155666262 and 155666272); the arms above ran from the snapshot 8af6a8f:

| arm | s per step before | s per step after | peak GPU memory before | after |
|---|---|---|---|---|
| `p768_slide16` | 19.08 | 11.43 | 183.4 GiB | 149.3 GiB |
| `p768_slide16_g4` | 83.9 | 76.0 | 157.9 GiB | 149.3 GiB |
| `p384_slide16` | 1.073 | 0.941 | 148.9 GiB | 142.1 GiB |
| `p384_slide16_g4` (job 155666352) | 1.693 | 1.574 | 146.4 GiB | 140.4 GiB |

* **At 768³ the sliding arm now takes 1.19x the time of `p768_block24`** (9.60 s), against 1.99x before. With
  four global blocks it takes 1.02x the time of `p768_block24_g4` (74.6 s).
* **At 384³, without checkpointing, it gains 12% as well** (7% with four global blocks), and its memory now
  equals the block-window arm's (142.0 GiB). `p384_slide24` in section 3 predates the fix and was not rerun.
* mia-train's unit tier (1517 tests), the tensor-parallel window tests and the GPU tests of sliding windows pass
  with the change. Two tests were added: a checkpointed sliding model compiles as one graph (CPU), and compiles
  and trains in half precision (GPU).

## Where things are

`/nrs/scicompsoft/orhane/mia-train-experiments/attention_variants/`: `jobs/throughput/` (job logs, the derived
configs, `results/`), `runs/throughput/` (the run directories of the short runs), `probes/`.
