# Can a training run read some of its volumes straight from a public bucket?

Several large public EM volumes are not on the Janelia cluster (MANC, the male CNS at 8 nm, H01, MICrONS
minnie65, ...). Their buckets can be read without a download: tensorstore, which miao reads through, opens
a bucket path directly. This experiment tests that in real training runs, with part of the windows read
from buckets and the rest from local disk.

Every arm: lmd_ssl_v2's `1b_simmim_scratch` recipe (SimMIM from random init, 3D DINOv3 ViT-L, patch 8,
256^3 crops at 8 nm, one B300 node, `dp_shard = 8`, batch 1 per rank, 8 loader workers per rank),
shortened to 5,000 steps. Arms with the same data and seed draw the same crops at every step, so their
curves compare step by step.

## Arms

| arm | what | differs from |
| --- | --- | --- |
| `1a_optic_local` | the lmd_ssl_v3 corpus plus the FlyEM optic-lobe EM, read from `/nrs/flyem/optic-lobe` | — |
| `1b_optic_bucket` | the optic lobe read from the public bucket `gs://flyem-optic-lobe` | 1a: where the optic lobe is read from |
| `1c_multi_bucket` | the lmd_ssl_v3 corpus plus six public EM volumes, each read from its bucket | 1a: the added volumes |

- **1a against 1b** is the exact control. The optic lobe is the one large volume that is both on the
  cluster and in a bucket, as the same data. The two arms train on identical samples (checked, see below),
  so a difference in step time or data waiting comes from reading the bucket. The optic lobe supplies
  half of the windows in both arms.
- **1c** is the realistic case: six volumes from six buckets (FlyEM's, Google's, and MICrONS' copy on
  Amazon), at their standard weights, 21.8% of the windows together. Its corpus differs from 1a's, so its loss does not compare with 1a's;
  its step time and data waiting do.

What to compare: `data_wait_frac` and `data_wait_frac_max` (logged every 100 steps), the step time, and
for 1a against 1b the loss curves.

    bash mixed_local_cloud_data_reads/submit.sh <arm>              # a 20-step smoke, then the real run after it
    bash mixed_local_cloud_data_reads/submit.sh --no-smoke <arm>   # the real run only
    bash mixed_local_cloud_data_reads/submit.sh --dry-run <arm>    # write the job script, print the bsub line
    SRC_FILE=<copy> bash mixed_local_cloud_data_reads/make_data.sh [optic|buckets] # rewrite the data configs (the script says how to get the copy)

Runs, job logs and probes land in `/nrs/scicompsoft/orhane/mia-train-experiments/mixed_local_cloud_data_reads/`.

## Data

- **The data configs** are written by `make_data.sh`. Each is lmd-configs' lmd_ssl_v3 default corpus config
  (8 nm, p1-99 windows, commit 1009c6d) with four changes:
  - since 2026-11-23, `tensorstore_context` sets how long a bucket read is retried ("Retry budget" below);
  - since 2026-11-24, every volume's `zarr_version` key is written as `format`, the key's name in miao
    since v0.5.0;
  - one lmd volume is left out, `exm-mouse-liconn-RID16-retina-spinningdisk/crop-001_RID16_18x`. Its
    store was renamed on 2026-11-13, after that config's census, so its path no longer exists;
  - volumes are appended: the optic lobe in `data/v3_plus_optic_lobe_local.yaml` (1a) and
    `data/v3_plus_optic_lobe_bucket.yaml` (1b), which differ only in that volume's `path` and
    `image_key`; six bucket volumes in `data/v3_plus_buckets.yaml` (1c).

  The corpus is 247 volumes in 1a and 1b (246 lmd volumes and the optic lobe) and 252 in 1c.
- **The optic-lobe volume** (1a, 1b): Neuroglancer precomputed, uint8, 8 nm, 36864 x 55296 x 45056 voxels.
  - `bounding_box`: its 5% tissue box (12.5 teravoxels), from
    `/nrs/scicompsoft/orhane/probes/large_volumes/tissue_boxes/out/optic_lobe.json`.
  - `normalize_min` / `normalize_max`: 0 and 253, the p1 and p99 inside the box, measured with
    lmd-configs' own functions, block count and seed (150 blocks of 128^3).
  - `weight`: 20.15, so the optic lobe supplies half of the windows. At the default weight of 1 it would
    supply 4.7% (size weighting with exponent 0.4).
- **The bucket volumes of 1c**, all Neuroglancer precomputed, uint8, each with a tissue box, a p1-p99
  window measured the same way from the bucket, and the default weight:

  | volume | read from | read at | tissue box from | window | share |
  | --- | --- | --- | --- | --- | --- |
  | optic lobe | `gs://flyem-optic-lobe/grayscale-clahe-jpeg` | 8 nm | as above | 0, 253 | 3.9% |
  | male CNS | `gs://flyem-male-cns/em/em-clahe-jpeg` | 8 nm | the 5% box of the 16 nm copy on disk, doubled (`.../tissue_boxes/out/malecns_16nm.json`) | 0, 253 | 6.6% |
  | hemibrain | `gs://neuroglancer-janelia-flyem-hemibrain/emdata/raw/jpeg` | 8 nm | the 5% box of the on-disk export (`.../hemibrain_full.json`), turned into the bucket's orientation | 58, 232 | 4.6% |
  | H01 | `gs://h01-release/data/20210601/4nm_raw` | 8 x 8 x 33 nm | the box proposed on 2026-11-09 (`probes/large_volumes/h01_microns_crops`) | 0, 255 | 1.1% |
  | minnie65 | `https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/em` (the copy on Amazon) | 8 x 8 x 40 nm | the 2026-11-09 box, counted from the volume's first voxel | 111, 154 | 1.1% |
  | MANC | `gs://flyem-vnc-2-26-213dba213ef26e094c16c860ae7f4be0/v3_emdata_clahe_xy/jpeg` | 8 nm | made from the bucket with the tissue-box tool (`probes/bucket_boxes_windows/` on /nrs) | 0, 253 | 4.6% |

  The male CNS box overlaps the optic lobe's, which is a corner of the same tissue, and test regions of
  our tasks lie inside the hemibrain and the male CNS. This arm tests the data path; it is not a corpus
  proposal.
- **miao**: branch `feature/remote-paths` of our fork, commit 156fa79, put first on `PYTHONPATH` by
  `submit.sh` from the read-only copy `/nrs/scicompsoft/orhane/probes/miao-pinned/156fa79`. Since
  2026-11-24 the branch is upstream miao v0.5.0 plus three commits:
  - a volume's `path` may be a URL (082226e);
  - `tensorstore_context` is the one config key for tensorstore's context, in place of `cache_bytes` and
    `file_io_concurrency` (b1c6b88);
  - a loader worker says so when a read takes long (156fa79, "Waiting for a bucket" below).

  The runs under "Results" used e962737, the same branch on an older base. Training needs no change in
  mia-train. Prediction does: `src/prediction/grid.py` still reads `config.cache_bytes` and each volume's
  `zarr_version`, so `predict.py` does not run with this commit. Predict with the copy of commit 28229ac
  (`miao-pinned/28229ac`) until mia-train moves to the new keys.
- **Bucket check at the start of every job**: `check_buckets.py` reads the metadata of each bucket volume
  in the run's config, so a node that cannot reach a bucket fails at once.
- **Stalled bucket requests**: since 2026-11-23 every job sets `TENSORSTORE_CURL_LOW_SPEED_TIME_SECONDS=60`
  (`submit.sh`). tensorstore then gives up on a bucket request that received nothing for 60 s and sends it
  again. Without the setting such a request waits forever. The runs under "Results" ran without it.
- **Retry budget**: since 2026-11-23 the data configs set `tensorstore_context` so that tensorstore retries
  a failed bucket request 680 times before the read fails and ends the run. That is about 6 hours for a
  bucket that refuses connections (31 s + 675 x 32 s), and longer for one that goes silent, because each
  attempt then first waits the 60 s above. tensorstore's default is 32 retries, about 15 minutes, which
  the runs under "Results" had.
- **Waiting for a bucket**: tensorstore prints nothing while it retries, and the trainer logs only when a
  step finishes. So since 2026-11-23 a loader worker that has waited 30 s for a read prints one line to
  the job's error log, and again each time the wait has doubled (60 s, 120 s, ...):
  `[miao] still waiting for a read from volume <name> (<path>) after 30 s`.

## Measured before the runs (2026-11-15)

- **One 256^3 window** takes 0.5-1.0 s from a bucket and 0.04 s from /nrs
  (`/nrs/scicompsoft/orhane/probes/large_volumes/remote_read/bench.py`).
- **A full node's load**: 64 processes reading back to back pulled 63-127 windows per second from each of
  five buckets (male CNS, MANC, hemibrain, H01, minnie65). A standard run uses 14.5. No window took longer
  than 2.6 s. One window downloads 5-11 MB. The workers wrote nothing to disk (same folder, `bench64.py`).
- **miao on the real buckets** (`/nrs/scicompsoft/orhane/probes/miao-dev/remote_paths/bucket_check.py`):
  six optic-lobe windows read from /nrs and from the bucket are identical arrays, EM and labels. The male
  CNS (JPEG and N5), H01 and minnie65 buckets read correctly, and one loader mixed a local zarr crop, the
  local optic lobe and the male CNS bucket.
- **Configs of 1a and 1b** (`probes/config_check/` on /nrs, job 155402993): both arms hold the same 247
  volumes with the same shares. Of 48 draws with the same seeds, 24 came from the optic lobe, and all 48
  samples are identical between the arms. One optic-lobe window took 0.03 s from /nrs and 0.54 s from
  the bucket.
- **Config of 1c** (same folder, `config_check_1c.py`, job 155403440): 252 volumes; the six bucket volumes
  have the shares above. Twelve windows drawn from each as trained: none blank except 1 of 12 from MANC;
  median read time 0.51-0.87 s per window.

## Results

**Smoke runs** (20 steps on one B300 GPU, 2026-11-15): all three finished. The B300 nodes reached the
buckets: the optic-lobe bucket in 1b's smoke (node i08u02), all six in 1c's (node i03u02). The losses of
1a and 1b agree step by step:

| step | 1a loss | 1b loss |
| --- | --- | --- |
| 5 | 0.355095 | 0.355128 |
| 10 | 0.183270 | 0.183262 |
| 15 | 0.252624 | 0.252604 |
| 20 | 0.203019 | 0.202993 |

**Real runs** (2026-11-15, one B300 node each; jobs 155403306, 155403308, 155403452; table from
`probes/results/summarize.py` on /nrs). Data wait is the share of the step time spent waiting for data,
logged every 100 steps: averaged over the 8 ranks, and for the slowest rank.

| arm | steps | s per step (median) | data wait, mean over ranks: median / max | data wait, slowest rank: median / max | val loss at 2500, 5000 | wall |
| --- | --- | --- | --- | --- | --- | --- |
| `1a_optic_local` | 5000 | 0.522 | 0.113% / 0.123% | 0.121% / 0.483% | 0.1701, 0.1653 | 47.6 min |
| `1b_optic_bucket` | 5000 | 0.521 | 0.110% / 0.117% | 0.117% / 0.156% | 0.1722, 0.1662 | 47.5 min |
| `1c_multi_bucket` | 5000 | 0.519 | 0.116% / 0.126% | 0.120% / 0.140% | 0.1408, 0.1421 | 47.4 min |

- **Speed and data waiting**: reading half of the windows from the bucket changed neither. The half is
  the optic lobe's sampling share, 0.5001, set by its `weight` (see "Data") and printed in each run's
  job log. At that share arm 1b drew about 20,000 of its 40,000 windows from the bucket. That is the
  expected number: the runs do not count windows per volume.
- **Loss, 1a against 1b**: the single-step losses rise and fall together at every logged step, as they
  must when both arms see the same batches (step 900: 0.1148 and 0.1220; step 1200: 0.1426 and 0.1421).
  They agree to four decimals at step 100. Afterwards they differ by 0.9% at the median, by 14.6% at the
  most (step 600), and by at most 3% after step 1500. The mean over the last ten logged steps is 0.1500
  and 0.1510. How much two identical runs on different nodes differ was not measured (a repeat of 1a
  would show it), so the cause of these differences is not established.
- **Arm 1c**: six bucket volumes supplying 21.8% of the windows left the step time and the data waiting
  where arm 1a has them. The shares are the ones in the table under "Data"; the run's job log prints
  them. At those shares the run drew about 8,700 of its 40,000 windows from the buckets (expected, not
  counted): about 2,600 from the male CNS, 1,800 each from the hemibrain and MANC, 1,600 from the optic
  lobe, and 440 each from H01 and minnie65. The job reached all six buckets at its start, finished all
  5,000 steps, and its error output holds nothing but the usual warnings. Its losses do not compare
  with 1a's, because its corpus differs.

## Open

- Traffic: arm 1b downloaded about 180 GB from the optic-lobe bucket (20,000 windows of about 9 MB) and arm
  1c about 70 GB over its six buckets. Both figures are estimates from the window sizes of the load test,
  not measurements of these runs. A bucket's owner pays for its traffic; who that is for the FlyEM buckets
  is not confirmed.
- A repeat of arm 1a, which would show how much two identical runs differ in loss, has not been run.
