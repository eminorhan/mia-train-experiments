# Can a larger, more diverse, and higher-quality pretraining dataset improve downstream evaluation results?

lmd_ssl_v3 pretrains on a new corpus and asks whether the downstream evaluation results improve on
lmd_ssl_v2's. The corpus differs from v2's in three ways:

- **Larger.** Training draws from 325 TVox instead of 92: 0.295 mm³ of tissue instead of 0.070.
- **More diverse.** 78 of its 340 volumes come from outside LMD (v2: 1 of 243): the labs' own volumes, which
  LMD holds only crops of or not at all. The largest volume takes 2.7% of an epoch (v2: 6.5%), and the
  effective number of volumes doubles, from 54 to 110.
- **Higher quality.** v3 addresses two problems of v2's corpus:
  - *Blank crops.* About 8% of v2's training crops were blank, and about 0.3% of v3's are. v3 holds 101 volumes
    to tissue boxes, which leave out most of the blank regions.
  - *Uneven normalization.* v2 divided every 8-bit volume by 255, whatever range its tissue used: some
    volumes filled the input range, others a tenth of it. v3 maps each volume's own p0.1-p99.9 range to
    [0, 1] and clips what lies outside it.

The Data section compares the two corpora in full. The planned runs follow it.

## Data: lmd_ssl_v2 against lmd_ssl_v3

- **v2**: [`lmd_ssl_v2/data/8nm_p256_swe0.4_min1e8_up8_census20261015.yaml`](../lmd_ssl_v2/data/8nm_p256_swe0.4_min1e8_up8_census20261015.yaml),
  the corpus of arms 1a-1c.
- **v3**: [`data_configs/ext_8nm_p256_swe0.4_min1e8_up8_census20261116.yaml`](data_configs/ext_8nm_p256_swe0.4_min1e8_up8_census20261116.yaml),
  a copy of lmd-configs' default config as rebuilt on 2026-11-20. One card per volume, with a picture and a
  neuroglancer view:
  [the volume page](https://fileglancer.int.janelia.org/files/dLLjve-fpPGXhnVi/nrs/scicompsoft/orhane/probes/lmd_census/config_page/lmd_ssl_v3_default.html).

Terms:

- A *volume* is one LMD store or one volume outside LMD. LMD is `/groups/miaai/miaai/lmd-v0.0.1/data`. The
  volumes outside it stay in the folders of the labs and project teams that own them.
- TVox is 10^12 voxels at the finest stored level.
- A *share of an epoch* is the probability that a training crop comes from that volume or group.

| | lmd_ssl_v2 | lmd_ssl_v3 |
| --- | --- | --- |
| **Corpus** | | |
| LMD census | 2026-10-15 (1,078 stores) | 2026-11-16 (1,187 stores) |
| Volumes | 243 | 340 |
| ... from LMD | 242 | 262 |
| ... from outside LMD | 1 (zebra finch j0126, added by hand) | 78 (of 79 listed in lmd-configs `corpus/external.yaml`) |
| Share of an epoch, LMD / outside | 98.6% / 1.4% | 30.0% / 70.0% |
| Whole volumes | 92 TVox | 1,341 TVox |
| Region training draws from | 92 TVox (all but j0126's margins) | 325 TVox (24% of the whole volumes) |
| ... LMD / outside | 91.5 / 0.6 TVox | 19.4 / 305.6 TVox |
| ... as tissue volume | 0.070 mm³ | 0.295 mm³ |
| **EM and ExM** | | |
| Volumes, EM / ExM | 177 / 66 | 250 / 90 |
| Region drawn from, EM / ExM | 53.9 / 38.2 TVox | 283.0 / 42.0 TVox |
| Share of an epoch, EM / ExM | 70.2% / 29.8% | 78.7% / 21.3% |
| **Composition** | | |
| Species | 12 | 15, plus 2 volumes of unrecorded species |
| Largest species, by share of an epoch | mouse 36%, Drosophila 32%, C. elegans 8%, zebrafish 7%, human 6% | mouse 39%, Drosophila 36%, zebrafish 5%, C. elegans 5%, human 4% |
| Largest sources, by share of an epoch | CellMap 26%, flyliconn 17%, liconn 12%, flyem 9% | CellMap 37%, flyem 16%, flyliconn 13%, liconn 7% |
| **Sampling** | | |
| Rule | a volume is drawn in proportion to its size ** 0.4 | same |
| Largest volume | 6.5% of an epoch (FlyID49 mirrorscope lobe, ExM) | 2.7% (zebrafish fish2, EM) |
| Ten largest volumes | 32.6% of an epoch | 19.2% |
| Effective number of volumes | 54 | 110 |
| **Blank crops** | | |
| Volumes held to a tissue box | 1 (j0126) | 101; 24 of them as three to five boxes each (425 config entries) |
| Blank crops | 8.1% (325 of 4,000 drawn as in training) | 0.34% (103 of 30,000) |
| ... of them all zero | 109 of 325 | 21 of 103 |
| **Resolution** | | |
| Patch and voxel size | 256³ at 8 nm | same |
| Size floor, upsampling cap | 10^8 voxels, 8x | same |
| Volumes read on a finer grid | 5 | 7 |
| Share of an epoch read at native resolution | 38% | 50% |
| ... upsampled along some axis | 51% | 38% |
| ... upsampled 2x or more | 42% | 30% |
| **Normalization** | | |
| 8-bit volumes | stored value / 255 (180 volumes, 72% of an epoch) | p0.1-p99.9 window (221 volumes, 69% of an epoch) |
| Input range an 8-bit volume's p1-p99 fills | 11% to 100% of it, median 92% | 38% to 100% of it, median 89% |
| Other dtypes | p0.1-p99.9 window (63 volumes) | p0.1-p99.9 window (119 volumes; four dim ones use p0.5-p99.5) |
| Window measured | over the whole store | inside the tissue box, with fill left out |
| Values past the window | clipped | clipped |
| **Storage** | | |
| Formats | zarr3 240, zarr2 3 | zarr3 270, zarr2 46, N5 19, precomputed 5 |
| dtypes | uint8 180, uint16 59, int16 1, float32 3 | uint8 221, uint16 105, int16 11, float32 3 |
| Expected read per sample | 20 MB | 20 MB |
| miao | 0.4.2 | 0.4.2+fork.5 (reads N5 and precomputed) |
| **Labeled volumes** | | |
| Labeled config (a separate file) | 16 volumes, split 8 / 8 | 21 volumes: v2's without Jiefu's cerebellum, plus 6; no split yet |
| Their images in the SSL corpus | all 16 | 15 as LMD stores; 6 are crops of full volumes the corpus holds |

How to read some rows:

- **Region training draws from**: a volume's tissue box, or the whole volume where it has none. Counted at the
  pyramid level training reads, as the volume page does ("TVox sampled"), it is 283 TVox for v3 and 90 for v2.
- **Size** in the sampling rule is that same region, counted at the level training reads. The boxes of one
  volume share its weight.
- **Effective number of volumes** is 1 / sum(p^2) over the volumes' shares of an epoch: the number of equally
  likely volumes that would give the same concentration.
- **Volumes read on a finer grid** are too small to hold a 256³ patch at 8 nm.
- **Sources** are the third token of a volume's name.

### What became of v2's 243 volumes

- **210 are still entries.** They were 70% of a v2 epoch. 16 of them are now held to a tissue box, among them
  the two FlyID49 mirrorscope crops, which fell from 10.1% of an epoch to 2.0%.
- **30 were replaced by the full volume they were cut from.** They were 29% of a v2 epoch: ten CellMap tissue
  crops (liver-zon-1 and -2 among them), eight hemibrain crops, six fish2 crops and the six CREMI crops, which
  lie in FAFB v14.
- **j0126** is now one of the volumes outside LMD, with a slightly smaller box.
- **2 dropped out**: two pancreas cells (MitoVerse muller21) whose tissue boxes are under the size floor.

New in v3: 52 LMD stores (6% of an epoch) and 77 volumes outside LMD besides j0126. By share of an epoch the
volumes outside LMD are CellMap (39 volumes, 31%), FlyEM (14, 14%), fly LICONN (8, 8%), two cuttlefish volumes
(3%) and 15 others (13%).

### Notes

- **Blank crops.** A crop is blank when its standard deviation as trained is under 0.02. Of 4,000 crops drawn
  as training draws them, 325 were blank in v2 and 14 are in v3 (0.35%). A count over 30,000 crops gives v3
  the same share: 103 blank crops, 0.34%.
  - *In v2* the two FlyID49 mirrorscope crops gave 176 of the 325. Five cultured-cell volumes gave 62 more,
    and those crops were not empty: divided by 255, their contrast range was under a tenth of the input range.
  - *In v3* 21 of the 103 are all zero: fill inside a tissue box. The boxes drawn on 2026-11-20 hold at most
    1% background; some older ones hold more. The other 82 are flat content. The largest groups are empty
    lumen enclosed by tissue (ut21-1413-003 14 crops, mus-liver 9) and flat or dim regions of two worm volumes
    whose resin no mask could separate (adult-d6a 10, mcnally-20240612 8). 15 of the 103 are in ExM volumes.
  - *The two counts do not use the same yardstick*, because "as trained" changed: v3's windows stretch the
    8-bit crops. With v2's normalization applied to v3's 4,000 crops, 74 of them (1.9%) would be blank. So the
    tissue boxes and the new volumes cut the share from 8.1% to 1.9%, and the windows account for the rest.
  - *0.34% understates the featureless crops in v3.* Those 74 include 60 crops that v3's windows lift past
    the threshold. 37 of the 60 come from nine cultured-cell volumes, where a crop is often plain resin, and
    37 of the 60 still have a standard deviation under 0.05. At a threshold of 0.05 the shares are 14.6% for
    v2 and 3.1% for v3 (925 of 30,000 crops).
- **The windows change what the network sees.** In v2 an 8-bit volume reached the network as value / 255, so
  its own p1-p99 range filled between 11% and 100% of the input range, and 15% of the 8-bit crops came from
  volumes that used less than half of it. v3 stretches every volume's p0.1-p99.9 range to [0, 1] (p0.5-p99.5
  for four dim 16-bit volumes, see "Rebuilt with tighter tissue boxes"). An 8-bit volume's p1-p99 range then
  fills between 38% and 100% of the input range, and 0.3% of the 8-bit crops come from volumes that use less
  than half of it. Against value / 255 the stretch is 1.08x for the median 8-bit volume and more than 2x for
  7% of the 8-bit crops. By construction about 0.1% of a volume's measured voxels lie below its window and
  0.1% above it (0.5% for those four); they are clipped to 0 and 1. A v2 encoder therefore sees differently
  scaled inputs on v3 data, and fine-tuning, validation and prediction have to use the windows of pretraining.
- **Restarted on 2026-11-20.** The arms were first submitted on 2026-11-18 with p1-p99 windows and no
  clipping. Unclipped, 7% of the crops held a voxel above 4, and the zero fill of three 16-bit volumes
  reached the network at values down to -32. Only 1f had started, and its loss jumped at step 7,000. All jobs
  were cancelled, every volume's window was measured again and 53 were replaced, the configs were rebuilt with
  the windows described here, and the arms were submitted again. The first launch is kept in
  `/nrs/scicompsoft/orhane/mia-train-experiments/lmd_ssl_v3/archive_p1-99_noclip_20261119/`.
- **Rebuilt with tighter tissue boxes on 2026-11-20.** None of the resubmitted arms had started when they were
  cancelled the same day, because too many crops were still blank or nearly uniform: counted over 30,000
  crops, 1.7% were blank and 8.0% had a standard deviation under 0.05. Three causes were addressed:
  - *Boxes that allowed 5% background.* 48 volumes outside LMD were boxed again. A box now holds at most 1%
    background. A block-shaped volume keeps the one box with the most tissue. A brain-shaped volume gets up to
    five boxes that do not touch, each further box at least 1,000 voxels on every side: 19 volumes, 65 more
    config entries.
  - *Embedding resin inside boxes.* For eight volumes a tissue mask marks the resin as background. Lumen
    enclosed by tissue counts as tissue, so vessel lumen stays. In the optic lobe, blocks that are empty at
    8 nm (4% of its tissue) are background too.
  - *Windows set by rare extreme voxels.* Four dim volumes (tavakoli-mouse2, mus-thymus-2a, adult-d6a, aphid4)
    use p0.5-p99.5 windows. The windows of the other re-boxed volumes were measured again inside the new boxes.

  The region training draws from fell from 375 to 325 TVox. The boxes, masks, counts and a page of the
  re-boxed volumes are in
  `/nrs/scicompsoft/orhane/mia-train-experiments/lmd_ssl_v3/probes/blank_crops/tighter_20261120/`; the files of
  the cancelled launch are in `/nrs/scicompsoft/orhane/mia-train-experiments/lmd_ssl_v3/archive_5pct_boxes_20261120/`.
  The ten arms were submitted with the rebuilt configs on 2026-11-20 at 16:48.
- **What each arm is fed.** Five mosaics per arm, each of 512 training crops as the model receives them, are
  in `/nrs/scicompsoft/orhane/mia-train-experiments/lmd_ssl_v3/probes/galleries_1pct_20261120/`.
- **Evaluation images are in both corpora.** Neither config holds the labeled volumes out. Six labeled crops
  (three CREMI crops, one hemibrain crop, two fish2 crops) are not entries of v3: the corpus holds the full
  volumes they were cut from. Whether each crop lies inside that volume's tissue boxes has not been checked.
- **Reversed contrast.** Nine v3 volumes, 1.8% of an epoch, have bright membranes, judged by eye. All nine are
  16-bit stores, and nothing in the config or in miao inverts them. One of them (funceworm HPF-TAOTO) was
  already in v2. They are kept as they are for now. Twenty other 16-bit EM volumes have not been judged.
- **Assumed expansion factors.** Four LICONN volumes outside LMD, 2.9% of an epoch, use an expansion factor of
  18x that is not confirmed by their labs. Their voxel sizes, and so their resampling to 8 nm, depend on it.
- **Volumes outside LMD are read in place.** Nothing is copied. Fourteen are read through metadata wrappers
  in `/groups/miaai/miaai/lmd-external-wrappers/`, which hold our metadata and links to the owner's arrays.
- **Slow volumes do not slow training.** The two cuttlefish volumes (3.3% of an epoch) are stored in 1024³
  gzip blocks, so one of their crops takes about 6 s to read, against 0.06 s at the median for the other
  volumes. They were paused for that from 2026-11-16 to 2026-11-18. Measured on two B300 GPUs with the base
  run config (2026-11-17), training waits for data 0.04% of the time and runs at 4.0 crops per second, with
  the two volumes and without them: each GPU has eight loader workers, and the step is bound by compute.

### Where the numbers come from

lmd-configs' `describe.py --exact` on both configs, and `compare.py` on its output, in
`/nrs/scicompsoft/orhane/mia-train-experiments/lmd_ssl_v3/probes/data_v2_vs_v3/` (`describe.sh`, `compare.sh`,
`numbers.json`). Two things were adjusted for v2: one store was renamed on disk on 2026-11-13, so the copy
described there names its new path; and the j0126 entry's shape and voxel size come from the comment on it in
the config, because the tool cannot read that store's metadata. v3 was described on 2026-11-20, after the
rebuild with the tighter tissue boxes (`refresh_numbers_rebox.sh` in the same folder).

Three numbers come from elsewhere. v2's blank crops were counted on 2026-10-23 by `census.py` in
`/nrs/scicompsoft/orhane/mia-train-experiments/lmd_ssl_v2/probes/blank_crops/`. v3's were counted on 2026-11-20
in `/nrs/scicompsoft/orhane/mia-train-experiments/lmd_ssl_v3/probes/blank_crops/`: over 30,000 crops by
`rule_census.py` in its `tighter_20261120` folder (`census_built_s*.json`), and over 4,000 crops, as many as
v2's count drew, by `census_v3_clip.py`, which also applies v2's normalization to them; its montages show the
blank ones. The
input range that the 8-bit volumes fill is from `uint8_range.sh` (v2) and `uint8_range_rebox.sh` (v3) in the
`data_v2_vs_v3` folder. v2's uses the p1-p99 windows on record in lmd-configs on 2026-11-17, v3's the p1 and
p99 measured together with the windows on 2026-11-20, inside the new boxes. The loader measurements are in the
`throughput_1a` and `cuttlefish` folders beside them.

## Planned runs

Every run changes one thing about the base config: the cell in bold. The base config is SimMIM on a DINOv3
ViT-L, trained on the default data config described above.

### Data config changes

Algorithm and model stay those of the base config. Each run's data config is in [`data_configs/`](data_configs/).

| Config | Resolution | Input size | Modality | Data | Size exponent | Data config |
| --- | --- | --- | --- | --- | --- | --- |
| Base config | 8 nm | 256³ | EM/ExM | all | 0.4 | [`ext_8nm_p256_swe0.4_min1e8_up8_census20261116.yaml`](data_configs/ext_8nm_p256_swe0.4_min1e8_up8_census20261116.yaml) |
| EM config | 8 nm | 256³ | **EM** | all | 0.4 | [`ext_sel-em_8nm_p256_swe0.4_min1e8_up8_census20261116.yaml`](data_configs/ext_sel-em_8nm_p256_swe0.4_min1e8_up8_census20261116.yaml) |
| ExM config | 8 nm | 256³ | **ExM** | all | 0.4 | [`ext_sel-exm_8nm_p256_swe0.4_min1e8_up8_census20261116.yaml`](data_configs/ext_sel-exm_8nm_p256_swe0.4_min1e8_up8_census20261116.yaml) |
| Resolution variability | **4-16 nm** | 256³ | EM/ExM | all | 0.4 | [`ext_4to16nm_p256_swe0.4_min1e8_up8_census20261116.yaml`](data_configs/ext_4to16nm_p256_swe0.4_min1e8_up8_census20261116.yaml) |
| Native resolution | **native** | 256³ | EM/ExM | all | 0.4 | [`ext_native_p256_swe0.4_min1e8_up8_census20261116.yaml`](data_configs/ext_native_p256_swe0.4_min1e8_up8_census20261116.yaml) |
| Larger inputs | 8 nm | **384³** | EM/ExM | all | 0.4 | [`ext_8nm_p384_swe0.4_min1e8_up8_census20261116.yaml`](data_configs/ext_8nm_p384_swe0.4_min1e8_up8_census20261116.yaml) |
| Data scaling (10%) | 8 nm | 256³ | EM/ExM | **10% of base** | 0.4 | [`ext_data10pct_8nm_p256_swe0.4_min1e8_up8_census20261116.yaml`](data_configs/ext_data10pct_8nm_p256_swe0.4_min1e8_up8_census20261116.yaml) |
| Data scaling (1%) | 8 nm | 256³ | EM/ExM | **1% of base** | 0.4 | [`ext_data1pct_8nm_p256_swe0.4_min1e8_up8_census20261116.yaml`](data_configs/ext_data1pct_8nm_p256_swe0.4_min1e8_up8_census20261116.yaml) |
| Size exp (0.2) | 8 nm | 256³ | EM/ExM | all | **0.2** | [`ext_8nm_p256_swe0.2_min1e8_up8_census20261116.yaml`](data_configs/ext_8nm_p256_swe0.2_min1e8_up8_census20261116.yaml) |
| Size exp (0.8) | 8 nm | 256³ | EM/ExM | all | **0.8** | [`ext_8nm_p256_swe0.8_min1e8_up8_census20261116.yaml`](data_configs/ext_8nm_p256_swe0.8_min1e8_up8_census20261116.yaml) |

A **Cloud data** run, which would use whole volumes read from the cloud, is not planned for this version. We
might try it in the next iteration (v4) of the pretraining runs, once we have settled on the optimized data,
model, and algorithm configs.

### Algorithm and model changes

These runs train on the base data config.

| Config | Algorithm | Model |
| --- | --- | --- |
| Base config | SimMIM | DINOv3 ViT-L |
| DINOv3 algorithm | **DINOv3, 3D version** | DINOv3 ViT-L |
| Larger model | SimMIM | **DINOv3 ViT-H+ (840M parameters)** |
