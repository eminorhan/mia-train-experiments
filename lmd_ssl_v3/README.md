# Can a larger, more diverse, and higher-quality pretraining dataset improve downstream evaluation results?

lmd_ssl_v3 pretrains on a new corpus and asks whether the downstream evaluation results improve on
lmd_ssl_v2's. The corpus differs from v2's in three ways:

- **Larger.** Training draws from 375 TVox instead of 92: 0.323 mm³ of tissue instead of 0.070.
- **More diverse.** 78 of its 340 volumes come from outside LMD (v2: 1 of 243): the labs' own volumes, which
  LMD holds only crops of or not at all. The largest volume takes 2.7% of an epoch (v2: 6.5%), and the
  effective number of volumes doubles, from 54 to 106.
- **Higher quality.** v3 addresses two problems of v2's corpus:
  - *Blank crops.* About 8% of v2's training crops were blank, and about 1.2% of v3's are. v3 holds 101 volumes
    to tissue boxes, which leave out most of the blank regions.
  - *Uneven normalization.* v2 divided every 8-bit volume by 255, whatever range its tissue used: some
    volumes filled the input range, others a tenth of it. v3 maps each volume's own p1-p99 range to [0, 1].

The Data section compares the two corpora in full. The planned runs follow it.

## Data: lmd_ssl_v2 against lmd_ssl_v3

- **v2**: [`lmd_ssl_v2/data/8nm_p256_swe0.4_min1e8_up8_census20261015.yaml`](../lmd_ssl_v2/data/8nm_p256_swe0.4_min1e8_up8_census20261015.yaml),
  the corpus of arms 1a-1c.
- **v3**: [`data_configs/ext_8nm_p256_swe0.4_min1e8_up8_win1-99_noclip_census20261116.yaml`](data_configs/ext_8nm_p256_swe0.4_min1e8_up8_win1-99_noclip_census20261116.yaml),
  a copy of lmd-configs' default config as rebuilt on 2026-11-18. One card per volume, with a picture and a
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
| Share of an epoch, LMD / outside | 98.6% / 1.4% | 28.8% / 71.2% |
| Whole volumes | 92 TVox | 1,341 TVox |
| Region training draws from | 92 TVox (all but j0126's margins) | 375 TVox (28% of the whole volumes) |
| ... LMD / outside | 91.5 / 0.6 TVox | 19.4 / 355.8 TVox |
| ... as tissue volume | 0.070 mm³ | 0.323 mm³ |
| **EM and ExM** | | |
| Volumes, EM / ExM | 177 / 66 | 250 / 90 |
| Region drawn from, EM / ExM | 53.9 / 38.2 TVox | 332.1 / 43.1 TVox |
| Share of an epoch, EM / ExM | 70.2% / 29.8% | 79.4% / 20.6% |
| **Composition** | | |
| Species | 12 | 15, plus 2 volumes of unrecorded species |
| Largest species, by share of an epoch | mouse 36%, Drosophila 32%, C. elegans 8%, zebrafish 7%, human 6% | mouse 39%, Drosophila 36%, zebrafish 5%, C. elegans 5%, human 4% |
| Largest sources, by share of an epoch | CellMap 26%, flyliconn 17%, liconn 12%, flyem 9% | CellMap 37%, flyem 16%, flyliconn 13%, liconn 7% |
| **Sampling** | | |
| Rule | a volume is drawn in proportion to its size ** 0.4 | same |
| Largest volume | 6.5% of an epoch (FlyID49 mirrorscope lobe, ExM) | 2.7% (D. yakuba CNS, EM) |
| Ten largest volumes | 32.6% of an epoch | 19.9% |
| Effective number of volumes | 54 | 106 |
| **Blank crops** | | |
| Volumes held to a tissue box | 1 (j0126) | 101; 5 of them as 5 boxes each (360 config entries) |
| Blank crops | 8.1% (325 of 4,000 drawn as in training) | 1.2% (49 of 4,000) |
| ... of them all zero | 109 | 31 |
| **Resolution** | | |
| Patch and voxel size | 256³ at 8 nm | same |
| Size floor, upsampling cap | 10^8 voxels, 8x | same |
| Volumes read on a finer grid | 5 | 7 |
| Share of an epoch read at native resolution | 38% | 51% |
| ... upsampled along some axis | 51% | 37% |
| ... upsampled 2x or more | 42% | 29% |
| **Normalization** | | |
| 8-bit volumes | stored value / 255 (180 volumes, 72% of an epoch) | p1-p99 window (221 volumes, 70% of an epoch) |
| Input range an 8-bit volume's p1-p99 fills | 11% to 100% of it, median 92% | all of it, for every volume |
| Other dtypes | p0.1-p99.9 window (63 volumes) | p1-p99 window (119 volumes), about half as wide |
| Window measured | over the whole store | inside the tissue box (ten boxes made in October keep whole-store windows) |
| Values past the window | clipped | kept (`normalize_clip: false`) |
| **Storage** | | |
| Formats | zarr3 240, zarr2 3 | zarr3 270, zarr2 46, N5 19, precomputed 5 |
| dtypes | uint8 180, uint16 59, int16 1, float32 3 | uint8 221, uint16 105, int16 11, float32 3 |
| Expected read per sample | 20 MB | 20 MB |
| miao | 0.4.2 | 0.4.2+fork.5 (reads N5 and precomputed) |
| **Labeled volumes** | | |
| Labeled config (a separate file) | 16 volumes, split 8 / 8 | 21 volumes: v2's without Jiefu's cerebellum, plus 6; no split yet |
| Their images in the SSL corpus | all 16 | all 21: 15 as LMD stores, 6 inside full volumes |

How to read some rows:

- **Region training draws from**: a volume's tissue box, or the whole volume where it has none. Counted at the
  pyramid level training reads, as the volume page does ("TVox sampled"), it is 332 TVox for v3 and 90 for v2.
- **Size** in the sampling rule is that same region, counted at the level training reads. The boxes of one
  volume share its weight.
- **Effective number of volumes** is 1 / sum(p^2) over the volumes' shares of an epoch: the number of equally
  likely volumes that would give the same concentration.
- **Volumes read on a finer grid** are too small to hold a 256³ patch at 8 nm.
- **Sources** are the third token of a volume's name.

### What became of v2's 243 volumes

- **210 are still entries.** They were 70% of a v2 epoch. 16 of them are now held to a tissue box, among them
  the two FlyID49 mirrorscope crops, which fell from 10.1% of an epoch to 1.9%.
- **30 were replaced by the full volume they were cut from.** They were 29% of a v2 epoch: ten CellMap tissue
  crops (liver-zon-1 and -2 among them), eight hemibrain crops, six fish2 crops and the six CREMI crops, which
  lie in FAFB v14.
- **j0126** is now one of the volumes outside LMD, with a slightly smaller box.
- **2 dropped out**: two pancreas cells (MitoVerse muller21) whose tissue boxes are under the size floor.

New in v3: 52 LMD stores (6% of an epoch) and 77 volumes outside LMD besides j0126. By share of an epoch the
volumes outside LMD are CellMap (39 volumes, 32%), FlyEM (14, 15%), fly LICONN (8, 7%), two cuttlefish volumes
(3%) and 15 others (14%).

### Notes

- **Blank crops.** A crop is blank when its standard deviation as trained is under 0.02. Of 4,000 crops drawn
  as training draws them, 325 were blank in v2 and 49 are in v3.
  - *In v2* the two FlyID49 mirrorscope crops gave 176 of the 325. Five cultured-cell volumes gave 62 more,
    and those crops were not empty: divided by 255, their contrast range was under a tenth of the input range.
  - *In v3* 31 of the 49 are all zero, 10 are empty gel in ExM volumes, and the other 8 are fill or plain
    resin in EM volumes.
  - *The two counts do not use the same yardstick*, because "as trained" changed: v3's windows stretch every
    crop. With v2's normalization applied to v3's crops, 159 of them (4.0%) would be blank. So the tissue
    boxes and the new volumes halve the share, and the windows account for the rest.
  - *1.2% understates the featureless crops in v3.* The 110 crops in between are of two kinds. Most are plain
    resin or lumen whose noise the window lifts past the threshold: 69 have a standard deviation under 0.05.
    The others are tissue of low stored contrast that v2's rule flattened. At a threshold of 0.05 the shares
    are 14.6% for v2 and 4.4% for v3.
- **The windows change what the network sees.** In v2 an 8-bit volume reached the network as value / 255, so
  its own p1-p99 range filled between 11% and 100% of the input range, and 15% of the 8-bit crops came from
  volumes that used less than half of it. In v3 every volume is stretched to its own p1-p99 range: by 1.2x
  for the median 8-bit volume, and by more than 2x for 27% of the 8-bit crops. By construction about 1% of a
  volume's measured voxels fall below 0 and 1% above 1. A v2 encoder therefore sees differently scaled inputs
  on v3 data, and fine-tuning, validation and prediction have to use the windows of pretraining.
- **Evaluation images are in both corpora.** Neither config holds the labeled volumes out.
- **Reversed contrast.** Nine v3 volumes, 1.7% of an epoch, have bright membranes, judged by eye. All nine are
  16-bit stores, and nothing in the config or in miao inverts them. One of them (funceworm HPF-TAOTO) was
  already in v2. They are kept as they are for now. Twenty other 16-bit EM volumes have not been judged.
- **Assumed expansion factors.** Four LICONN volumes outside LMD, 2.8% of an epoch, use an expansion factor of
  18x that is not confirmed by their labs. Their voxel sizes, and so their resampling to 8 nm, depend on it.
- **Volumes outside LMD are read in place.** Nothing is copied. Fourteen are read through metadata wrappers
  in `/groups/miaai/miaai/lmd-external-wrappers/`, which hold our metadata and links to the owner's arrays.
- **Slow volumes do not slow training.** The two cuttlefish volumes (3.2% of an epoch) are stored in 1024³
  gzip blocks, so one of their crops takes about 6 s to read, against 0.06 s at the median for the other
  volumes. They were paused for that from 2026-11-16 to 2026-11-18. Measured on two B300 GPUs with the base
  run config (2026-11-17), training waits for data 0.04% of the time and runs at 4.0 crops per second, with
  the two volumes and without them: each GPU has eight loader workers, and the step is bound by compute.

### Where the numbers come from

lmd-configs' `describe.py --exact` on both configs, and `compare.py` on its output, in
`/nrs/scicompsoft/orhane/mia-train-experiments/lmd_ssl_v3/probes/data_v2_vs_v3/` (`describe.sh`, `compare.sh`,
`numbers.json`). Two things were adjusted for v2: one store was renamed on disk on 2026-11-13, so the copy
described there names its new path; and the j0126 entry's shape and voxel size come from the comment on it in
the config, because the tool cannot read that store's metadata.

Three numbers come from elsewhere. v2's blank crops were counted on 2026-10-23 by `census.py` in
`/nrs/scicompsoft/orhane/mia-train-experiments/lmd_ssl_v2/probes/blank_crops/`. v3's were counted the same way
on 2026-11-18 by `census_v3.py` in `/nrs/scicompsoft/orhane/mia-train-experiments/lmd_ssl_v3/probes/blank_crops/`,
which also applies v2's normalization to the same crops; its montages show the crops. The input range that
v2's 8-bit volumes filled is from `uint8_range.sh` in the `data_v2_vs_v3` folder, which reads the p1-p99
windows on record in lmd-configs today. The loader measurements are in the `throughput_1a` and `cuttlefish`
folders beside them.

## Planned runs

Every run changes one thing about the base config: the cell in bold. The base config is SimMIM on a DINOv3
ViT-L, trained on the default data config described above.

### Data config changes

Algorithm and model stay those of the base config. Each run's data config is in [`data_configs/`](data_configs/).

| Config | Resolution | Input size | Modality | Data | Size exponent | Data config |
| --- | --- | --- | --- | --- | --- | --- |
| Base config | 8 nm | 256³ | EM/ExM | all | 0.4 | [`ext_8nm_p256_swe0.4_min1e8_up8_win1-99_noclip_census20261116.yaml`](data_configs/ext_8nm_p256_swe0.4_min1e8_up8_win1-99_noclip_census20261116.yaml) |
| EM config | 8 nm | 256³ | **EM** | all | 0.4 | [`ext_sel-em_8nm_p256_swe0.4_min1e8_up8_win1-99_noclip_census20261116.yaml`](data_configs/ext_sel-em_8nm_p256_swe0.4_min1e8_up8_win1-99_noclip_census20261116.yaml) |
| ExM config | 8 nm | 256³ | **ExM** | all | 0.4 | [`ext_sel-exm_8nm_p256_swe0.4_min1e8_up8_win1-99_noclip_census20261116.yaml`](data_configs/ext_sel-exm_8nm_p256_swe0.4_min1e8_up8_win1-99_noclip_census20261116.yaml) |
| Resolution variability | **4-16 nm** | 256³ | EM/ExM | all | 0.4 | [`ext_4to16nm_p256_swe0.4_min1e8_up8_win1-99_noclip_census20261116.yaml`](data_configs/ext_4to16nm_p256_swe0.4_min1e8_up8_win1-99_noclip_census20261116.yaml) |
| Native resolution | **native** | 256³ | EM/ExM | all | 0.4 | [`ext_native_p256_swe0.4_min1e8_up8_win1-99_noclip_census20261116.yaml`](data_configs/ext_native_p256_swe0.4_min1e8_up8_win1-99_noclip_census20261116.yaml) |
| Larger inputs | 8 nm | **384³** | EM/ExM | all | 0.4 | [`ext_8nm_p384_swe0.4_min1e8_up8_win1-99_noclip_census20261116.yaml`](data_configs/ext_8nm_p384_swe0.4_min1e8_up8_win1-99_noclip_census20261116.yaml) |
| Data scaling (10%) | 8 nm | 256³ | EM/ExM | **10% of base** | 0.4 | [`ext_data10pct_8nm_p256_swe0.4_min1e8_up8_win1-99_noclip_census20261116.yaml`](data_configs/ext_data10pct_8nm_p256_swe0.4_min1e8_up8_win1-99_noclip_census20261116.yaml) |
| Data scaling (1%) | 8 nm | 256³ | EM/ExM | **1% of base** | 0.4 | [`ext_data1pct_8nm_p256_swe0.4_min1e8_up8_win1-99_noclip_census20261116.yaml`](data_configs/ext_data1pct_8nm_p256_swe0.4_min1e8_up8_win1-99_noclip_census20261116.yaml) |
| Size exp (0.2) | 8 nm | 256³ | EM/ExM | all | **0.2** | [`ext_8nm_p256_swe0.2_min1e8_up8_win1-99_noclip_census20261116.yaml`](data_configs/ext_8nm_p256_swe0.2_min1e8_up8_win1-99_noclip_census20261116.yaml) |
| Size exp (0.8) | 8 nm | 256³ | EM/ExM | all | **0.8** | [`ext_8nm_p256_swe0.8_min1e8_up8_win1-99_noclip_census20261116.yaml`](data_configs/ext_8nm_p256_swe0.8_min1e8_up8_win1-99_noclip_census20261116.yaml) |

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
