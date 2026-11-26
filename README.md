# mia-train-experiments

Experiments run with [mia-train](https://github.com/AI-HHMI/mia-train). Each experiment has its configs, its
submission and scoring scripts, and a README saying what it asked and what it found. Until 2026-10-17 these lived
in mia-train's `experiments/` folder. They were copied from AI-HHMI/mia-train@580a5a5, and their earlier history
is in that repo.

## Layout

- One folder per experiment.
- Its artifacts (run directories, job logs, predictions, probes) live on
  `/nrs/scicompsoft/orhane/mia-train-experiments/<experiment>/`, under the same name.
- These are working records of runs on the Janelia cluster. Paths are absolute and point at that filesystem, so
  nothing here runs elsewhere without editing.

## How the experiments use mia-train

- Code comes from the mia-train checkout at `/groups/scicompsoft/home/orhane/projects/mia-train` (`MIA_TRAIN` in
  the scripts).
- Jobs `cd` there and run `src/train.py` or `src/predict.py`. Python helpers put `$MIA_TRAIN/src` first on the
  import path.
- A TOML's `config_path` names a data YAML in this repo by its absolute path, so copies of the TOML still find it
  (smoke configs, run directories). `configs/data/...` paths resolve inside mia-train, as before.
- Each run directory records the mia-train commit it ran (`git_commit.txt`). It also keeps copies of the TOML and
  the data YAML.

## miao 0.5.0 (since 2026-11-26)

miao 0.5.0 has a new `format` key (`zarr2`, `zarr3`, `n5`, `precomputed`) and refuses the
earlier `zarr_version` key; older miao versions refuse `format: zarr3`. A data config therefore loads on one side
only. Since 2026-11-26 lmd-configs (46fe95a), mia-train (4b8dcca), mia-evals (9d6c9dc) and myvenv are on miao
0.5.0.

- **Experiments from before that date are records.** Their data configs keep the old format and were not
  converted. The one exception is `mixed_local_cloud_data_reads`, whose data configs have said `format` since
  2026-11-24.
- **New work** uses read-only copies of the new code under `/nrs/scicompsoft/orhane/probes/`: miao
  `miao-pinned/c53052d` and mia-train `mia-train-pinned/a1812c2`. That mia-train commit is the first that
  writes the ground truth for a volume read above pyramid level 0, as miao 0.5.0 reads CREMI.
- **To run something from an older experiment**, use read-only copies of the code from before the change, not
  the checkouts:
  - miao: the copy that the experiment's own scripts point to, for example `miao-pinned/62b3746` in
    `lmd_ssl_v2/submit.sh`. If its scripts point to none, use `miao-pinned/28229ac`, the version myvenv had
    until 2026-11-26.
  - mia-train: `mia-train-pinned/5b3aac1`, the last commit before the change.
  - mia-evals' task data in the old format: `mia-evals-pinned/fb08d48/configs/`.

  Put `<miao copy>/src:<mia-train copy>/src` first on `PYTHONPATH`, and `cd` into the mia-train copy before you
  run `src/train.py` or `src/predict.py`.
- **Five scoring drivers predict that way** and score with the mia-evals checkout: `lmd_ssl_v1/predict_eval.sh`,
  `lmd_ssl_v2/score.sh`, `lmd_ssl_v2/score_zebrafinch.sh`, `lmd_ssl_v2_cellmap/score.sh` and
  `lm_zebrafish_cell_instance/score.sh`. They keep their miao and their data configs, so what they read does not
  change: miao 0.5.0 would read the three CREMI volumes of `lmd_neuron_instance_v2` at another pyramid level.
- **`gary_comparison/score.sh` and `large_inputs_nisb/score.sh`** need no change: they run the checkouts and
  predict from mia-evals' task data, which are in the new format.
- **Any other script** that runs the checkouts with the miao installed in myvenv fails on an old-format data
  config, because miao 0.5.0 refuses it. Run such a script as described above.
- **`lmd_ssl_v3`** keeps the read-only copies its jobs run on (mia-train `8af6a8f`, miao `28229ac`) and its data
  configs as they are until its arms have finished.

## Experiments

- [`attention_variants/`](attention_variants/): windowed and sliding-window attention in place of global attention: how much faster is a large-crop training step?
- [`b300_capability_run/`](b300_capability_run/): how large a volume can a 7B DINOv3 fine-tune on, on 8 B300 nodes?
- [`banis_parity/`](banis_parity/): can a DINOv3 ViT reach the published BANIS baselines on NISB `base`?
- [`data_scaling/`](data_scaling/): how much of NISB `base` does a model actually need?
- [`gary_comparison/`](gary_comparison/): our models against a colleague's, on his volume and his split (hemibrain Ellipsoid Body).
- [`init_comparison/`](init_comparison/): does pretraining help on NISB, and do EM augmentations?
- [`large_inputs/`](large_inputs/): how large an input can a model train on per B300 node, and at what throughput?
- [`large_inputs_nisb/`](large_inputs_nisb/): does labelled-data scaling on NISB pay off at a larger window?
- [`lm_zebrafish_cell_instance/`](lm_zebrafish_cell_instance/): an affinity model for mia-evals' `lm_zebrafish_cell_instance_v1` task.
- [`lmd_ssl_v1/`](lmd_ssl_v1/): does SSL on the lmd corpus beat a pretrained checkpoint?
- [`lmd_ssl_v2/`](lmd_ssl_v2/): does SimMIM on the 2026-10-15 lmd corpus beat the LVD checkpoint for affinity prediction, and do LSDs or a SAM model help?
- [`lmd_ssl_v2_cellmap/`](lmd_ssl_v2_cellmap/): which lmd_ssl_v2 encoder transfers best to CellMap organelle semantic segmentation?
- [`lmd_ssl_v3/`](lmd_ssl_v3/): can a larger, more diverse, and higher-quality pretraining dataset improve downstream evaluation results?
- [`lora_vs_fullft/`](lora_vs_fullft/): can a 3.4%-trainable fine-tune match a full one on NISB?
- [`lsd_aux_v1/`](lsd_aux_v1/): do local shape descriptors help the affinity head?
- [`mixed_local_cloud_data_reads/`](mixed_local_cloud_data_reads/): can a training run read some of its volumes straight from a public bucket?
- [`new_ssl_recipe/`](new_ssl_recipe/): does a two-phase SSL recipe beat plain joint SSL, and does either beat no SSL at all?
- [`promptable_seg_v1/`](promptable_seg_v1/): promptable ("segment this") instance segmentation, v1.
- [`pseudo_labeling/`](pseudo_labeling/): NoisyStudent-style pseudo-labelling from five labelled NISB cubes to a hundred unlabelled ones.
- [`sam_lmd_v1/`](sam_lmd_v1/): does a Segment-Anything model beat affinity prediction on the lmd eval set, and can it use the unlabeled corpus?
- [`simmim_vs_direct/`](simmim_vs_direct/): does SSL pretraining on NISB help downstream instance segmentation?
- [`subpixel_decoder/`](subpixel_decoder/): is the trilinear head what limits boundary sharpness?
