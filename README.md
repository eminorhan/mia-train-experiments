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

miao 0.5.0 names how a volume is stored with `format` (`zarr2`, `zarr3`, `n5`, `precomputed`) and refuses the
earlier `zarr_version` key; an older miao refuses `format: zarr3`. A data config therefore loads on one side
only. Since 2026-11-26 lmd-configs (46fe95a), mia-train (4b8dcca), mia-evals (9d6c9dc) and myvenv are on miao
0.5.0.

- **Experiments from before that date are records.** Their data configs keep the old format and were not
  converted. The one exception is `mixed_local_cloud_data_reads`, whose data configs have said `format` since
  2026-11-24.
- **New work** uses read-only copies of the new code under `/nrs/scicompsoft/orhane/probes/`: miao
  `miao-pinned/c53052d` and mia-train `mia-train-pinned/4b8dcca`.
- **To run something from an older experiment**, use the code from before the change: the miao pin the
  experiment names (else `miao-pinned/28229ac`, which myvenv held until then) and mia-train
  `mia-train-pinned/5b3aac1`. Put `<miao>/src:<mia-train>/src` first on `PYTHONPATH` and `cd <mia-train>`
  before `src/train.py` or `src/predict.py`. mia-evals' task data in the old format are in
  `mia-evals-pinned/fb08d48/configs/`.
- **Five scoring drivers predict that way** and score with the mia-evals checkout: `lmd_ssl_v1/predict_eval.sh`,
  `lmd_ssl_v2/score.sh`, `lmd_ssl_v2/score_zebrafinch.sh`, `lmd_ssl_v2_cellmap/score.sh` and
  `lm_zebrafish_cell_instance/score.sh`. They keep their miao and their data configs, so what they read does not
  change: miao 0.5.0 would read the three CREMI volumes of `lmd_neuron_instance_v2` at another pyramid level.
- **`gary_comparison/score.sh` and `large_inputs_nisb/score.sh`** name no pin and predict from mia-evals' task
  data, so they run on the checkouts as they are.
- **Any other script** that runs `src/train.py` or `src/predict.py` from the checkouts on an old-format data
  config, with no miao pin, is refused by myvenv's miao. Run it as described above.
- **`lmd_ssl_v3`** keeps its pins (mia-train 8af6a8f, miao 28229ac) and its data configs as they are until its
  arms have finished.

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
