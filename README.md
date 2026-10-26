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

## Experiments

- [`b300_capability_run/`](b300_capability_run/): how large a volume can a 7B DINOv3 fine-tune on, on 8 B300 nodes?
- [`banis_parity/`](banis_parity/): can a DINOv3 ViT reach the published BANIS baselines on NISB `base`?
- [`data_scaling/`](data_scaling/): how much of NISB `base` does a model actually need?
- [`gary_comparison/`](gary_comparison/): our models against a colleague's, on his volume and his split (hemibrain Ellipsoid Body).
- [`init_comparison/`](init_comparison/): does pretraining help on NISB, and do EM augmentations?
- [`large_inputs/`](large_inputs/): how large an input can a model train on per B300 node, and at what throughput?
- [`large_inputs_nisb/`](large_inputs_nisb/): does labelled-data scaling on NISB pay off at a larger window?
- [`lm_zebrafish_cell_instance/`](lm_zebrafish_cell_instance/): an affinity model for mia-evals' `lm_zebrafish_cell_instance` task.
- [`lmd_ssl_v1/`](lmd_ssl_v1/): does SSL on the lmd corpus beat a pretrained checkpoint?
- [`lmd_ssl_v2/`](lmd_ssl_v2/): does SimMIM on the 2026-10-15 lmd corpus beat the LVD checkpoint for affinity prediction, and do LSDs or a SAM model help?
- [`lmd_ssl_v2_cellmap/`](lmd_ssl_v2_cellmap/): which lmd_ssl_v2 encoder transfers best to CellMap organelle semantic segmentation?
- [`lora_vs_fullft/`](lora_vs_fullft/): can a 3.4%-trainable fine-tune match a full one on NISB?
- [`lsd_aux_v1/`](lsd_aux_v1/): do local shape descriptors help the affinity head?
- [`new_ssl_recipe/`](new_ssl_recipe/): does a two-phase SSL recipe beat plain joint SSL, and does either beat no SSL at all?
- [`promptable_seg_v1/`](promptable_seg_v1/): promptable ("segment this") instance segmentation, v1.
- [`pseudo_labeling/`](pseudo_labeling/): NoisyStudent-style pseudo-labelling from five labelled NISB cubes to a hundred unlabelled ones.
- [`sam_lmd_v1/`](sam_lmd_v1/): does a Segment-Anything model beat affinity prediction on the lmd eval set, and can it use the unlabeled corpus?
- [`simmim_vs_direct/`](simmim_vs_direct/): does SSL pretraining on NISB help downstream instance segmentation?
- [`subpixel_decoder/`](subpixel_decoder/): is the trilinear head what limits boundary sharpness?
