"""lmd_ssl_v2_cellmap: every data config and run config, from mia-evals' two CellMap organelle tasks.

Run it in a bsub job: it reads every fit crop's labels and opens every store through miao and mia-train.

    bsub -P miaai -n 4 -W 2:00 -o /nrs/scicompsoft/orhane/mia-train-experiments/lmd_ssl_v2_cellmap/jobs/make_configs_%J.log \
        bash lmd_ssl_v2_cellmap/make_configs.sh

Writes, next to this file:

  data/train_<task>_<res>.yaml    the task's fit crops, without bounding boxes and with labels outside each crop
                                  filled with -1 (ignored), minus the crops miao cannot centre a 256^3 window on.
                                  A crop holding only composite ids (whole organelles painted without their parts)
                                  trains through `label_sets` at weight 0.25.
  data/predict_<task>_<res>.yaml  the task's test crops, each widened by half a window of context, shifted inward at
                                  the image faces, and a whole number of half-window steps long at the pyramid level
                                  miao reads, so the prediction lattice covers every crop voxel. No label_key: the
                                  scorer reads the truth from the store.
  <arm>.toml, <arm>_fewshot.toml  one run per arm and task (ARMS)

<task> is `large` (cellmap_organelle_semantic) or `fewshot` (cellmap_organelle_semantic_fewshot), <res> `8nm` or
`native` (each dataset's raw voxel size). It prints what each filter removed and every test crop's coverage.
"""

from __future__ import annotations

import csv
import hashlib
import subprocess
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.name
MIA_EVALS = Path("/groups/scicompsoft/home/orhane/projects/mia-evals")
TASKS = {"large": "cellmap_organelle_semantic", "fewshot": "cellmap_organelle_semantic_fewshot"}
WINDOW = 256
COMPOSITE_WEIGHT = 0.25
# classes.csv as mia-evals' task builder read it (the hash in its data YAML headers).
CLASSES_SHA256 = "61fea48f0278f95281b752855b46d69ec367b1c367df24d979e0c16608c01958"

LVD = "/groups/miaai/miaai/pretrained_models/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
V2_RUNS = Path("/nrs/scicompsoft/orhane/mia-train-experiments/lmd_ssl_v2/runs")
CHECKPOINTS = {
    "simmim_lvd": V2_RUNS / "lmd2__1a_simmim_lvd_b32_20261017_185845/checkpoints/step_100000",
    "simmim_scratch": V2_RUNS / "lmd2__1b_simmim_scratch_b32_20261017_205909/checkpoints/step_100000",
    "affinity_simmim_lvd": V2_RUNS / "lmd2__2f_affinity_unetr_simmim_lvd_20261020_200659/checkpoints/step_100000",
}
SCHEDULES = {
    "large": dict(steps=100_000, warmup=5_000, every=25_000),
    "fewshot": dict(steps=25_000, warmup=2_000, every=5_000),
}
# Series 1: the encoder's starting point, at the default recipe (8 nm, cross-entropy, seed 0). Series 2: one change
# each to 1e, the arm whose encoder lmd_ssl_v2 found best for affinities.
ARMS = {
    "1a_scratch": dict(init=None, note="encoder from random initialisation"),
    "1b_lvd": dict(init="lvd", note="encoder from the released DINOv3 LVD-1689M weights"),
    "1c_simmim_lvd": dict(init="simmim_lvd", note="encoder from lmd_ssl_v2 1a_simmim_lvd_b32 (LVD, then SimMIM on lmd)"),
    "1d_simmim_scratch": dict(
        init="simmim_scratch", note="encoder from lmd_ssl_v2 1b_simmim_scratch_b32 (SimMIM on lmd from scratch)"),
    "1e_affinity_simmim_lvd": dict(
        init="affinity_simmim_lvd", note="encoder from lmd_ssl_v2 2f (LVD, SimMIM on lmd, then affinities)"),
    "2a_affinity_simmim_lvd_native": dict(
        init="affinity_simmim_lvd", res="native", note="1e at each dataset's native voxel size"),
    "2b_affinity_simmim_lvd_dice": dict(init="affinity_simmim_lvd", dice=1.0, note="1e with cross-entropy + Dice 1:1"),
    "2c_affinity_simmim_lvd_seed1": dict(init="affinity_simmim_lvd", seed=1, note="1e with seed 1"),
}


def git_commit(repo: Path) -> str:
    out = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(repo), "status", "--short", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip()
    return out.stdout.strip() + (" (dirty)" if dirty else "")


# ----------------------------------------------------------------------------------------------------- classes

def read_classes(stores: list[str]) -> tuple[dict[str, int], dict[int, list[int]]]:
    """CellMap's ids (name -> id) and composites (id -> its leaf ids), the same in every store."""
    texts = {Path(s, "classes.csv").read_bytes() for s in stores}
    assert len(texts) == 1, "the CellMap stores disagree about classes.csv"
    text = texts.pop()
    assert hashlib.sha256(text).hexdigest() == CLASSES_SHA256, "classes.csv is not the one the tasks were built from"
    ids, composites = {}, {}
    for name, cid, members in csv.reader(text.decode().splitlines()):
        ids[name] = int(cid)
        if members:
            composites[int(cid)] = [int(m) for m in members.split(",")]
    leaves = set(ids.values()) - set(composites)
    assert all(set(m) <= leaves for m in composites.values()), "a composite lists another composite"
    return ids, composites


def has_leaves(volume: dict, composites: dict[int, list[int]]) -> bool:
    """Whether any voxel of the crop carries a leaf id (0 is unannotated, composites are sets)."""
    import zarr

    counts = np.bincount(np.asarray(zarr.open_array(f"{volume['path']}/{volume['label_key']}/s0", mode="r")[:]).ravel(),
                         minlength=256)
    return bool(np.delete(counts[1:], [c - 1 for c in composites]).any())


# ----------------------------------------------------------------------------------------------------- miao

def miao_config(volumes: list[dict], output_axes: str):
    from miao.config import MiaoConfig

    return MiaoConfig(volumes=volumes, resolutions=[[8.0, 8.0, 8.0]], output_axes=output_axes,
                      patch_size=[WINDOW] * 3)


def resolved(volume: dict, output_axes: str):
    """miao's resolved entry for one volume (its levels, read shapes, offsets), or the error refusing it."""
    from miao.dataset import VolumeDataset

    try:
        return VolumeDataset(miao_config([volume], output_axes))._volumes[0]
    except ValueError as error:
        return str(error).splitlines()[0]


def xyz(values: list) -> list:
    """A zyx triple (mia-evals' task YAMLs) in the training YAMLs' xyz order."""
    return list(reversed(values))


# ----------------------------------------------------------------------------------------------------- training

def training_volumes(task: str, res: str, fit: list[dict], composite_only: set[str]) -> tuple[list[dict], list[str]]:
    volumes, refused = [], []
    for v in fit:
        entry = {"name": v["name"], "path": v["path"], "image_key": v["image_key"], "label_key": v["label_key"],
                 "zarr_version": v["zarr_version"], "label_fill": -1}
        if res == "native":
            entry["resolutions"] = [xyz(r) for r in v["resolutions"]]
        if v["name"] in composite_only:
            entry["weight"] = COMPOSITE_WEIGHT
        info = resolved(entry, "lcxyz")
        if isinstance(info, str):
            refused.append(f"{v['name']}: {info}")
            continue
        if res == "native":   # the override must reach miao in storage order as the raw's own voxel size
            assert np.allclose(info.img_level_voxels[info.scales.chosen_levels[0]], v["resolutions"][0]), v["name"]
        volumes.append(entry)
    return volumes, refused


# ----------------------------------------------------------------------------------------------------- prediction

def prediction_box(crop: list[list[int]], info) -> list[list[int]]:
    """The level-0 box (storage order) predicting one test crop: the read level's voxels touching the crop plus half a
    window each side, a whole number of half-window steps long, shifted inward at the image's faces.

    In miao's frame level-0 voxel j is reference coordinate j and voxel i of the read level is `i * f - o`, `f` wide
    (`VolumeGrid._at_image_level` inverts the box conversion made here). Whole steps because a resampled axis drops a
    lattice's tail rather than ending it flush.
    """
    factor = np.asarray(info.scales.relative_scale_factors[0], float)
    offsets = info.scales.image_center_offsets
    offset = np.asarray(offsets[0], float) if offsets is not None else np.zeros(3)
    reads = np.asarray(info.scales.read_shapes[0], int)
    level = int(info.scales.chosen_levels[0])
    # The read level's own array bounds the box, not level 0's: where an image is not a whole number of the
    # level's voxels long, its last voxel there reaches past level 0's end, and a crop at that face lies partly in
    # it. miao reads it in training as well; it accepts a box out to there, as its windows only have to fit the
    # array it reads.
    image = np.asarray(info.image_meta.scales[level].shape, int)[info.img_spatial_idx]
    box = []
    for (lo, hi), f, o, r, size in zip(crop, factor, offset, reads, image, strict=True):
        read = int(r + r % 2)                               # VolumeGrid's even read
        step = read // 2
        touch_lo = int(np.floor((lo - 0.5 + o) / f - 0.5 - 1e-9)) + 1
        touch_hi = int(np.ceil((hi - 0.5 + o) / f + 0.5 + 1e-9))
        first, last = 0, int(size)
        span = touch_hi - touch_lo
        n = read + int(np.ceil(span / step)) * step         # >= the crop plus half a window each side
        while n > last - first and n - step >= max(read, span):
            n -= step
        # miao's bounding-box rule wants a box at least one voxel longer than a window (the open "bbox needs
        # window + 1" issue), so a one-window lattice gets one voxel of slack, which VolumeGrid leaves past its tail.
        slack = 1 if n == read else 0
        if n + slack > last - first or n < span:
            raise SystemExit(f"no whole-step lattice of {read} fits the crop {lo}:{hi} in an image of {size}")
        start = min(max(touch_lo - (n - span) // 2, first), last - n - slack)
        box.append([int(np.floor((start - 0.5) * f + 0.5 - o + 1e-9)),
                    int(np.ceil((start + n + slack - 0.5) * f + 0.5 - o - 1e-9))])
    return box


def check_coverage(config, name: str, crop: list[list[int]]) -> str:
    """Build the real prediction lattice and confirm it covers the crop; one line for the log."""
    from prediction.grid import VolumeGrid

    grid = VolumeGrid(config, name, [WINDOW] * 3, cover_box=True)
    covered = []
    for (a, b), f, o in zip(grid.native_box(), grid.level_factor, grid.level_offset, strict=True):
        covered.append([a * f - o - f / 2, b * f - o - f / 2])   # reference coordinates
    inside = all(c0 <= lo - 0.5 + 1e-6 and hi - 0.5 <= c1 + 1e-6
                 for (lo, hi), (c0, c1) in zip(crop, covered, strict=True))
    assert inside, f"{name}: lattice {grid.native_box()} at level {grid.image_level} misses crop {crop}"
    return (f"level {grid.image_level}, read {grid.read}, voxel {[round(v, 3) for v in grid.effective_voxel]} nm, "
            f"{len(grid.tiles)} tiles, flush {grid.covers_full_box}")


def prediction_volumes(task: str, res: str, test: list[dict]) -> list[dict]:
    from miao.config import MiaoConfig

    volumes = []
    for v in test:
        entry = {"name": v["name"], "path": v["path"], "image_key": v["image_key"], "zarr_version": v["zarr_version"]}
        if res == "native":
            entry["resolutions"] = v["resolutions"]
        info = resolved(entry, "lczyx")
        assert not isinstance(info, str), f"{v['name']}: {info}"
        assert info.img_spatial_axes == "zyx", v["name"]
        entry["bounding_box"] = prediction_box(v["bounding_box"], info)
        config = MiaoConfig(volumes=[entry], resolutions=[[8.0, 8.0, 8.0]], output_axes="lczyx",
                            patch_size=[WINDOW] * 3)
        print(f"   {v['name']}: crop {v['bounding_box']} -> box {entry['bounding_box']}; "
              f"{check_coverage(config, v['name'], v['bounding_box'])}")
        volumes.append(entry)
    return volumes


# ----------------------------------------------------------------------------------------------------- writing

class _Dumper(yaml.SafeDumper):
    """mia-evals' layout: one block per volume, boxes and voxel sizes inline."""


def _sequence(dumper: yaml.SafeDumper, data: list) -> yaml.Node:
    inline = all(not isinstance(x, dict) for x in data)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=inline)


_Dumper.add_representer(list, _sequence)


def write_yaml(path: Path, header: str, output_axes: str, volumes: list[dict]) -> None:
    body = yaml.dump({"resolutions": [[8.0, 8.0, 8.0]], "output_axes": output_axes,
                      "patch_size": [WINDOW] * 3, "volumes": volumes},
                     Dumper=_Dumper, sort_keys=False, width=120, default_flow_style=False)
    path.write_text("".join(f"# {line}\n".replace("# \n", "#\n") for line in header.strip().splitlines()) + body)


INIT = {
    None: "",
    "lvd": f"""
[init]
path = "{LVD}"
inflate_2d_to_3d = true
skip = ["rope_embed."]    # a 2D checkpoint says nothing about z
strict = true
""",
    **{key: f"""
[init]
path = "{path}"
prefix = "model."         # the encoder only
strict = true
""" for key, path in CHECKPOINTS.items()},
}

TOML = """\
# lmd_ssl_v2_cellmap {arm}{suffix}: {note}.
# semantic_seg on {task_name}'s fit crops at {res_text}. Generated by make_configs.py; edit there.
experiment_name = "lmd2cm__{arm}{suffix}"

[model]
name = "dinov3_vit3d"
img_size = 256
patch_size = 8
in_chans = 1
embed_dim = 1024
depth = 24
num_heads = 16
n_storage_tokens = 4
layerscale_init = 1.0e-05
mask_k_bias = true
pos_embed_rope_dtype = "fp32"
pos_embed_rope_type = "vanilla"
drop_path_rate = 0.0
use_fa4 = false
attn_max_score = 1e5
{init}
[algorithm]
name = "semantic_seg"
num_classes = 75          # CellMap ids 0-74
ignore_labels = [0]       # unannotated
decoder = "unetr"         # lmd_ssl_v2 2a's head
decoder_skip_layers = [7, 15, 23]
decoder_widths = [16, 32, 64]
decoder_image_skip = true
decoder_zero_init_output = false
dice_weight = {dice}

[algorithm.label_sets]    # a composite id is any of its parts (classes.csv)
{label_sets}

[data]
name = "miao_volumes"
config_path = "{data}"
samples_per_epoch = 1000000
defer_image_ops = true

[trainer]
max_steps = {steps}
batch_size = 1
lr = 0.0003
warmup_steps = {warmup}
min_lr_ratio = 0.001
weight_decay = 0.05
grad_clip_norm = 1.0
layerwise_lr_decay = 1.0
patch_embed_lr_mult = 1.0
lr_schedule = "linear"
precision = "bf16"
compile = true            # needs TORCHINDUCTOR_LAYOUT_OPTIMIZATION=0 (submit.sh)
activation_checkpointing = false
log_every = 1000
val_every = 0
checkpoint_every = {every}
num_workers = 8
seed = {seed}

[parallelism]
dp_replicate = 1
dp_shard = 8
tp = 1

[augment]
rotate = "inplane"
drop_slice_prob = 0.05
shift_slice_prob = 0.05
shift_magnitude = 10
intensity = true
mul_intensity = 0.1
add_intensity = 0.1
noise_scale = 0.5
"""


def main() -> None:
    import sys

    sys.path.insert(0, "/groups/scicompsoft/home/orhane/projects/mia-train/src")
    import miao

    evals_commit = git_commit(MIA_EVALS)
    print(f"[make_configs] miao {miao.__file__}, mia-evals {evals_commit}, mia-train "
          f"{git_commit(Path('/groups/scicompsoft/home/orhane/projects/mia-train'))}")
    splits = {task: {split: yaml.safe_load(open(MIA_EVALS / "configs" / name / "data" / f"{split}.yaml"))["volumes"]
                     for split in ("fit", "test")} for task, name in TASKS.items()}
    ids, composites = read_classes(sorted({v["path"] for s in splits.values() for vs in s.values() for v in vs}))
    names = {cid: name for name, cid in ids.items()}
    label_sets = "\n".join(f"{c} = {m}  # {names[c]}" for c, m in sorted(composites.items()))
    fit_all = {v["name"]: v for s in splits.values() for v in s["fit"]}
    composite_only = {name for name, v in fit_all.items() if not has_leaves(v, composites)}
    print(f"[make_configs] {len(composite_only)} of {len(fit_all)} fit crops hold only 0 and composite ids")

    today = date.today().isoformat()
    for task, name in TASKS.items():
        for res in ("8nm", "native"):
            res_text = "8 nm" if res == "8nm" else "each dataset's native voxel size"
            fit = splits[task]["fit"]
            volumes, refused = training_volumes(task, res, fit, composite_only)
            weighted = sum("weight" in v for v in volumes)
            print(f"\n== train {task} {res}: {len(volumes)} of {len(fit)} fit crops "
                  f"({weighted} composite-only at weight {COMPOSITE_WEIGHT}); refused {len(refused)}")
            for line in refused:
                print(f"   refused {line}")
            write_yaml(HERE / "data" / f"train_{task}_{res}.yaml", f"""
{EXPERIMENT} training crops: {name}'s fit crops at {res_text} ({len(volumes)} of {len(fit)}).
Generated by make_configs.py on {today} from mia-evals {evals_commit}
configs/{name}/data/fit.yaml; edit the generator, not this file.
No bounding boxes: each window is centred on a crop and labels outside it are -1 (label_fill), which
semantic_seg ignores. Crops holding only composite ids weigh {COMPOSITE_WEIGHT} ({weighted} here).
Refused by miao (no centred {WINDOW}^3 window inside the image): {len(refused)}.""",
                       "lcxyz", volumes)

            test = splits[task]["test"]
            print(f"\n== predict {task} {res}: {len(test)} test crops")
            pred = prediction_volumes(task, res, test)
            write_yaml(HERE / "data" / f"predict_{task}_{res}.yaml", f"""
{EXPERIMENT} prediction crops: {name}'s {len(pred)} test crops at {res_text}.
Generated by make_configs.py on {today} from mia-evals {evals_commit}
configs/{name}/data/test.yaml; edit the generator, not this file.
Each bounding_box (zyx, level-0 voxels) is the crop plus half a window of context, shifted inward at the
image faces and a whole number of half-window steps long at the level miao reads, so the lattice covers
every crop voxel. No label_key: predict with --no-truth; the scorer reads the store's labels.""",
                       "lczyx", pred)

    for arm, spec in ARMS.items():
        res = spec.get("res", "8nm")
        for task, name in TASKS.items():
            suffix = "" if task == "large" else "_fewshot"
            text = TOML.format(
                arm=arm, suffix=suffix, task_name=name, note=spec["note"],
                res_text="8 nm" if res == "8nm" else "native voxel size", init=INIT[spec["init"]],
                dice=spec.get("dice", 0.0), label_sets=label_sets,
                data=HERE / "data" / f"train_{task}_{res}.yaml", seed=spec.get("seed", 0), **SCHEDULES[task])
            (HERE / f"{arm}{suffix}.toml").write_text(text)
    print(f"\n[make_configs] wrote {2 * len(ARMS)} run configs")
    check_run_configs()


def check_run_configs() -> None:
    """Every TOML parses, finds its data config and [init] checkpoint, and semantic_seg accepts its label settings
    (on a toy encoder with a linear head: the label checks do not depend on the model)."""
    import tomllib

    from algorithms.semantic_seg import SemanticSegmentation
    from models.vit import ViT3D

    labels = ("num_classes", "ignore_labels", "label_sets", "dice_weight")
    for path in sorted(HERE.glob("*.toml")):
        config = tomllib.load(open(path, "rb"))
        init = config.get("init", {}).get("path")
        assert init is None or Path(init).exists(), f"{path.name}: no checkpoint at {init}"
        assert Path(config["data"]["config_path"]).is_file(), path.name
        toy = ViT3D(img_size=(32, 32, 32), patch_size=(8, 8, 8), in_channels=1, embed_dim=32, depth=1, num_heads=4)
        algorithm = SemanticSegmentation(toy, input_axes="lcxyz", decoder="linear",
                                         **{k: config["algorithm"][k] for k in labels})
        assert int(algorithm.accepts.sum()) > 75, path.name   # the composites' rows hold their parts
    print(f"[make_configs] {len(list(HERE.glob('*.toml')))} run configs parse, find their data and checkpoints, and "
          "semantic_seg accepts their label settings")


if __name__ == "__main__":
    main()
