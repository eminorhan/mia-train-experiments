"""Turn one lmd_ssl_v2 prediction of the zebrafinch 11 um cube into the labelling mia-evals' identity route scores.

    python zf_convert.py <prediction.zarr> <out.zarr>

The prediction is predict.py's artifact on its own lattice -- the arms' 8 nm one, or the store's 20 x 9 x 9 nm grid --
covering mia-evals' zebrafinch_neurite_tracing_11um cube with context. An affinity prediction is segmented by exact
mutex watershed with mia-evals' own code, as the task's mws_blockwise route segments a native prediction:
`segment_block`, one block over the lattice voxels the cube's voxels fall in, repulsive stride 1, the FFN neuropil mask
placed on the lattice by `lattice_mask` and every edge with an end outside it dropped. A promptable arm's
segment-everything prediction is a labelling already, restricted to the same mask. Either labelling is relabelled into
6-connected components on its own lattice, inside the cube, and only then goes to the store's voxels -- each cube voxel
takes the lattice voxel its centre falls in, by the OME scale and translation both sides record, the rule
`lattice_mask` uses -- and is restricted to the neuropil mask on the store's grid. It is not relabelled again there:
a store grid coarser than the lattice (20 nm sections against 8 nm) would cut a connected segment into pieces that no
longer touch face to face, and score the cuts as splits (the user's choice, 2026-10-27). On a lattice that is the
store's grid this is exactly the relabel configs/zebrafinch_neurite_tracing_11um/identity.toml asks for.

The output is a single-level OME-Zarr (zarr v2) group laid out like FFN's reference artifact, with the prediction's
`run` and `step`, so mia-evals names the record after the run. It is built at <out.zarr>.partial and renamed when
complete; an existing <out.zarr> is refused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cc3d
import numpy as np
import yaml
import zarr

EVALS = Path("/groups/scicompsoft/home/orhane/projects/mia-evals")
sys.path.insert(0, str(EVALS / "src"))

from artifact import open_artifact  # noqa: E402
from postprocess.mws_blockwise import (  # noqa: E402
    ArraySource,
    BlockwiseMutexWatershed,
    _artifact_geometry,
    _ome_level,
    lattice_mask,
    segment_block,
)

TASK = EVALS / "configs" / "zebrafinch_neurite_tracing_11um"
MASK_KEY = "labels/neuropil_mask"   # the task's mws_blockwise.toml `mask`, relative to the store
CONTEXT = 0.75                      # of a patch per side: the context the task's docs ask a prediction to have


def cube() -> tuple[str, str, np.ndarray, np.ndarray]:
    """(store, image key, low, high) of the task's test volume, in z, y, x store voxels."""
    data = yaml.safe_load((TASK / "data" / "test.yaml").read_text())
    if data["output_axes"] != "lczyx":
        raise SystemExit(f"{TASK}/data/test.yaml is not lczyx; its box would not be z, y, x")
    (volume,) = data["volumes"]
    box = np.asarray(volume["bounding_box"], dtype=np.int64)
    return volume["path"], volume["image_key"], box[:, 0], box[:, 1]


def lattice_indices(low: np.ndarray, high: np.ndarray, store_scale: list[float], store_shift: list[float],
                    scale: list[float], shift: list[float], shape: tuple[int, ...]) -> list[np.ndarray]:
    """Per axis, the lattice voxel each store voxel centre in [low, high) falls in."""
    out = []
    for axis in range(len(low)):
        centre = store_shift[axis] + np.arange(low[axis], high[axis], dtype=np.float64) * store_scale[axis]
        j = np.floor((centre - shift[axis]) / scale[axis] + 0.5).astype(np.int64)
        if j.min() < 0 or j.max() >= shape[axis]:
            raise SystemExit(f"axis {axis}: the cube maps to lattice voxels [{j.min()}, {j.max()}], "
                             f"outside the prediction's {shape[axis]}")
        out.append(j)
    return out


def git_state(path: Path) -> str:
    head = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip()
    return head + (" (dirty)" if dirty else "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("prediction", type=Path, help="predict.py's <volume>.zarr")
    parser.add_argument("out", type=Path, help="the labelling to write, <dir>/zebrafinch_11um_test.zarr")
    args = parser.parse_args()
    started = time.time()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; delete it to rebuild")

    store, image_key, low, high = cube()
    prediction = open_artifact(args.prediction)
    attrs = prediction.attrs
    kind = attrs.get("kind")
    if kind not in ("affinity", "instances"):
        raise SystemExit(f"{args.prediction} is kind={kind!r}; expected affinity or instances")
    if attrs.get("source_path") != store or attrs.get("source_image_key") != image_key:
        raise SystemExit(f"{args.prediction} predicts {attrs.get('source_path')}:{attrs.get('source_image_key')}, "
                         f"not the task's {store}:{image_key}")
    if int(attrs.get("image_level", 0)) != 0:
        raise SystemExit(f"{args.prediction} was read from level {attrs.get('image_level')}, not level 0")
    names, scale, shift = _artifact_geometry(prediction)
    _, store_names, store_scale, store_shift = _ome_level(Path(store) / image_key)
    if names != store_names:
        raise SystemExit(f"{args.prediction} has axes {names}, the store {store_names}")
    array = zarr.open_array(str(prediction.array_path), mode="r")
    lattice_shape = tuple(int(s) for s in array.shape[-3:])

    index = lattice_indices(low, high, store_scale, store_shift, scale, shift, lattice_shape)
    box_low = [int(j.min()) for j in index]
    box_high = [int(j.max()) + 1 for j in index]
    size = tuple(h - lo for lo, h in zip(box_low, box_high, strict=True))
    need = [int(np.ceil(CONTEXT * p)) for p in attrs["patch"]]
    for axis, name in enumerate(names):
        before, after = box_low[axis], lattice_shape[axis] - box_high[axis]
        if min(before, after) < need[axis]:
            raise SystemExit(f"axis {name}: the prediction has {before} / {after} lattice voxels of context around "
                             f"the cube, fewer than {need[axis]} (3/4 of its patch)")
    window = tuple(slice(lo, h) for lo, h in zip(box_low, box_high, strict=True))
    voxel = " x ".join(f"{v:g}" for v in scale)
    info: dict = {
        "prediction": str(args.prediction.resolve()),
        "prediction_kind": kind,
        "prediction_attrs": {k: attrs.get(k) for k in ("code_version", "model_axes", "offsets", "effective_voxel_nm",
                                                        "scale", "native_box", "patch", "data_config")},
        "lattice_voxel_nm": scale,
        "lattice_translation_nm": shift,
        "lattice_box": [[lo, h] for lo, h in zip(box_low, box_high, strict=True)],
        "cube": [[int(lo), int(h)] for lo, h in zip(low, high, strict=True)],
        "context_lattice_voxels": [[box_low[a], lattice_shape[a] - box_high[a]] for a in range(len(names))],
    }

    mask = lattice_mask(prediction, Path(store) / MASK_KEY, tuple(box_low), size)
    if kind == "affinity":
        BlockwiseMutexWatershed().check_artifact(prediction)       # its declared offsets are the route's
        affinities = np.asarray(array[(slice(None), *window)])
        if affinities.shape[0] != 6:
            raise SystemExit(f"{args.prediction} has {affinities.shape[0]} channels; mutex watershed needs 6")
        t = time.time()
        data = segment_block(ArraySource(affinities=affinities, mask_array=mask), (0, 0, 0), size, size, 1)
        del affinities
        lattice_labels = data["labels"]                            # 0 outside the mask
        edges, growths, live = (int(v) for v in data["stats"])
        info["mws"] = {"edges": edges, "growths": growths, "live_mutexes": live, "clusters": int(data["clusters"]),
                       "lattice_voxels": int(np.prod(size)), "seconds": round(time.time() - t, 1)}
        del data
        made = (f"exact mutex watershed (mia-evals segment_block: one block over the cube, repulsive stride 1, the "
                f"FFN neuropil mask inside the watershed) of {attrs['run']} step {attrs['step']}'s affinities on "
                f"their {voxel} nm lattice")
    else:
        if prediction.background_id != 0:
            raise SystemExit(f"{args.prediction} has background_id {prediction.background_id}, expected 0")
        lattice_labels = np.array(array[window])
        lattice_labels[~mask] = 0
        made = f"{attrs['run']} step {attrs['step']}'s segment-everything labelling on its {voxel} nm lattice"
    del mask
    lattice_labels, lattice_segments = cc3d.connected_components(lattice_labels, connectivity=6, return_N=True,
                                                                 out_dtype=np.uint32)
    info["segments_on_lattice_in_cube"] = int(lattice_segments)

    labels = lattice_labels[np.ix_(*[j - lo for j, lo in zip(index, box_low, strict=True)])]
    del lattice_labels

    partial = args.out.with_name(args.out.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    group = zarr.open_group(str(partial), mode="w", zarr_format=2)
    translation = [float(store_shift[a] + low[a] * store_scale[a]) for a in range(len(names))]
    group.attrs.update(multiscales=[{
        "version": "0.4",
        "name": args.out.stem,
        "axes": [{"name": n, "type": "space", "unit": "nanometer"} for n in names],
        "datasets": [{"path": "s0", "coordinateTransformations": [
            {"type": "scale", "scale": [float(v) for v in store_scale]},
            {"type": "translation", "translation": translation},
        ]}],
    }])
    # The cube on the store's grid, through the same function the scorer's route masks with.
    stub = SimpleNamespace(path=partial, origin=tuple(int(v) for v in low), array_path=partial / "s0")
    store_mask = lattice_mask(stub, Path(store) / MASK_KEY, tuple(int(v) for v in low),
                              tuple(int(v) for v in high - low))
    labels[~store_mask] = 0
    present = np.unique(labels)
    info["segments"] = int(present.size - int(present[0] == 0))        # lattice segments some store voxel takes
    info["mask_fraction"] = round(float(store_mask.mean()), 6)
    level = group.create_array(name="s0", shape=labels.shape, dtype=labels.dtype, chunks=(64, 128, 128))
    level[:] = labels
    script = Path(__file__).resolve()
    info.update(script=str(script), script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
                mia_evals_commit=git_state(EVALS), seconds=round(time.time() - started, 1),
                written=time.strftime("%Y-%m-%d %H:%M:%S"))
    group.attrs.update(
        kind="instances",
        background_id=0,
        origin=[int(v) for v in low],
        covers_full_box=True,
        mask=MASK_KEY,
        run=attrs["run"],
        run_dir=attrs.get("run_dir"),
        step=attrs["step"],
        source_path=store,
        source_image_key=image_key,
        convention=(f"{made}, relabelled into 6-connected components on that lattice inside the cube and the FFN "
                    f"neuropil mask, then taken to the store's {' x '.join(f'{v:g}' for v in store_scale)} nm voxels "
                    "by the lattice voxel each centre falls in and restricted to the mask there, with no second "
                    "relabel on the store's grid (lmd_ssl_v2/zf_convert.py)"),
        zf_convert=info,
    )
    partial.rename(args.out)
    print(json.dumps({"out": str(args.out), **{k: info[k] for k in ("segments", "segments_on_lattice_in_cube",
                                                                    "lattice_box", "seconds")},
                      **({"mws": info["mws"]} if "mws" in info else {})}), flush=True)


if __name__ == "__main__":
    main()
