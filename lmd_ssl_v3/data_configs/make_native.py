#!/usr/bin/env python3
"""Write lmd_ssl_v3's native-resolution data config from the base config.

The base config reads every volume at 8 nm. The native config holds the same volumes, boxes, windows and weights,
and reads each volume at its own stored (level-0) voxel size: every entry gets a `resolutions` override equal to
that voxel size, so a 256^3 patch is 256^3 stored voxels and miao resamples nothing.

What follows from that, and is written into the config's header:
  - An entry whose volume (or tissue box) is thinner than a patch on some axis at its stored voxel size cannot
    give a patch without upsampling, so it is left out.
  - mia-train's in-plane rotation needs a sample's x and y voxel sizes to be equal. A few stores record x and y
    sizes that differ by a fraction of a percent; those two are written as the smaller one. miao then still
    reads 256 stored voxels on both axes, so nothing is resampled.

Each entry is constructed in miao twice: as the base has it, to take miao's own level-0 voxel size, and with the
override, to check that it reads the finest level at the patch's shape. Run it as a cluster job.

    python make_native.py BASE.yaml OUT.yaml
"""
import contextlib
import hashlib
import io
import sys
from datetime import date
from pathlib import Path

import numpy as np
import yaml
from miao.config import MiaoConfig
from miao.dataset import VolumeDataset

INPLANE_TOLERANCE = 0.005      # x and y voxel sizes closer than this (relative) are written as one value
RULE = "# " + "-" * 75
AFTER_RESOLUTIONS = ("normalize_max", "normalize_min", "normalize_clip", "weight")


def construct(vol, probe_globals):
    """One entry alone in miao: its resolved volume, or None and why it does not hold a patch."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            ds = VolumeDataset(MiaoConfig(volumes=[vol], **probe_globals))
    except ValueError as e:                                  # how miao refuses a window that does not fit
        return None, str(e).split("\n")[0]
    vi = ds._volumes[0]
    if not np.all(vi.max_center - vi.min_center > 0):
        return None, "admits a single patch centre"
    return vi, ""


def with_resolutions(vol, resolutions):
    """The entry with its `resolutions` set, placed before the window and weight keys."""
    out, placed = {}, False
    for key, value in vol.items():
        if key == "resolutions":
            continue
        if key in AFTER_RESOLUTIONS and not placed:
            out["resolutions"], placed = resolutions, True
        out[key] = value
    if not placed:
        out["resolutions"] = resolutions
    return out


def main():
    base_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    base = yaml.safe_load(base_path.read_text())
    volumes = base.pop("volumes")
    probe_globals = {k: v for k, v in base.items() if k != "size_weighting_exponent"}
    out_axes = [a for a in base["output_axes"] if a in "xyz"]
    patch = base["patch_size"][0]

    kept, dropped, equalized = [], [], []
    for i, vol in enumerate(volumes, 1):
        if i % 50 == 0:
            print(f"  [{i}/{len(volumes)}] {len(kept)} kept", flush=True)
        vi, why = construct(vol, probe_globals)
        if vi is None:
            raise SystemExit(f"{vol['name']} does not construct as the base config has it: {why}")
        finest = dict(zip(vi.img_spatial_axes, (float(v) for v in vi.finest_voxel_size)))
        native = [finest[a] for a in out_axes]
        x, y = out_axes.index("x"), out_axes.index("y")
        if native[x] != native[y] and abs(native[x] - native[y]) <= INPLANE_TOLERANCE * max(native[x], native[y]):
            equalized.append((vol["name"], native[x], native[y]))
            native[x] = native[y] = min(native[x], native[y])
        new = with_resolutions(vol, [native])
        vi, why = construct(new, probe_globals)
        if vi is None:
            if "weight" in vol:
                raise SystemExit(f"{vol['name']} is one box of a multi-box volume and does not hold a native "
                                 f"patch ({why}): the weights of its other boxes would need recomputing.")
            box = vol.get("bounding_box")
            sides = "x".join(str(hi - lo) for lo, hi in box) if box else ""
            dropped.append((vol["name"], native, sides, why))
            continue
        level = vi.scales.chosen_levels[0]
        if not np.array_equal(vi.img_level_voxels[level], vi.finest_voxel_size):
            raise SystemExit(f"{vol['name']}: reads level {level}, not its finest level")
        if not np.array_equal(vi.scales.read_shapes[0], np.asarray(vi.read_shape)):
            raise SystemExit(f"{vol['name']}: reads {vi.scales.read_shapes[0].tolist()} stored voxels for a "
                             f"{vi.read_shape} patch, so it would be resampled")
        kept.append(new)

    sha = hashlib.sha256(base_path.read_bytes()).hexdigest()[:16]
    lines = [
        '# lmd_ssl_v3 data config: Native resolution (README, "Planned runs"). Derived on '
        f"{date.today().isoformat()} by make_native.py",
        "# (in this folder) from the base config,",
        f"#   {base_path.name}  sha256:{sha}",
        "# The base's volumes, boxes, windows and weights, with one change: every volume is read at its own stored",
        f"# (level-0) voxel size instead of at 8 nm, through a per-volume `resolutions`. A {patch}^3 patch is "
        f"{patch}^3 stored",
        "# voxels and nothing is resampled. miao still draws a volume in proportion to size ** exponent, with size",
        "# counted at the level it reads, which here is level 0 for every volume.",
        "#",
        f"# {len(kept)} of the base's {len(volumes)} entries. {len(dropped)} are left out: at its stored voxel "
        "size the volume, or its tissue",
        f"# box, is thinner than a {patch}^3 patch on some axis. They are named at the end of this header.",
    ]
    if equalized:
        lines += [
            "#",
            f"# {len(equalized)} entries store x and y voxel sizes that differ by under "
            f"{100 * INPLANE_TOLERANCE:g}%. Both are written as the smaller one, because",
            "# mia-train's in-plane rotation needs them equal. miao still reads 256 stored voxels on both axes:",
        ]
        lines += [f"#   {name[:84]}  {a:.6g} / {b:.6g} nm" for name, a, b in equalized]
    lines += ["#", "# Left out (stored voxel size x, y, z in nm; box in stored voxels where the entry has one):"]
    for name, native, sides, _ in dropped:
        voxel = " x ".join(f"{v:.4g}" for v in native)
        lines.append(f"#   {name[:84]}  {voxel}" + (f"  box {sides}" if sides else ""))

    body = (
        "\n".join(lines) + "\n\n" + RULE + "\n"
        "# Global settings. miao needs a global `resolutions`; every volume below overrides it.\n" + RULE + "\n"
        + yaml.safe_dump(base, sort_keys=False, default_flow_style=False)
        + "\n" + RULE + "\n" + f"# Volumes ({len(kept)}).\n" + RULE + "\n"
        + yaml.safe_dump({"volumes": kept}, sort_keys=False, default_flow_style=False)
    )
    out_path.write_text(body)

    print(f"wrote {out_path}: {len(kept)} of {len(volumes)} entries, {len(dropped)} left out, "
          f"{len(equalized)} with x and y written as one value")
    for name, native, sides, why in dropped:
        print(f"  left out  {name[:70]:72s} {native}  {sides}  {why[:90]}")
    ratios = np.array([max(v["resolutions"][0]) / min(v["resolutions"][0]) for v in kept])
    finest = np.array([min(v["resolutions"][0]) for v in kept])
    print(f"finest voxel side of the kept entries: min {finest.min():.3g}, median {np.median(finest):.3g}, "
          f"max {finest.max():.3g} nm; coarsest / finest side: median {np.median(ratios):.2f}, "
          f"max {ratios.max():.2f}; above 1.5 on {(ratios > 1.5).sum()}, above 3 on {(ratios > 3).sum()} entries")


if __name__ == "__main__":
    main()
