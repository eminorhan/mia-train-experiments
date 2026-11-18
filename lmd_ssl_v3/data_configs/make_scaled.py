#!/usr/bin/env python3
"""Write lmd_ssl_v3's data-scaling configs from the base config.

A data-scaling config holds every volume of the base config and a fraction of its data. Each entry is held to a
box that keeps that fraction of the region the base draws from (the entry's tissue box, or the whole volume):
  - The box is centered in the region and has the region's proportions.
  - No side is shorter than a patch plus 25% at the entry's resolution, so the entry still gives more than one
    patch position. Where a side is held at that minimum, the other sides shrink further, so the entry still
    keeps the fraction. An entry whose smallest such box is larger than the fraction keeps that box.
  - The boxes of a smaller fraction lie inside those of a larger one.
Resolutions, windows and the sampling rule (size ** exponent, measured by miao inside the box) are the base's.
The boxes of a volume held as several boxes get the weights lmd-configs gives them: together they count as one
volume of their total size.

Each entry is constructed in miao as the base has it, for its stored shape and voxel size, and again with its new
box. Run it as a cluster job. Writes ext_data<percent>pct_... beside the base, one file per fraction.

    python make_scaled.py BASE.yaml FRACTION [FRACTION ...]        e.g. 0.1 0.01
"""
import contextlib
import hashlib
import io
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
import yaml
from miao.config import MiaoConfig
from miao.dataset import VolumeDataset

MARGIN = 1.25                  # shortest side of a box, in patches
RULE = "# " + "-" * 75
AFTER_BOX = ("resolutions", "resolution_sampling", "normalize_max", "normalize_min", "normalize_clip", "weight")


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


def with_box(vol, box):
    """The entry with its `bounding_box` set, where the base writes one."""
    out, placed = {}, False
    for key, value in vol.items():
        if key == "bounding_box":
            out[key], placed = box, True
            continue
        if key in AFTER_BOX and not placed:
            out["bounding_box"], placed = box, True
        out[key] = value
    if not placed:
        out["bounding_box"] = box
    return out


def shrink(sides, minimum, fraction):
    """Sides of the box that keeps `fraction` of a region with these sides, none under its minimum."""
    sides = np.asarray(sides, dtype=np.float64)
    minimum = np.minimum(np.asarray(minimum, dtype=np.float64), sides)
    target = fraction * sides.prod()

    def box(scale):
        return np.minimum(sides, np.maximum(scale * sides, minimum))

    if minimum.prod() >= target:
        return [int(v) for v in minimum]
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if box(mid).prod() < target else (lo, mid)
    return [int(v) for v in np.minimum(np.ceil(box(hi)), sides)]


def probabilities(volumes, sizes, exponent):
    w = np.array([v.get("weight", 1.0) for v in volumes]) * np.asarray(sizes) ** exponent
    return w / w.sum()


def main():
    base_path = Path(sys.argv[1])
    fractions = sorted((float(x) for x in sys.argv[2:]), reverse=True)
    base = yaml.safe_load(base_path.read_text())
    volumes = base.pop("volumes")
    probe_globals = {k: v for k, v in base.items() if k != "size_weighting_exponent"}
    exponent = base["size_weighting_exponent"]
    out_axes = [a for a in base["output_axes"] if a in "xyz"]
    sha = hashlib.sha256(base_path.read_bytes()).hexdigest()[:16]

    # What the base draws from, per entry: the region (level-0 voxels, output axis order), the shortest sides a
    # box may have, and miao's own size of it.
    regions = []
    for i, vol in enumerate(volumes, 1):
        if i % 100 == 0:
            print(f"  [{i}/{len(volumes)}] read", flush=True)
        vi, why = construct(vol, probe_globals)
        if vi is None:
            raise SystemExit(f"{vol['name']} does not construct as the base config has it: {why}")
        finest = min(vi.img_level_voxels, key=lambda lv: float(np.prod(vi.img_level_voxels[lv])))
        stored = np.asarray(vi.image_meta.scales[finest].shape)[vi.img_spatial_idx]
        shape = dict(zip(vi.img_spatial_axes, (int(s) for s in stored)))
        voxel = dict(zip(vi.img_spatial_axes, (float(v) for v in vi.finest_voxel_size)))
        box = vol.get("bounding_box") or [[0, shape[a]] for a in out_axes]
        lo = [int(b[0]) for b in box]
        sides = [int(b[1]) - int(b[0]) for b in box]
        ladder = vol.get("resolutions", base["resolutions"])
        if len(ladder) != 1:
            raise SystemExit(f"{vol['name']}: {len(ladder)} scales; this script handles single-scale configs")
        minimum = [min(s, math.ceil(MARGIN * p * r / voxel[a]))
                   for a, s, p, r in zip(out_axes, sides, base["patch_size"], ladder[0])]
        regions.append({"lo": lo, "sides": sides, "minimum": minimum, "size": vi.size})
    base_voxels = np.array([float(np.prod(r["sides"])) for r in regions])
    base_p = probabilities(volumes, [r["size"] for r in regions], exponent)

    outer = None                                             # the boxes of the previous, larger fraction
    for fraction in fractions:
        percent = f"{100 * fraction:g}"
        new_volumes, boxes, sizes, kept_voxels = [], [], [], []
        for vol, region in zip(volumes, regions):
            sides = shrink(region["sides"], region["minimum"], fraction)
            box = [[lo + (full - side) // 2, lo + (full - side) // 2 + side]
                   for lo, full, side in zip(region["lo"], region["sides"], sides)]
            new = with_box(vol, box)
            vi, why = construct(new, probe_globals)
            if vi is None:
                raise SystemExit(f"{vol['name']}: its {percent}% box {box} does not hold a patch: {why}")
            new_volumes.append(new)
            boxes.append(box)
            sizes.append(vi.size)
            kept_voxels.append(float(np.prod(sides)))
        if outer is not None:
            for vol, box, big in zip(volumes, boxes, outer):
                if any(b[0] < o[0] or b[1] > o[1] for b, o in zip(box, big)):
                    raise SystemExit(f"{vol['name']}: its {percent}% box {box} is not inside the larger {big}")
        outer = boxes

        # The boxes of one volume count together as one volume of their total size (lmd-configs' rule).
        by_path = {}
        for k, vol in enumerate(new_volumes):
            if "weight" in vol:
                by_path.setdefault(vol["path"], []).append(k)
        reweighted = 0
        for ks in by_path.values():
            total = sum(kept_voxels[k] for k in ks)
            for k in ks:
                weight = round((kept_voxels[k] / total) ** (1.0 - exponent), 6)
                reweighted += abs(weight - new_volumes[k]["weight"]) > 1e-4
                new_volumes[k]["weight"] = weight

        kept_voxels = np.array(kept_voxels)
        share = kept_voxels / base_voxels
        more = int((share > 1.05 * fraction).sum())
        whole = int((share > 0.999).sum())
        at_minimum = sum(1 for box, r in zip(boxes, regions)
                         if any(b[1] - b[0] == m for b, m in zip(box, r["minimum"])))
        p = probabilities(new_volumes, sizes, exponent)
        moved = 0.5 * float(np.abs(p - base_p).sum())
        total_share = kept_voxels.sum() / base_voxels.sum()
        lines = [
            f'# lmd_ssl_v3 data config: Data scaling ({percent}%) (README, "Planned runs"). Derived on '
            f"{date.today().isoformat()} by",
            "# make_scaled.py (in this folder) from the base config,",
            f"#   {base_path.name}  sha256:{sha}",
            "# The base's volumes, resolutions, windows and sampling rule, with one change: every entry is held to a",
            f"# box that keeps {percent}% of the region the base draws from (its tissue box, or the whole volume).",
            "#   - The box is centered in that region and has its proportions.",
            f"#   - No side is shorter than a patch plus {round(100 * (MARGIN - 1))}% at the entry's resolution. "
            "Where a side is held at that",
            "#     minimum the other sides shrink further. An entry whose smallest such box is larger than its "
            f"{percent}%",
            "#     keeps that box.",
            "#   - The boxes of a smaller data-scaling config lie inside the boxes of a larger one.",
            "#",
            f"# {len(new_volumes)} entries, the base's. They hold {kept_voxels.sum() / 1e12:.2f} T of the base's "
            f"{base_voxels.sum() / 1e12:.1f} T level-0 voxels ({100 * total_share:.2f}%).",
            f"# {at_minimum} entries have a side at the minimum; {more} of them keep more than {percent}% of "
            f"their region, {whole} keep all of it.",
            f"# miao draws an entry in proportion to size ** {exponent:g}, with size measured inside its box. "
            "Against the base,",
            f"# the draw probabilities move by {100 * moved:.1f}% in total, towards the entries that keep more "
            "than their share.",
        ]
        body = (
            "\n".join(lines) + "\n\n" + RULE + "\n"
            "# Global settings.\n" + RULE + "\n"
            + yaml.safe_dump(base, sort_keys=False, default_flow_style=False)
            + "\n" + RULE + "\n" + f"# Volumes ({len(new_volumes)}).\n" + RULE + "\n"
            + yaml.safe_dump({"volumes": new_volumes}, sort_keys=False, default_flow_style=False)
        )
        out_path = base_path.with_name(base_path.name.replace("ext_", f"ext_data{percent}pct_", 1))
        out_path.write_text(body)
        print(f"\nwrote {out_path}")
        print(f"  {len(new_volumes)} entries; level-0 voxels {kept_voxels.sum() / 1e12:.3f} T of "
              f"{base_voxels.sum() / 1e12:.3f} T = {100 * total_share:.3f}%; miao's sizes "
              f"{sum(sizes) / 1e12:.3f} T of {sum(r['size'] for r in regions) / 1e12:.3f} T = "
              f"{100 * sum(sizes) / sum(r['size'] for r in regions):.3f}%")
        print(f"  a side at the minimum: {at_minimum}; keep more than {percent}%: {more}; keep everything: {whole}; "
              f"box weights changed by more than 1e-4: {reweighted}")
        print(f"  share of its region an entry keeps: median {100 * np.median(share):.3f}%, "
              f"90th percentile {100 * np.percentile(share, 90):.2f}%, max {100 * share.max():.1f}%")
        print(f"  draw probabilities against the base: total variation {100 * moved:.2f}%; largest gain "
              f"{100 * (p - base_p).max():.3f} points ({volumes[int((p - base_p).argmax())]['name'][:70]})")
        order = np.argsort(-share)[:5]
        for k in order:
            print(f"    keeps {100 * share[k]:5.1f}%  {volumes[k]['name'][:70]}  region {regions[k]['sides']}  "
                  f"box {[b[1] - b[0] for b in boxes[k]]}")


if __name__ == "__main__":
    main()
