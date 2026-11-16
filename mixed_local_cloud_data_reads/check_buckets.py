"""mixed_local_cloud_data_reads: read the metadata of every volume a run config reads from a URL (a bucket).

submit.sh runs this at the start of every job, with the miao the run imports. A node that cannot reach a bucket
then fails at once, with the volume's name, instead of stalling in the data loader.

    python check_buckets.py <run config .toml>
"""

import sys
import tomllib

import yaml
from miao.formats import read_metadata
from miao.store import is_url

with open(sys.argv[1], "rb") as f:
    run = tomllib.load(f)
found = 0
for path in sorted({run[s]["config_path"] for s in ("data", "val_data") if "config_path" in run.get(s, {})}):
    with open(path) as f:
        volumes = yaml.safe_load(f)["volumes"]
    for v in volumes:
        if is_url(v["path"]):
            meta = read_metadata(v["path"], v["image_key"], v.get("format") or v.get("zarr_version", "zarr2"))
            print(f"[buckets] {v['name']}: {v['path']}/{v['image_key']} reached, {len(meta.scales)} levels",
                  flush=True)
            found += 1
print(f"[buckets] {found} bucket volume(s) in {sys.argv[1]}", flush=True)
