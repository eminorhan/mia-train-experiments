"""Step time, loader wait and peak GPU memory of a run config, through mia-train's real trainer.

Launched like src/train.py (torchrun, one rank per GPU), then `engine.run.run`. Nothing in mia-train changes:
this file wraps `MetricLogger.log` to keep every logged metrics dict, and records the rank's peak CUDA memory
and whether the run ran out of memory. Writes <out>_rank<r>.json; throughput_table.py builds the table.
mia-train is whatever PYTHONPATH names (throughput.sh puts the pinned snapshot there and checks it).
"""

import argparse
import json
import os
import sys
import time
import tomllib
import traceback
from pathlib import Path

import torch

import components  # noqa: F401  (populates the registries)
from engine.run import run
from utils.metrics import MetricLogger

LOGGED: list[dict] = []
_original_log = MetricLogger.log


def recording_log(self, step, metrics, prefix="train"):
    LOGGED.append({"step": step, "prefix": prefix, "time": time.time(), **metrics})
    return _original_log(self, step, metrics, prefix)


MetricLogger.log = recording_log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="results are written to <out>_rank<r>.json")
    args = ap.parse_args()
    settings = tomllib.loads(args.config.read_text())
    status, error = "ok", ""
    try:
        run(args.config, args.output_root, None)
    except torch.OutOfMemoryError as exc:
        status, error = "oom", str(exc).splitlines()[0] if str(exc) else ""
        traceback.print_exc()  # shows which step ran out of memory
    except Exception as exc:  # recorded, then the rank exits non-zero
        status, error = "error", f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"
        traceback.print_exc()
    finally:
        rank = int(os.environ.get("RANK", 0))
        gib = 2**30
        devices = range(torch.cuda.device_count())  # a rank allocates on its own GPU only
        out = {
            "config": str(args.config),
            "experiment_name": settings["experiment_name"],
            "rank": rank,
            "ranks": int(os.environ.get("WORLD_SIZE", 1)),
            "batch_size": settings["trainer"]["batch_size"],
            "status": status,
            "error": error,
            "max_allocated_gib": max(torch.cuda.max_memory_allocated(d) for d in devices) / gib,
            "max_reserved_gib": max(torch.cuda.max_memory_reserved(d) for d in devices) / gib,
            "logged": LOGGED,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        Path(f"{args.out}_rank{rank}.json").write_text(json.dumps(out))
    if status != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
