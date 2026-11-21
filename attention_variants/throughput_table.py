"""The throughput table of one throughput.sh job, from the files throughput_probe.py wrote.

    python throughput_table.py <results dir> <job id>

One row per arm. Seconds per step = ranks x batch per rank / samples per second, and every figure is the
median over the run's log intervals that carry a rate: all but the first, which holds compilation.
`data wait` is the worst rank's share of wall time spent waiting for a batch.
"""

import json
import statistics
import sys
from pathlib import Path

results, job = Path(sys.argv[1]), sys.argv[2]
arms: dict[str, list[dict]] = {}
for path in sorted(results.glob(f"*_{job}_rank*.json")):
    record = json.loads(path.read_text())
    arms.setdefault(path.name.split(f"_{job}_rank")[0], []).append(record)

print(f"{'arm':28} {'ranks':>5} {'s/step':>8} {'samples/s':>10} {'data wait':>10} {'MFU':>6} {'TFLOP/s':>8} {'peak GiB':>9}")
summary = {}
for arm, ranks in arms.items():
    status = sorted({r["status"] for r in ranks})
    if status != ["ok"]:
        print(f"{arm:28} {status}: {ranks[0]['error']}")
        continue
    rated = [entry for entry in ranks[0]["logged"] if entry["prefix"] == "train" and "samples_per_s" in entry]
    steady = rated
    middle = {key: statistics.median(entry[key] for entry in steady)
              for key in ("samples_per_s", "data_wait_frac_max", "mfu", "tflops_per_s")}
    seconds = ranks[0]["ranks"] * ranks[0]["batch_size"] / middle["samples_per_s"]
    peak = max(r["max_allocated_gib"] for r in ranks)
    print(f"{arm:28} {ranks[0]['ranks']:5d} {seconds:8.3f} {middle['samples_per_s']:10.3f} "
          f"{middle['data_wait_frac_max']:10.2%} {middle['mfu']:6.3f} {middle['tflops_per_s']:8.1f} {peak:9.1f}")
    summary[arm] = {"ranks": ranks[0]["ranks"], "seconds_per_step": seconds, "peak_allocated_gib": peak,
                    "steady_steps": [steady[0]["step"], steady[-1]["step"]], **middle}
    print("    per log interval, step: s/step (data wait): "
          + "  ".join(f"{e['step']}: {ranks[0]['ranks'] * ranks[0]['batch_size'] / e['samples_per_s']:.3f} "
                      f"({e['data_wait_frac_max']:.1%})" for e in rated))
(results / f"summary_{job}.json").write_text(json.dumps(summary, indent=1))
