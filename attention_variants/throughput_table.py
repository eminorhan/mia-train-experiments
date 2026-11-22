"""The throughput table of one throughput.sh job, from the files throughput_probe.py wrote.

    python throughput_table.py <results dir> <job id>

One row per arm. Every figure is the median over the run's log intervals except the first, which holds
compilation. Seconds per step come from the trainer's samples per second (ranks x batch per rank / samples per
second) or, when the run has `measure_mfu = false` and so reports none, from the wall-clock time between two log
lines. `data wait` is the worst rank's share of wall time spent waiting for a batch.
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
    peak = max(r["max_allocated_gib"] for r in ranks)
    if status != ["ok"]:
        print(f"{arm:28} {status}, peak {peak:.1f} GiB allocated: {ranks[0]['error'][:150]}")
        continue
    per_step = ranks[0]["ranks"] * ranks[0]["batch_size"]  # samples per step
    logged = [entry for entry in ranks[0]["logged"] if entry["prefix"] == "train"]
    intervals = []
    for before, entry in zip(logged, logged[1:]):
        if "samples_per_s" in entry:
            seconds = per_step / entry["samples_per_s"]
        else:
            seconds = (entry["time"] - before["time"]) / (entry["step"] - before["step"])
        intervals.append({"step": entry["step"], "seconds": seconds, "wait": entry["data_wait_frac_max"],
                          "mfu": entry.get("mfu"), "tflops": entry.get("tflops_per_s")})
    seconds = statistics.median(i["seconds"] for i in intervals)
    wait = statistics.median(i["wait"] for i in intervals)
    measured = [i for i in intervals if i["mfu"] is not None]
    mfu = statistics.median(i["mfu"] for i in measured) if measured else None
    tflops = statistics.median(i["tflops"] for i in measured) if measured else None
    print(f"{arm:28} {ranks[0]['ranks']:5d} {seconds:8.3f} {per_step / seconds:10.3f} {wait:10.2%} "
          + (f"{mfu:6.3f} {tflops:8.1f}" if measured else f"{'-':>6} {'-':>8}") + f" {peak:9.1f}")
    summary[arm] = {"ranks": ranks[0]["ranks"], "seconds_per_step": seconds, "samples_per_s": per_step / seconds,
                    "data_wait_frac_max": wait, "mfu": mfu, "tflops_per_s": tflops, "peak_allocated_gib": peak,
                    "steps": [intervals[0]["step"], intervals[-1]["step"]]}
    print("    per log interval, step: s/step (data wait): "
          + "  ".join(f"{i['step']}: {i['seconds']:.3f} ({i['wait']:.1%})" for i in intervals))
(results / f"summary_{job}.json").write_text(json.dumps(summary, indent=1))
