"""The checkpoint as one more post-processing hyperparameter: chosen on the fit half together with the size
filter, and only the winning checkpoint scored on the test half. Driven by score.sh --steps:

    python lmd_ssl_v2/sweep.py fit <scoring config> --val <step dir>/fit --scratch <dir> --out <json>
    python lmd_ssl_v2/sweep.py best <fit json>... --out <json> [-- <command for the winner, {step} in it>]

`fit` scores one checkpoint's fit half under every candidate of the config's search space, through mia-evals'
own calls: evaluate.fit's loop, keeping every candidate instead of only the best. `best` takes the
(checkpoint, candidate) pair that ranks first, ties going to the earlier checkpoint and then to the earlier
candidate as in evaluate.fit, and runs the command for that checkpoint: `mia-evals score`, which re-fits the
candidate on the same fit half and writes the record a single-checkpoint scoring would. `best` then checks
that the record chose the same candidate at the same fit score. The record names only the winning step; the
other checkpoints' scores are in `best`'s --out.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def cmd_fit(args: argparse.Namespace) -> None:
    from config import load_scoring_config      # mia-evals, an editable install in myvenv
    from evaluate import build, check_compatible, check_same_volume, resolve_artifacts, score_once

    config = load_scoring_config(args.config)
    if config.fit_volumes is None:
        raise SystemExit(f"{args.config} declares no [data.fit] split")
    task, processor, metrics = build(config)
    artifacts = {name: task.place(artifact)
                 for name, artifact in resolve_artifacts(args.val, config.fit_volumes).items()}
    for volume in config.fit_volumes:
        check_compatible(artifacts[volume.name], processor, task)
        check_same_volume(volume, artifacts[volume.name], "--val")
    args.scratch.mkdir(parents=True, exist_ok=True)
    processor.use_scratch(args.scratch)
    ranking = metrics[config.rank_by]
    candidates = []
    for params in processor.search_space():
        scores, per_volume, _ = score_once(
            artifacts, config.fit_volumes, task, processor, metrics, params, args.scratch)
        value = scores[config.rank_by][ranking.primary]
        print(f"  {processor.describe(params)}: {config.rank_by}.{ranking.primary} = {value:.4f}", flush=True)
        candidates.append({
            "params": params, "describe": processor.describe(params), "value": value,
            "per_volume": {name: float(entry[config.rank_by][ranking.primary])
                           for name, entry in per_volume.items()},
        })
    attrs = next(iter(artifacts.values())).attrs
    args.out.write_text(json.dumps({
        "config": str(args.config.resolve()), "task": config.task_name, "route": config.route,
        "run": attrs.get("run"), "step": attrs.get("step"), "val": str(args.val.resolve()),
        "rank_by": config.rank_by, "key": ranking.primary, "higher_is_better": ranking.higher_is_better,
        "candidates": candidates,
    }, indent=1) + "\n")
    print(f"wrote {args.out}", flush=True)


def cmd_best(args: argparse.Namespace, command: list[str]) -> None:
    sweeps = sorted((json.loads(path.read_text()) for path in args.fit_scores), key=lambda s: s["step"])
    shared = ("config", "task", "route", "run", "rank_by", "key", "higher_is_better")
    for sweep in sweeps[1:]:
        differ = [key for key in shared if sweep[key] != sweeps[0][key]]
        if differ:
            raise SystemExit(f"{sweep['val']} and {sweeps[0]['val']} were scored differently: {differ}")
    steps = [sweep["step"] for sweep in sweeps]
    if len(set(steps)) != len(steps):
        raise SystemExit(f"a checkpoint is listed twice: {steps}")

    sign = 1 if sweeps[0]["higher_is_better"] else -1
    metric = f"{sweeps[0]['rank_by']}.{sweeps[0]['key']}"
    best = None
    for sweep in sweeps:
        top = max(sweep["candidates"], key=lambda c: sign * c["value"])       # the first of equals
        print(f"  step {sweep['step']:>7}: {top['describe']}: {metric} = {top['value']:.4f}", flush=True)
        if best is None or sign * top["value"] > sign * best[1]["value"]:
            best = (sweep, top)
    sweep, chosen = best
    step = sweep["step"]
    print(f"chose step {step}, {chosen['describe']}: {metric} = {chosen['value']:.4f}", flush=True)
    args.out.write_text(json.dumps({
        "chosen": {"step": step, "params": chosen["params"], "describe": chosen["describe"],
                   "value": chosen["value"]},
        "sweeps": sweeps,
    }, indent=1) + "\n")
    if not command:
        return

    command = [part.replace("{step}", str(step)) for part in command]
    subprocess.run(command, check=True)
    from evaluate import DEFAULT_LEADERBOARD
    root = Path(command[command.index("--leaderboard") + 1]) if "--leaderboard" in command else DEFAULT_LEADERBOARD
    path = root / sweep["task"] / "records" / f"{sweep['run']}.step{step}.{sweep['route']}.json"
    record = json.loads(path.read_text())
    fitted = record["postprocess"]
    refit = (record["producer"]["step"], fitted["params"],
             fitted["validation_scores"][sweep["rank_by"]][sweep["key"]])
    if refit != (step, chosen["params"], chosen["value"]):
        raise SystemExit(f"{path} re-fitted (step, params, fit score) = {refit}; the sweep chose "
                         f"{(step, chosen['params'], chosen['value'])}")
    print(f"{path.name} re-fitted the sweep's candidate at the same fit score", flush=True)


def main() -> None:
    argv = sys.argv[1:]
    command = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="action", required=True)
    fit = sub.add_parser("fit", help="every candidate's score on one checkpoint's fit half")
    fit.add_argument("config", type=Path, help="a mia-evals scoring config, configs/<task>/<route>.toml")
    fit.add_argument("--val", type=Path, required=True, help="that checkpoint's <volume>.zarr for [data.fit]")
    fit.add_argument("--scratch", type=Path, required=True)
    fit.add_argument("--out", type=Path, required=True)
    best = sub.add_parser("best", help="the best (checkpoint, candidate); then the command for that checkpoint")
    best.add_argument("fit_scores", type=Path, nargs="+", help="`fit`'s outputs, one per checkpoint")
    best.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "fit":
        cmd_fit(args)
    else:
        cmd_best(args, command)


if __name__ == "__main__":
    main()
