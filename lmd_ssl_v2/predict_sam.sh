#!/usr/bin/env bash
# predict.py's segment-everything pass for one volume of a promptable (SAM) arm, with sam_lmd_v1's labeller settings:
# its pseudolabel.LABEL_AMG, and points_per_batch from the arm's mask stride, as sam_lmd_v1/predict_eval.sh does.
# Run inside a job by score.sh, which sets PYTHONPATH (the miao pin, then mia-train's src). Anything after the five
# arguments goes to predict.py as is (score_zebrafinch.sh: --no-truth, --allow-resolution-mismatch).
#
#   bash lmd_ssl_v2/predict_sam.sh <run dir> <step> <data config> <volume> <out dir> [predict.py flags...]
set -euo pipefail
RUN=$1 STEP=$2 CONFIG=$3 VOLUME=$4 OUT=$5
shift 5
MIA_TRAIN=/groups/scicompsoft/home/orhane/projects/mia-train
SAM=/groups/scicompsoft/home/orhane/projects/mia-train-experiments/sam_lmd_v1
PY=/groups/scicompsoft/home/orhane/myvenv/bin/python

# The labeller's click grid (14 per side) is for a 256-voxel window; sam_lmd_v1 rescales it for other crops.
crop=$($PY -c "import json,sys; print(int(json.load(open(sys.argv[1]))['model']['kwargs'].get('img_size', 256)))" "$RUN/resolved_config.json")
[[ $crop == 256 ]] || { echo "$RUN trained on a $crop crop; this script assumes 256" >&2; exit 2; }
stride=$($PY -c "import json,sys; r=json.load(open(sys.argv[1])); print(int(r['model']['kwargs'].get('patch_size', 16)) // int(r['algorithm']['kwargs'].get('mask_upscale', 4)))" "$RUN/resolved_config.json")
case $stride in 1) batch=8 ;; 2) batch=32 ;; *) batch=64 ;; esac      # a throughput knob, as in predict_eval.sh
overrides=$($PY - "$SAM" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from pseudolabel import LABEL_AMG
def toml(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    return '"%s"' % v if isinstance(v, str) else str(v)
print(" ".join(f"--override algorithm.{k}={toml(v)}" for k, v in LABEL_AMG.items() if k != "points_per_batch"))
PY
)
overrides+=" --override algorithm.points_per_batch=$batch"
echo "[predict_sam] mask stride $stride, points_per_batch $batch; $overrides"
cd "$MIA_TRAIN"
# shellcheck disable=SC2086                    # word-split on purpose: a list of --override flags
$PY src/predict.py "$RUN" --step "$STEP" --data-config "$CONFIG" --volume "$VOLUME" --out "$OUT" $overrides "$@"
