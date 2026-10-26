#!/bin/bash
# make_configs.py with the miao the training jobs import (submit.sh's MIAO_SRC); it adds mia-train's src itself.
#   cd ~/projects/mia-train-experiments && bsub -P miaai -n 4 -W 2:00 \
#     -o /nrs/scicompsoft/orhane/mia-train-experiments/lmd_ssl_v2_cellmap/jobs/make_configs_%J.log \
#     bash lmd_ssl_v2_cellmap/make_configs.sh
set -euo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONPYCACHEPREFIX=/scratch/$USER/pyc_$LSB_JOBID
export PYTHONPATH=/nrs/scicompsoft/orhane/probes/miao-pinned/62b3746/src
HERE="$(cd "$(dirname "$0")" && pwd)"
/groups/scicompsoft/home/orhane/myvenv/bin/python -u "$HERE/make_configs.py"
rm -rf "$PYTHONPYCACHEPREFIX"
