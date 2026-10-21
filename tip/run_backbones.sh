#!/usr/bin/env bash
# Depth-network comparison of the full system (supplementary material): TUM fr1 (9), seed 0.
#   bash tip/run_backbones.sh <data_root> <runs_root>
DATA=$1; RUNS=$2
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
for SPEC in DPT_Large:dptlarge MiDaS_small:midassmall DepthAnythingV2_Small:dav2s; do
  DM=${SPEC%%:*}; TAG=${SPEC#*:}
  for S in 360 desk desk2 floor plant room rpy teddy xyz; do
    venv/Scripts/python.exe tip/run_suite.py --root $DATA/rgbd_dataset_freiburg1_$S --name fr1_$S --variants cgssf \
        --seeds 0 --runs $RUNS --depth-model $DM --tag-suffix _$TAG
  done
done
echo BACKBONES_DONE
