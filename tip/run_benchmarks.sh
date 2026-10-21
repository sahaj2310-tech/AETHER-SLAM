#!/usr/bin/env bash
# Full benchmark of the full system (Table V and the supplementary material): TUM RGB-D fr1 (9) + fr3 dynamic (3) and Replica (8),
# all alignment variants x seeds, synchronous replay. Resumable (finished runs are skipped).
#   bash tip/run_benchmarks.sh <data_root> <runs_root> <worker: tum|replica> "<seeds>"
DATA=$1; RUNS=$2; WORKER=$3; SEEDS=${4:-0,1,2}
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
PY=venv/Scripts/python.exe
ABL=proto_depthls,proto_ema,inv_ls,inv_irls,irls_ema,cgssf_norank1,cgssf_nogate,cgssf_uniform,cgssf
if [ "$WORKER" = "tum" ]; then
  for S in room desk xyz 360 desk2 floor plant rpy teddy; do
    MESH=""; [ "$S" = "room" ] && MESH="--save-mesh inv_irls,irls_ema,cgssf"
    $PY tip/run_suite.py --root $DATA/rgbd_dataset_freiburg1_$S --name fr1_$S --variants $ABL --seeds $SEEDS --runs $RUNS $MESH
  done
  for S in walking_xyz walking_halfsphere sitting_xyz; do
    $PY tip/run_suite.py --root $DATA/rgbd_dataset_freiburg3_$S --name fr3_$S --variants cgssf,cgssf_nomask,proto_ema --seeds $SEEDS --runs $RUNS
  done
else
  for S in room0 room1 room2 office0 office1 office2 office3 office4; do
    MESH=""; [ "$S" = "room0" ] && MESH="--save-mesh inv_irls,irls_ema,cgssf"
    $PY tip/run_suite.py --root $DATA/Replica/$S --name $S --variants $ABL --seeds $SEEDS --runs $RUNS \
        --recon-gt $DATA/Replica/cull_replica_mesh/$S.ply $MESH
  done
fi
echo "BENCHMARK_${WORKER}_DONE"
