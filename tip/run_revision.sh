#!/usr/bin/env bash
# Runs of the revised evaluation (view-indexed CG-SSF with the normalized gate), resumable
# (finished runs are skipped). Run one job at a time: two concurrent processes exceed 16 GB of RAM.
#   bash tip/run_revision.sh <data_root> <runs_root> <job>
# jobs:
#   tum_normal      fr1 (9): CG-SSF and its ablations, fr3 (3): CG-SSF and no-mask; 3 seeds, gate diagnostics
#   replica_normal  Replica (8): CG-SSF (+ extra fusion volumes) and its ablations; 3 seeds, gate diagnostics
#   tum_oracle      fr1 (9) with ground-truth poses, all estimators and the noise sensitivity, seed 0
#   stx_oracle      fr3 structure vs. texture (8) with ground-truth poses, all estimators, seed 0
#   band_oracle     anchor-restriction test on fr1 (9) and fr3 structure vs. texture (8): in every second
#                   block of 50 frames only the anchors within +-10% of the median network output; seed 0
#   band_replica    the same test on Replica (8) with reconstruction (extra fusion volumes of CG-SSF); seed 0
#   replica_oracle  Replica scenes given as 4th argument (default: all 8) with ground-truth poses, seed 0
#   backbones       fr1 (9): CG-SSF with DPT-Large, MiDaS-small, Depth Anything V2-S, seed 0
#   design_tune     held-out fr2/xyz with ground-truth poses: window length W of the sliding-window fit (l)
#                   (DPT-Hybrid) and process noise of the scale-only filter (m) (metric Depth Anything V2-S)
#   design_tum      alignment designs of the closest methods on fr1 (9) and fr3 structure vs. texture (8) with
#                   ground-truth poses, normal and anchor-restricted (tag suffix _band): (l) with DPT-Hybrid;
#                   metric Depth Anything V2-S without alignment, with (m), with CG-SSF; relative Depth
#                   Anything V2-S with (d) and CG-SSF; seed 0
#   design_replica  the same on Replica (8): (l) normal and restricted; metric (m) and CG-SSF, relative CG-SSF
# Runs of the fixed-threshold gate (before the normalized gate) carry the tag suffix _fixedgate.
# Ground-truth-pose runs evaluate the per-pixel uncertainty of the CG-SSF variants (--unc-eval).
DATA=$1; RUNS=$2; JOB=$3; SCENES=${4:-"room0 room1 room2 office0 office1 office2 office3 office4"}
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
PY=venv/Scripts/python.exe
FR1="360 desk desk2 floor plant room rpy teddy xyz"
# fr3 structure vs. texture: <name>:<folder suffix>
STX="str_tex_far:structure_texture_far str_tex_near:structure_texture_near str_notex_far:structure_notexture_far
     str_notex_near:structure_notexture_near nostr_tex_far:nostructure_texture_far
     nostr_tex_near_loop:nostructure_texture_near_withloop nostr_notex_far:nostructure_notexture_far
     nostr_notex_near_loop:nostructure_notexture_near_withloop"
ABL=cgssf,cgssf_timeq,cgssf_norank1,cgssf_nogate
DAM=DepthAnythingV2_MetricIndoor_Small
ORACLE_VARIANTS=${ORACLE_VARIANTS:-proto_ema,inv_irls,irls_ema,oracle_align,cgssf,cgssf_timeq,cgssf_norank1,cgssf_nogate,cgssf_fixedgate}
case $JOB in
  tum_normal)
    for S in $FR1; do
      MESH=""; [ "$S" = "desk" ] && MESH="--save-mesh cgssf"
      $PY tip/run_suite.py --root $DATA/rgbd_dataset_freiburg1_$S --name fr1_$S --variants $ABL \
          --seeds 0,1,2 --runs $RUNS --diag $MESH
    done
    for S in walking_xyz walking_halfsphere sitting_xyz; do
      $PY tip/run_suite.py --root $DATA/rgbd_dataset_freiburg3_$S --name fr3_$S --variants cgssf,cgssf_nomask \
          --seeds 0,1,2 --runs $RUNS --diag
    done ;;
  replica_normal)
    for S in $SCENES; do
      MESH=""; [ "$S" = "room0" ] && MESH="--save-mesh cgssf"
      $PY tip/run_suite.py --root $DATA/Replica/$S --name $S --variants $ABL --seeds 0,1,2 \
          --runs $RUNS --recon-gt $DATA/Replica/cull_replica_mesh/$S.ply --extra-fusion --diag $MESH
    done ;;
  tum_oracle)
    for S in $FR1; do
      $PY tip/run_suite.py --root $DATA/rgbd_dataset_freiburg1_$S --name fr1_$S --variants $ORACLE_VARIANTS,cgssf_eta025,cgssf_eta4 \
          --seeds 0 --runs $RUNS --tag-suffix _oracle --oracle-poses --unc-eval
    done ;;
  stx_oracle)
    for SPEC in $STX; do
      N=${SPEC%%:*}; F=${SPEC#*:}
      $PY tip/run_suite.py --root $DATA/rgbd_dataset_freiburg3_$F --name fr3_$N --variants $ORACLE_VARIANTS \
          --seeds 0 --runs $RUNS --tag-suffix _oracle --oracle-poses --unc-eval
    done ;;
  band_oracle)
    BAND=proto_ema,inv_irls,irls_ema,cgssf_norank1,cgssf
    for S in $FR1; do
      $PY tip/run_suite.py --root $DATA/rgbd_dataset_freiburg1_$S --name fr1_$S --variants $BAND --seeds 0 \
          --runs $RUNS --tag-suffix _oracle_band --oracle-poses --config ANCHOR_BAND=0.1
    done
    for SPEC in $STX; do
      N=${SPEC%%:*}; F=${SPEC#*:}
      $PY tip/run_suite.py --root $DATA/rgbd_dataset_freiburg3_$F --name fr3_$N --variants $BAND --seeds 0 \
          --runs $RUNS --tag-suffix _oracle_band --oracle-poses --config ANCHOR_BAND=0.1
    done ;;
  band_replica)
    BAND=proto_ema,inv_irls,irls_ema,cgssf_norank1,cgssf
    for S in $SCENES; do
      $PY tip/run_suite.py --root $DATA/Replica/$S --name $S --variants $BAND --seeds 0 --runs $RUNS \
          --tag-suffix _oracle_band --oracle-poses --extra-fusion --recon-gt $DATA/Replica/cull_replica_mesh/$S.ply \
          --config ANCHOR_BAND=0.1
    done ;;
  replica_oracle)
    for S in $SCENES; do
      $PY tip/run_suite.py --root $DATA/Replica/$S --name $S --variants $ORACLE_VARIANTS --seeds 0 --runs $RUNS \
          --tag-suffix _oracle --oracle-poses --extra-fusion --recon-gt $DATA/Replica/cull_replica_mesh/$S.ply --unc-eval
    done ;;
  backbones)
    for SPEC in DPT_Large:dptlarge MiDaS_small:midassmall DepthAnythingV2_Small:dav2s; do
      DM=${SPEC%%:*}; TAG=${SPEC#*:}
      for S in $FR1; do
        $PY tip/run_suite.py --root $DATA/rgbd_dataset_freiburg1_$S --name fr1_$S --variants cgssf --seeds 0 \
            --runs $RUNS --depth-model $DM --tag-suffix _$TAG
      done
    done ;;
  design_tune)
    $PY tip/run_suite.py --root $DATA/rgbd_dataset_freiburg2_xyz --name fr2_xyz --seeds 0 --runs $RUNS --oracle-poses \
        --plan "|_W5_tune|win_irls||ALIGN_WINDOW_FRAMES=5;|_W10_tune|win_irls||ALIGN_WINDOW_FRAMES=10;|_W20_tune|win_irls||ALIGN_WINDOW_FRAMES=20;|_W40_tune|win_irls||ALIGN_WINDOW_FRAMES=40;$DAM|_dav2m_eta0.071_tune|scale_kf||SCALE_KF_ETA=0.071;$DAM|_dav2m_eta0.284_tune|scale_kf||SCALE_KF_ETA=0.284;$DAM|_dav2m_eta1.136_tune|scale_kf||SCALE_KF_ETA=1.136" ;;
  design_tum)
    for SPEC in $(for S in $FR1; do echo freiburg1_$S:fr1_$S; done) $(for X in $STX; do echo freiburg3_${X#*:}:fr3_${X%%:*}; done); do
      F=${SPEC%%:*}; N=${SPEC#*:}
      $PY tip/run_suite.py --root $DATA/rgbd_dataset_$F --name $N --seeds 0 --runs $RUNS --oracle-poses \
          --plan "|_oracle|win_irls|win_irls;$DAM|_dav2m_oracle|metric_raw,scale_kf,cgssf|scale_kf,cgssf;DepthAnythingV2_Small|_dav2s_oracle|inv_irls,cgssf|cgssf"
    done ;;
  design_replica)
    for S in $SCENES; do
      $PY tip/run_suite.py --root $DATA/Replica/$S --name $S --seeds 0 --runs $RUNS --oracle-poses \
          --recon-gt $DATA/Replica/cull_replica_mesh/$S.ply \
          --plan "|_oracle|win_irls|win_irls;$DAM|_dav2m_oracle|scale_kf,cgssf|;DepthAnythingV2_Small|_dav2s_oracle|cgssf|"
    done ;;
  *) echo "unknown job $JOB"; exit 1 ;;
esac
echo "JOB_DONE $JOB"
