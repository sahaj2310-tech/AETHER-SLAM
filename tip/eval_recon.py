"""Reconstruction accuracy / completion / completion ratio / F-score (NICE-SLAM-style protocol).

The predicted mesh (map units) is mapped into the ground-truth frame with the Sim(3) of the trajectory
alignment, then both meshes are sampled uniformly. Distances in metres; thresholds default to 5 cm.
For comparability with published numbers, use the same culled ground-truth meshes as prior work.

Usage:
  python eval_recon.py --pred-mesh run/mesh.ply --gt-mesh Replica/room0_mesh_culled.ply \
      --traj-est run/traj_tum.txt --traj-gt Replica/room0_traj_tum.txt
"""
import argparse, json
import numpy as np
from scipy.spatial import cKDTree
from eval_common import read_tum_traj, associate, umeyama_sim3


def recon_metrics(pred, gt, thr=0.05):
    d_acc = cKDTree(gt).query(pred, workers=-1)[0]     # predicted -> ground truth
    d_com = cKDTree(pred).query(gt, workers=-1)[0]     # ground truth -> predicted
    prec, rec = float(np.mean(d_acc < thr)), float(np.mean(d_com < thr))
    f = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return dict(acc_cm=100 * float(d_acc.mean()), comp_cm=100 * float(d_com.mean()), comp_ratio=100 * rec,
                precision=100 * prec, fscore=100 * f, chamfer_cm=50 * float(d_acc.mean() + d_com.mean()))


def sample_mesh(path, n):
    import open3d as o3d
    return np.asarray(o3d.io.read_triangle_mesh(path).sample_points_uniformly(number_of_points=n).points)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-mesh", required=True); ap.add_argument("--gt-mesh", required=True)
    ap.add_argument("--traj-est", required=True); ap.add_argument("--traj-gt", required=True)
    ap.add_argument("--n", type=int, default=200_000); ap.add_argument("--thr", type=float, default=0.05)
    a = ap.parse_args()
    te, pe, _ = read_tum_traj(a.traj_est); tg, pg, _ = read_tum_traj(a.traj_gt)
    ia, ib = associate(te, tg)
    s, R, t = umeyama_sim3(pe[ia], pg[ib])
    pred = (s * (R @ sample_mesh(a.pred_mesh, a.n).T)).T + t
    print(json.dumps(recon_metrics(pred, sample_mesh(a.gt_mesh, a.n), a.thr), indent=1))


if __name__ == "__main__":
    main()
