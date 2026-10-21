"""Depth accuracy and temporal scale consistency with ONE similarity scale per sequence.

AbsRel and delta_1 use the scale s of the Sim(3) trajectory alignment (est -> gt) for the whole
sequence, so per-frame scale flicker is penalized. SC = std_k log median_u(z_gt / (s z_hat)).

Usage (TUM RGB-D):
  python eval_depth.py --pred-dir run/depth_npy --gt-root rgbd_dataset_freiburg1_desk \
      --traj-est run/traj_tum.txt --traj-gt rgbd_dataset_freiburg1_desk/groundtruth.txt
  pred-dir holds one <timestamp>.npy per fused frame (float depth in map units, 0 = invalid).
  Replica (NICE-SLAM renders): --replica --gt-root Replica/room0 (depth*.png, scale 6553.5).
"""
import argparse, glob, os, json
import numpy as np
from PIL import Image
from eval_common import read_tum_traj, associate, ate_rmse


def depth_metrics(pairs, scale, zmin=0.1, zmax=10.0):
    absrel, d1, logm = [], [], []
    for zh, zg in pairs:
        zh = zh * scale
        m = (zg > zmin) & (zg < zmax) & (zh > 0) & np.isfinite(zh)
        if m.sum() < 100:
            continue
        r = zh[m] / zg[m]
        absrel.append(np.mean(np.abs(zh[m] - zg[m]) / zg[m]))
        d1.append(np.mean(np.maximum(r, 1 / r) < 1.25))
        logm.append(np.log(np.median(zg[m] / zh[m])))
    return dict(absrel=float(np.mean(absrel)), delta1=float(np.mean(d1)), sc=float(np.std(logm)), frames=len(absrel))


def load_tum_depth(root):
    rows = [l.split() for l in open(os.path.join(root, "depth.txt")) if not l.startswith("#")]
    return np.array([float(r[0]) for r in rows]), [os.path.join(root, r[1]) for r in rows], 5000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True); ap.add_argument("--gt-root", required=True)
    ap.add_argument("--traj-est", required=True); ap.add_argument("--traj-gt", required=True)
    ap.add_argument("--replica", action="store_true"); ap.add_argument("--max-dt", type=float, default=0.02)
    a = ap.parse_args()
    te, pe, _ = read_tum_traj(a.traj_est); tg, pg, _ = read_tum_traj(a.traj_gt)
    ia, ib = associate(te, tg, a.max_dt)
    ate, s, R, t = ate_rmse(pe[ia], pg[ib])
    preds = {float(os.path.basename(f)[:-4]): f for f in glob.glob(os.path.join(a.pred_dir, "*.npy"))}
    pt = np.array(sorted(preds))
    if a.replica:   # frames indexed 0..N-1; timestamps = frame index
        files = sorted(glob.glob(os.path.join(a.gt_root, "results", "depth*.png")))
        dt, df, fac = np.arange(len(files), dtype=float), files, 6553.5
    else:
        dt, df, fac = load_tum_depth(a.gt_root)
    ip, ig = associate(pt, dt, a.max_dt)
    pairs = []
    for i, j in zip(ip, ig):
        zh, g = np.load(preds[pt[i]]), Image.open(df[j])
        if g.size != (zh.shape[1], zh.shape[0]):          # e.g., Replica 1200x680 GT vs 640-wide prediction
            g = g.resize((zh.shape[1], zh.shape[0]), Image.NEAREST)
        pairs.append((zh, np.asarray(g, dtype=np.float64) / fac))
    res = depth_metrics(pairs, s); res.update(ate_cm=100 * ate, sim3_scale=float(s))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
