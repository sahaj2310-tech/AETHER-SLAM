"""Camera motion of the TUM RGB-D sequences from the ground-truth trajectories: median rotation
rate, translation speed and view-change rate (rotation angle in rad + translation / 2 m, the
index of the CG-SSF process noise with a median scene depth of 2 m), resampled at 30 Hz.

  python tip/motion_stats.py --data <data> --out results/motion_stats.json
"""
import argparse
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation

from eval_common import read_tum_traj

SEQS = ["freiburg2_xyz", "freiburg1_360", "freiburg1_desk", "freiburg1_desk2", "freiburg1_floor", "freiburg1_plant",
        "freiburg1_room", "freiburg1_rpy", "freiburg1_teddy", "freiburg1_xyz", "freiburg3_walking_xyz",
        "freiburg3_walking_halfsphere", "freiburg3_sitting_xyz"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--depth", type=float, default=2.0)
    a = ap.parse_args()
    res = {}
    for s in SEQS:
        path = os.path.join(a.data, f"rgbd_dataset_{s}", "groundtruth.txt")
        if not os.path.exists(path):
            continue
        t, p, q = read_tum_traj(path)
        tt = np.arange(t[0], t[-1], 1 / 30.0)
        idx = np.searchsorted(t, tt).clip(1, len(t) - 1)
        R = Rotation.from_quat(q[idx]); P = p[idx]
        dr = (R[:-1].inv() * R[1:]).magnitude()
        dp = np.linalg.norm(np.diff(P, axis=0), axis=1)
        key = s.replace("freiburg", "fr").replace("_", "/", 1)
        res[key] = dict(rot_deg_s=float(np.degrees(np.median(dr) * 30)), trans_m_s=float(np.median(dp) * 30),
                        view_change_s=float(np.median(dr + dp / a.depth) * 30))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(dict(depth_m=a.depth, rate_hz=30, sequences=res), open(a.out, "w"), indent=1)
    for k, v in res.items():
        print(f"{k:26s} {v['rot_deg_s']:6.1f} deg/s {v['trans_m_s']:6.3f} m/s  view change {v['view_change_s']:.3f}/s")


if __name__ == "__main__":
    main()
