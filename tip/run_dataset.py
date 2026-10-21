"""
Replay a TUM RGB-D or Replica (NICE-SLAM render) sequence through AETHER-SLAM in synchronous mode
(every frame processed) and write evaluation-ready outputs:

  <out>/traj_tum.txt     camera-to-world poses, TUM format (timestamp tx ty tz qx qy qz qw)
  <out>/depth/<t>.npy    fused-frame depth in map units (float32, 0 = invalid)
  <out>/mesh.ply         final mesh
  <out>/timing.csv       per-frame stage timings (ms)
  <out>/meta.json        variant, seed, dataset, intrinsics, git commit, GPU, peak GPU memory
  <out>/gt_tum.txt       (Replica only) ground-truth trajectory in TUM format

  python run_dataset.py --dataset tum --root data/rgbd_dataset_freiburg1_desk \
         --variant cgssf --seed 0 --out runs/cgssf/fr1_desk/seed0
  python run_dataset.py --dataset replica --root data/Replica/room0 \
         --variant cgssf --seed 0 --out runs/cgssf/room0/seed0

Connect your pipeline in AetherAdapter below. --dry-run checks the I/O plumbing with a dummy adapter;
dry runs are marked in meta.json and are ignored by collect_results.py.
"""
import argparse, csv, glob, json, os, subprocess, time
import numpy as np
from PIL import Image

VARIANTS = {
    "proto_depthls": "(a) depth-domain least squares, no smoothing",
    "proto_ema": "(b) depth-domain least squares + EMA + reuse (prototype)",
    "inv_ls": "(c) inverse-domain least squares",
    "inv_irls": "(d) inverse-domain IRLS",
    "irls_ema": "(e) IRLS + EMA + reuse",
    "cgssf_norank1": "(g) CG-SSF without rank-one update",
    "cgssf_nogate": "(h) CG-SSF without gate",
    "cgssf_uniform": "(j) CG-SSF, uniform fusion weights",
    "cgssf": "(k) full system",
    "cgssf_nomask": "full system without person masking",
}
# TUM RGB-D calibration (fx, fy, cx, cy, [k1, k2, p1, p2, k3]) -- check against the TUM website.
TUM_CALIB = {
    "freiburg1": (517.3, 516.5, 318.6, 255.3, [0.2624, -0.9531, -0.0054, 0.0026, 1.1633]),
    "freiburg2": (520.9, 521.0, 325.1, 249.7, [0.2312, -0.7849, -0.0033, -0.0001, 0.9172]),
    "freiburg3": (535.4, 539.2, 320.1, 247.6, [0.0, 0.0, 0.0, 0.0, 0.0]),
}
REPLICA_CALIB = (600.0, 600.0, 599.5, 339.5)   # NICE-SLAM renders, 1200 x 680


# ======================================================================= ADAPTER (edit this)
class AetherAdapter:
    """Glue between this runner and AETHER-SLAM. Implement the three methods."""

    def __init__(self, variant, seed, K, size):
        # Example (adapt names to your code):
        #   from aether_slam.pipeline import Pipeline
        #   self.p = Pipeline(K=K, size=size, synchronous=True, seed=seed,
        #                     alignment=variant,                       # see VARIANTS
        #                     person_masking=(variant != "cgssf_nomask"))
        raise NotImplementedError("Connect AETHER-SLAM in AetherAdapter.__init__")

    def step(self, rgb, t):
        """Process one frame. Return dict(pose=4x4 camera-to-world or None,
        depth=HxW fused depth in map units or None,
        timing_ms={'track': .., 'depth': .., 'seg': .., 'align': .., 'fuse': ..})."""
        raise NotImplementedError

    def export_mesh(self, path):
        raise NotImplementedError

    def peak_gpu_mb(self):
        try:
            import torch
            return torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None
        except ImportError:
            return None


class DummyAdapter(AetherAdapter):
    """Plumbing test only: constant pose and depth. Never use its outputs as results."""

    def __init__(self, variant, seed, K, size):
        self.size = size

    def step(self, rgb, t):
        return dict(pose=np.eye(4), depth=np.ones((self.size[1], self.size[0]), np.float32), timing_ms={"track": 0.0})

    def export_mesh(self, path):
        open(path, "w").write("ply\nformat ascii 1.0\nelement vertex 0\nend_header\n")


# ======================================================================= datasets
def tum_sequence(root):
    name = os.path.basename(os.path.normpath(root))
    calib = next((v for k, v in TUM_CALIB.items() if k in name), None)
    if calib is None:
        raise ValueError(f"cannot infer TUM camera from folder name {name}")
    frames = []
    for line in open(os.path.join(root, "rgb.txt")):
        if line.strip() and not line.startswith("#"):
            t, f = line.split()[:2]
            frames.append((float(t), os.path.join(root, f)))
    fx, fy, cx, cy, dist = calib
    return frames, np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]), np.array(dist)


def replica_sequence(root, out):
    files = sorted(glob.glob(os.path.join(root, "results", "frame*.jpg")))
    poses = np.loadtxt(os.path.join(root, "traj.txt")).reshape(-1, 4, 4)
    from scipy.spatial.transform import Rotation
    with open(os.path.join(out, "gt_tum.txt"), "w") as fh:
        for i, T in enumerate(poses):
            q = Rotation.from_matrix(T[:3, :3]).as_quat()
            fh.write(f"{float(i):.6f} {T[0,3]:.6f} {T[1,3]:.6f} {T[2,3]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")
    fx, fy, cx, cy = REPLICA_CALIB
    return [(float(i), f) for i, f in enumerate(files)], np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]), None


def load_rgb(path, K, dist, width):
    img = np.asarray(Image.open(path).convert("RGB"))
    if dist is not None and np.any(np.abs(dist) > 0):
        import cv2
        img = cv2.undistort(img, K, dist)
    s = width / img.shape[1]
    if abs(s - 1) > 1e-9:
        img = np.asarray(Image.fromarray(img).resize((width, int(round(img.shape[0] * s))), Image.BILINEAR))
    Ks = K.copy(); Ks[:2] *= s
    return img, Ks


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["tum", "replica"], required=True)
    ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "depth"), exist_ok=True)
    frames, K, dist = tum_sequence(a.root) if a.dataset == "tum" else replica_sequence(a.root, a.out)
    _, Ks = load_rgb(frames[0][1], K, dist, a.width)
    h0 = int(round(Image.open(frames[0][1]).size[1] * a.width / Image.open(frames[0][1]).size[0]))
    adapter = (DummyAdapter if a.dry_run else AetherAdapter)(a.variant, a.seed, Ks, (a.width, h0))
    traj, rows, t_start = [], [], time.time()
    from scipy.spatial.transform import Rotation
    for t, f in frames:
        rgb, _ = load_rgb(f, K, dist, a.width)
        t0 = time.perf_counter(); res = adapter.step(rgb, t); wall = 1e3 * (time.perf_counter() - t0)
        if res.get("pose") is not None:
            T = np.asarray(res["pose"]); q = Rotation.from_matrix(T[:3, :3]).as_quat()
            traj.append(f"{t:.6f} {T[0,3]:.6f} {T[1,3]:.6f} {T[2,3]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}")
        if res.get("depth") is not None:
            np.save(os.path.join(a.out, "depth", f"{t:.6f}.npy"), np.asarray(res["depth"], np.float32))
        rows.append({"t": t, "wall_ms": wall, **res.get("timing_ms", {})})
    adapter.export_mesh(os.path.join(a.out, "mesh.ply"))
    open(os.path.join(a.out, "traj_tum.txt"), "w").write("\n".join(traj) + "\n")
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "t", k))
    with open(os.path.join(a.out, "timing.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    gpu = None
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    json.dump(dict(dataset=a.dataset, root=os.path.abspath(a.root), variant=a.variant, seed=a.seed,
                   dry_run=a.dry_run, frames=len(frames), tracked=len(traj), intrinsics=Ks.tolist(),
                   git_commit=git_commit(), gpu=gpu, peak_gpu_mb=adapter.peak_gpu_mb(),
                   wall_s=time.time() - t_start, finished=time.strftime("%Y-%m-%d %H:%M:%S")),
              open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(f"{a.variant} on {os.path.basename(os.path.normpath(a.root))}: {len(traj)}/{len(frames)} frames tracked"
          + ("  [DRY RUN - not a result]" if a.dry_run else ""))


if __name__ == "__main__":
    main()
