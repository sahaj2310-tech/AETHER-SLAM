"""Frame sources: webcam, or recorded sequences (TUM RGB-D, Replica / NICE-SLAM
renders, video files) that the live pipeline replays as a virtual camera and
the benchmark runner reads frame by frame.

A recorded source yields (timestamp_s, frame_bgr) with undistorted frames
resized to the working width; `K` holds the matching pinhole intrinsics.
"""
from __future__ import annotations

import glob
import os

import cv2
import numpy as np

# TUM RGB-D calibration (fx, fy, cx, cy, [k1, k2, p1, p2, k3]) from the TUM website
TUM_CALIB = {
    "freiburg1": (517.3, 516.5, 318.6, 255.3, [0.2624, -0.9531, -0.0054, 0.0026, 1.1633]),
    "freiburg2": (520.9, 521.0, 325.1, 249.7, [0.2312, -0.7849, -0.0033, -0.0001, 0.9172]),
    "freiburg3": (535.4, 539.2, 320.1, 247.6, [0.0, 0.0, 0.0, 0.0, 0.0]),
}
REPLICA_CALIB = (600.0, 600.0, 599.5, 339.5)   # NICE-SLAM renders, 1200 x 680
REPLICA_FPS = 30.0                             # renders carry no timestamps; t = index / 30


class RecordedSequence:
    """Common interface: len(), frame(i) -> BGR, timestamps, K, size, name, kind."""

    def __init__(self, root: str, width: int = 640):
        self.root = os.path.normpath(root)
        self.name = os.path.basename(self.root)
        self.width = int(width)
        self.kind = None
        self.timestamps: list[float] = []
        self.files: list[str] = []
        self.K = None
        self.size = None
        self._maps = None
        self._K_raw = None
        self._dist = None

    # ------------------------------------------------------------------ factories
    @staticmethod
    def open(root: str, width: int = 640) -> "RecordedSequence":
        if os.path.isfile(root):
            return VideoSequence(root, width)
        if os.path.exists(os.path.join(root, "rgb.txt")):
            return TUMSequence(root, width)
        if os.path.exists(os.path.join(root, "traj.txt")) and os.path.isdir(os.path.join(root, "results")):
            return ReplicaSequence(root, width)
        raise ValueError(f"unrecognised sequence folder: {root}")

    def __len__(self):
        return len(self.files)

    def _setup_geometry(self, K_raw, dist, w0, h0):
        s = self.width / float(w0)
        h = int(round(h0 * s))
        K = np.array(K_raw, np.float64).copy()
        K[:2] *= s
        self._K_raw, self._dist = np.array(K_raw, np.float64), dist
        self.K, self.size, self._scale = K, (self.width, h), s
        if dist is not None and np.any(np.abs(dist) > 0):
            # undistort at the original resolution with the same camera matrix
            self._maps = cv2.initUndistortRectifyMap(self._K_raw, np.asarray(dist, np.float64), None,
                                                     self._K_raw, (w0, h0), cv2.CV_32FC1)

    def _prepare(self, img: np.ndarray, interp=cv2.INTER_LINEAR) -> np.ndarray:
        if self._maps is not None:
            img = cv2.remap(img, self._maps[0], self._maps[1], interp)
        if img.shape[1] != self.size[0] or img.shape[0] != self.size[1]:
            img = cv2.resize(img, self.size, interpolation=cv2.INTER_AREA if interp == cv2.INTER_LINEAR else interp)
        return img

    def frame(self, i: int) -> np.ndarray:
        img = cv2.imread(self.files[i], cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"cannot read {self.files[i]}")
        return self._prepare(img)


class TUMSequence(RecordedSequence):
    def __init__(self, root, width=640):
        super().__init__(root, width)
        self.kind = "tum"
        calib = next((v for k, v in TUM_CALIB.items() if k in self.name), None)
        if calib is None:
            raise ValueError(f"cannot infer TUM camera from folder name {self.name}")
        for line in open(os.path.join(root, "rgb.txt")):
            if line.strip() and not line.startswith("#"):
                t, f = line.split()[:2]
                self.timestamps.append(float(t))
                self.files.append(os.path.join(root, f))
        fx, fy, cx, cy, dist = calib
        w0, h0 = 640, 480
        self._setup_geometry([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], np.array(dist), w0, h0)

    def gt_trajectory_path(self):
        return os.path.join(self.root, "groundtruth.txt")

    def gt_depth_index(self):
        """(timestamps, files, depth factor) of the registered ground-truth depth."""
        rows = [l.split() for l in open(os.path.join(self.root, "depth.txt")) if l.strip() and not l.startswith("#")]
        return np.array([float(r[0]) for r in rows]), [os.path.join(self.root, r[1]) for r in rows], 5000.0

    def gt_depth(self, path: str) -> np.ndarray:
        """Ground-truth depth (metres) undistorted with the RGB map (nearest neighbour)."""
        d = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 5000.0
        return self._prepare(d, interp=cv2.INTER_NEAREST)


class ReplicaSequence(RecordedSequence):
    def __init__(self, root, width=640):
        super().__init__(root, width)
        self.kind = "replica"
        self.files = sorted(glob.glob(os.path.join(root, "results", "frame*.jpg")))
        self.timestamps = [i / REPLICA_FPS for i in range(len(self.files))]
        fx, fy, cx, cy = REPLICA_CALIB
        self._setup_geometry([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], None, 1200, 680)

    def gt_poses(self) -> np.ndarray:
        return np.loadtxt(os.path.join(self.root, "traj.txt")).reshape(-1, 4, 4)

    def write_gt_tum(self, path: str):
        from scipy.spatial.transform import Rotation
        with open(path, "w") as fh:
            for t, T in zip(self.timestamps, self.gt_poses()):
                q = Rotation.from_matrix(T[:3, :3]).as_quat()
                fh.write(f"{t:.6f} {T[0,3]:.6f} {T[1,3]:.6f} {T[2,3]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")


class VideoSequence(RecordedSequence):
    """A video file; frames are decoded sequentially (random access is emulated)."""

    def __init__(self, path, width=640):
        super().__init__(path, width)
        self.kind = "video"
        self.cap = cv2.VideoCapture(path)
        n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        w0 = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h0 = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.files = [path] * n
        self.timestamps = [i / fps for i in range(n)]
        f = 0.82 * w0          # ~70 deg horizontal field of view if nothing better is known
        self._setup_geometry([[f, 0, w0 / 2], [0, f, h0 / 2], [0, 0, 1.0]], None, w0, h0)
        self._next = 0

    def frame(self, i):
        if i != self._next:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, img = self.cap.read()
        self._next = i + 1
        if not ok:
            raise IOError("video read failed")
        return self._prepare(img)
