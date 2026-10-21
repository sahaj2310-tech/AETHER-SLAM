"""Pluggable feature extraction + matching backends for the SLAM tracker.

Two backends are provided:

* `DiskLightGlueBackend`: kornia's DISK detector/descriptor paired with the
  LightGlue matcher. Recommended on CUDA (RTX-class).  Functionally
  equivalent in role to SuperPoint+LightGlue; both are learned 2D-feature
  pipelines.  We use DISK because true SuperPoint weights are not bundled in
  kornia and we want to avoid a git-based install.

* `OrbLkBackend`: the classical OpenCV ORB + Lucas-Kanade tracker.  Pure
  CPU/OpenCV; used as a fallback when CUDA / kornia / model downloads fail.

Both backends present the same `detect(image)` and `match(prev, curr)` API.
The tracker uses descriptors instead of optical flow when available; this
gives much better robustness against motion blur, viewpoint change, and is
also what loop-closure uses.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("FeatureBackend")


@dataclass
class FrameFeatures:
    """Per-frame feature payload, framework-agnostic."""
    keypoints: np.ndarray         # (N, 2) float32 pixel coords (x, y)
    descriptors: np.ndarray       # (N, D) descriptor; dtype/shape backend-specific
    image_size: Tuple[int, int]   # (h, w)
    raw_descriptors: object = None  # backend-native object (e.g. torch tensor)
    frame_id: object = None       # optional frame identity (replay caches)
    subset: object = None         # identity of a keypoint subset (dynamic masking)


class _BaseBackend:
    name: str = "base"

    def detect(self, image_bgr: np.ndarray) -> FrameFeatures:
        raise NotImplementedError

    def match(self, f1: FrameFeatures, f2: FrameFeatures) -> np.ndarray:
        """Return (M, 2) array of matched index pairs (i in f1, j in f2)."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# DISK + LightGlue (CUDA)
# ---------------------------------------------------------------------------

class DiskLightGlueBackend(_BaseBackend):
    name = "disk_lightglue"

    def __init__(self, max_features: int = 512, device: str = "cuda",
                 half_precision: bool = True, lightglue_params: dict | None = None):
        import torch
        from kornia.feature import DISK, LightGlueMatcher

        self.torch = torch
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_features = max_features
        self.use_half = bool(half_precision) and self.device.type == "cuda"
        # LightGlue's attention can be unstable in pure FP16 with some kornia
        # builds; keep the matcher FP32 and only run DISK FP16.
        self.dtype = torch.float16 if self.use_half else torch.float32

        logger.info(f"Loading DISK + LightGlue on {self.device} "
                    f"(half_disk={self.use_half})...")
        self.disk = DISK.from_pretrained("depth").to(self.device).eval()
        if self.use_half:
            self.disk = self.disk.half()
        # Adaptive early stopping / point pruning trade recall for speed; tracking
        # wants every possible match, so they can be disabled (LIGHTGLUE_ADAPTIVE).
        self.matcher = LightGlueMatcher("disk", params=lightglue_params or {}).to(self.device).eval()
        logger.info(f"Feature backend ready: DISK+LightGlue ({max_features} feats max).")

    def _bgr_to_tensor(self, image_bgr: np.ndarray):
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        t = self.torch.from_numpy(rgb).to(device=self.device, dtype=self.dtype)
        return (t.permute(2, 0, 1).unsqueeze(0) / 255.0).contiguous()

    def detect(self, image_bgr: np.ndarray) -> FrameFeatures:
        h, w = image_bgr.shape[:2]
        with self.torch.inference_mode():
            t = self._bgr_to_tensor(image_bgr)
            out = self.disk(t, n=self.max_features, pad_if_not_divisible=True)
        feats = out[0]
        kps = feats.keypoints.detach().cpu().numpy().astype(np.float32)
        # LightGlue expects FP32 descriptors regardless of DISK dtype.
        desc = feats.descriptors.float()
        return FrameFeatures(
            keypoints=kps,
            descriptors=desc.detach().cpu().numpy(),
            image_size=(h, w),
            raw_descriptors=desc,
        )

    def match(self, f1: FrameFeatures, f2: FrameFeatures) -> np.ndarray:
        if f1.keypoints.shape[0] == 0 or f2.keypoints.shape[0] == 0:
            return np.zeros((0, 2), dtype=np.int64)

        torch = self.torch
        # Re-use on-GPU descriptors when available, else upload.
        d1 = f1.raw_descriptors if f1.raw_descriptors is not None else torch.from_numpy(f1.descriptors).to(self.device)
        d2 = f2.raw_descriptors if f2.raw_descriptors is not None else torch.from_numpy(f2.descriptors).to(self.device)

        n1, n2 = d1.shape[0], d2.shape[0]
        lafs1 = self._kps_to_lafs(f1.keypoints, n1)
        lafs2 = self._kps_to_lafs(f2.keypoints, n2)

        with torch.inference_mode():
            _, idxs = self.matcher(d1, d2, lafs1, lafs2)
        if idxs is None or idxs.numel() == 0:
            return np.zeros((0, 2), dtype=np.int64)
        return idxs.detach().cpu().numpy().astype(np.int64)

    def _kps_to_lafs(self, kps_np: np.ndarray, n: int):
        torch = self.torch
        lafs = torch.zeros(1, n, 2, 3, device=self.device)
        lafs[0, :, 0, 0] = 1.0
        lafs[0, :, 1, 1] = 1.0
        kps_t = torch.from_numpy(kps_np).to(device=self.device, dtype=torch.float32)
        lafs[0, :, :, 2] = kps_t
        return lafs


# ---------------------------------------------------------------------------
# ORB + LK fallback (CPU)
# ---------------------------------------------------------------------------

class OrbLkBackend(_BaseBackend):
    name = "orb_lk"

    def __init__(self, max_features: int = 1024):
        self.orb = cv2.ORB_create(nfeatures=max_features)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def detect(self, image_bgr: np.ndarray) -> FrameFeatures:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        kps, des = self.orb.detectAndCompute(gray, None)
        if kps is None or des is None or len(kps) == 0:
            return FrameFeatures(
                keypoints=np.zeros((0, 2), dtype=np.float32),
                descriptors=np.zeros((0, 32), dtype=np.uint8),
                image_size=gray.shape,
            )
        kp_np = np.array([k.pt for k in kps], dtype=np.float32)
        return FrameFeatures(
            keypoints=kp_np,
            descriptors=des,
            image_size=gray.shape,
        )

    def match(self, f1: FrameFeatures, f2: FrameFeatures) -> np.ndarray:
        if f1.descriptors.shape[0] == 0 or f2.descriptors.shape[0] == 0:
            return np.zeros((0, 2), dtype=np.int64)
        try:
            matches = self.bf.match(f1.descriptors, f2.descriptors)
        except cv2.error:
            return np.zeros((0, 2), dtype=np.int64)
        if not matches:
            return np.zeros((0, 2), dtype=np.int64)
        pairs = np.array([[m.queryIdx, m.trainIdx] for m in matches], dtype=np.int64)
        # Quality gate
        dists = np.array([m.distance for m in matches], dtype=np.float32)
        keep = dists < (dists.min() + 25.0)
        return pairs[keep]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_feature_backend(config) -> _BaseBackend:
    """Pick a backend based on config and runtime availability."""
    preferred = getattr(config, "FEATURE_BACKEND", "auto")
    max_feat = int(getattr(config, "DETECT_FEATURES", getattr(config, "MAX_FEATURES", 512)))
    half = bool(getattr(config, "FEATURE_HALF_PRECISION", True))

    if preferred in ("auto", "disk_lightglue"):
        try:
            import torch  # noqa: F401
            lg = {} if getattr(config, "LIGHTGLUE_ADAPTIVE", True) else {"depth_confidence": -1, "width_confidence": -1}
            return DiskLightGlueBackend(
                max_features=max_feat,
                device="cuda" if torch.cuda.is_available() else "cpu",
                half_precision=half,
                lightglue_params=lg,
            )
        except Exception as e:
            logger.warning(f"DISK+LightGlue unavailable ({e}); falling back to ORB+LK.")
    return OrbLkBackend(max_features=max_feat)
