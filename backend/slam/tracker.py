"""Visual SLAM tracker (frame-to-frame visual odometry against a sparse map).

Per frame:
  1. Run the feature backend (DISK+LightGlue or ORB) on the new frame and drop
     keypoints inside dynamic-object (person) boxes.
  2. Match against the previous frame's features; every matched keypoint that
     carries a map point gives a 2D-3D correspondence.
  3. Solve PnP-RANSAC and refine the pose with Levenberg-Marquardt over the
     inliers.
  4. Triangulate: every inlier keeps the observation at which it was created
     (reference ray). Once the parallax between the reference ray and the
     current ray reaches TRI_MIN_PARALLAX_DEG, the point is re-estimated by
     two-view midpoint triangulation, accepted if it lies in front of both
     cameras and reprojects within TRI_MAX_REPROJ_PX in both views; it is
     re-triangulated whenever the parallax grows. Triangulated inliers are the
     anchors of depth alignment (`anchor_*`): sparse geometry that does not
     come from the depth network.
  5. Create map points for the remaining keypoints by back-projecting them with
     an aligned dense depth map (`spawn_points`).

`track_pose` (steps 1-4) and `spawn_points` (step 5) are separate so that a
synchronous caller can align the depth of the *current* frame between them;
`track` runs both back to back, as the live SLAM thread does.

Map gauge: a monocular map is defined only up to a similarity transformation.
At every (re)initialisation the gauge is fixed by back-projecting the keypoints
with the "gauge depth" of that frame, z = D0 / nu~(u), where nu~ is the depth
network's output normalised by its median (DepthEstimator.gauge_depth): the
frame's median depth becomes D0 map units.
"""

from __future__ import annotations

import hashlib
import logging

import cv2
import numpy as np

from backend.config import Config
from backend.slam.feature_backend import FrameFeatures, make_feature_backend

logger = logging.getLogger("SLAMTracker")

# per-point state (struct of arrays): world position, last (or predicted) pixel, id,
# feature index in the last frame, descriptor of the last observation, frames since the
# last observation (age), reference observation (camera centre, world ray, R_CW, t_CW,
# pixel), parallax of the current triangulation in degrees (0 = none) and its position
_FIELDS = ("p3", "p2", "ids", "fi", "desc", "age", "rc", "rr", "rR", "rt", "rpx", "tri", "xtri")


def _empty_state(desc_dim=128, desc_dtype=np.float32):
    return {"p3": np.zeros((0, 3), np.float32), "p2": np.zeros((0, 2), np.float32),
            "ids": np.zeros((0,), np.int64), "fi": np.zeros((0,), np.int64),
            "desc": np.zeros((0, desc_dim), desc_dtype), "age": np.zeros((0,), np.int32),
            "rc": np.zeros((0, 3)), "rr": np.zeros((0, 3)), "rR": np.zeros((0, 3, 3)),
            "rt": np.zeros((0, 3)), "rpx": np.zeros((0, 2)), "tri": np.zeros((0,), np.float32),
            "xtri": np.zeros((0, 3), np.float32)}


def _take(S, idx):
    return {k: S[k][idx] for k in _FIELDS}


def _cat(*parts):
    parts = [P for P in parts if P["p3"].shape[0] > 0] or [parts[0]]
    return {k: np.concatenate([P[k] for P in parts], 0) for k in _FIELDS}


class SLAMTracker:
    def __init__(self, config: Config, feature_backend=None, seed: int = 0):
        self.config = config
        self.backend = feature_backend or make_feature_backend(config)
        self.K = np.asarray(self.config.INTRINSICS_MATRIX, np.float64)
        self.K_inv = np.linalg.inv(self.K)
        self.rng = np.random.default_rng(seed)
        self.tri_min_deg = float(getattr(config, "TRI_MIN_PARALLAX_DEG", 2.0))
        self.tri_max_px = float(getattr(config, "TRI_MAX_REPROJ_PX", 2.0))
        self.anchor_mode = getattr(config, "ANCHOR_MODE", "triangulated")
        self.tri_rule = getattr(config, "TRI_RULE", "latest")          # "latest" | "grow"
        self.tri_update = bool(getattr(config, "TRI_UPDATE_POINTS", True))

        self.T_WC = np.eye(4, dtype=np.float32)
        self.initialized = False
        self.last_features: FrameFeatures | None = None
        self.S = _empty_state()            # tracked points (against last_features)

        # PnP inliers of the last tracked frame used as alignment anchors
        self.anchor_pts_3d = np.zeros((0, 3), dtype=np.float32)
        self.anchor_px = np.zeros((0, 2), dtype=np.float32)
        self.anchor_ids = np.zeros((0,), dtype=np.int64)
        self.n_inliers = 0
        self.n_triangulated = 0

        self._next_point_id = 0
        # Bumped every time a fresh map is bootstrapped (initial + every
        # reinit). Downstream consumers (alignment, dense fusion) watch this
        # and reset, so geometry from a stale coordinate frame never mixes in.
        self.map_generation = 0
        self._consecutive_failures = 0
        self._failure_tolerance = int(getattr(config, "TRACK_FAILURE_TOLERANCE", 5))
        self._pending = None          # state of a tracked frame awaiting spawn_points()
        self.last_status = "INIT"
        # Ground-truth-pose evaluation (oracle): camera-to-world pose of the current frame,
        # set by the caller before track_pose(); replaces PnP, matches are checked epipolarly.
        self.oracle_T = None
        self.oracle_epi_px = float(getattr(config, "ORACLE_EPIPOLAR_PX", 4.0))
        self.oracle_min_baseline = float(getattr(config, "ORACLE_MIN_BASELINE", 0.01))

    # ----- compatibility views ------------------------------------------
    # (points observed in the last frame; points lost for a few frames are kept in S
    # for re-matching but have no current observation)
    @property
    def _obs(self):
        return self.S["age"] == 0

    @property
    def tracked_pts_3d(self):
        return self.S["p3"][self._obs]

    @property
    def tracked_pts_2d(self):
        return self.S["p2"][self._obs]

    @property
    def tracked_pt_ids(self):
        return self.S["ids"][self._obs]

    @property
    def _tracked_feat_idx(self):
        return self.S["fi"][self._obs]

    # ----- helpers -------------------------------------------------------
    def _spawn_ids(self, n: int) -> np.ndarray:
        ids = np.arange(self._next_point_id, self._next_point_id + n, dtype=np.int64)
        self._next_point_id += n
        return ids

    def _initial_depth_for_batch(self, vs: np.ndarray, h: int) -> np.ndarray:
        """Image-row prior used only when no network depth exists at all
        (top of frame = far, bottom = near; avoids a coplanar initial map)."""
        ny = np.clip(vs / max(1, int(h)), 0.0, 1.0)
        top = float(getattr(self.config, "INIT_DEPTH_TOP_M", 4.0))
        bottom = float(getattr(self.config, "INIT_DEPTH_BOTTOM_M", 1.0))
        return (top + (bottom - top) * ny).astype(np.float32)

    def _depth_at_batch(self, depth_map, us: np.ndarray, vs: np.ndarray, h: int, w: int) -> np.ndarray:
        """Vectorised nearest-pixel depth lookup; NaN where the depth map is invalid
        (0, non-finite or outside [0.1, 10]). Without a depth map: image-row prior."""
        if depth_map is None:
            return self._initial_depth_for_batch(vs, h)
        ui = np.round(us).astype(np.int64)
        vi = np.round(vs).astype(np.int64)
        inb = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        z = np.full(us.shape, np.nan, dtype=np.float32)
        if np.any(inb):
            zz = depth_map[vi[inb], ui[inb]].astype(np.float32)
            zz[~((zz > 0.1) & (zz < 10.0) & np.isfinite(zz))] = np.nan
            z[inb] = zz
        return z

    def _new_points(self, kps_px, zs, feat_idx, T_WC, desc):
        """State for new points created at pixels kps_px with depth zs at pose T_WC."""
        T = np.asarray(T_WC, np.float64)
        n = kps_px.shape[0]
        rays_c = (self.K_inv @ np.c_[kps_px, np.ones(n)].T).T            # z = 1
        p_cam = rays_c * zs[:, None]
        p3 = (p_cam @ T[:3, :3].T + T[:3, 3]).astype(np.float32)
        rw = rays_c @ T[:3, :3].T
        rw /= np.linalg.norm(rw, axis=1, keepdims=True)
        Rcw = T[:3, :3].T
        tcw = -Rcw @ T[:3, 3]
        return {"p3": p3, "p2": kps_px.astype(np.float32), "ids": self._spawn_ids(n),
                "fi": feat_idx.astype(np.int64), "desc": np.asarray(desc), "age": np.zeros((n,), np.int32),
                "rc": np.repeat(T[None, :3, 3], n, 0), "rr": rw,
                "rR": np.repeat(Rcw[None], n, 0), "rt": np.repeat(tcw[None], n, 0),
                "rpx": kps_px.astype(np.float64), "tri": np.zeros((n,), np.float32),
                "xtri": np.full((n, 3), np.nan, np.float32)}

    def _triangulate(self, S, T_WC):
        """Two-view midpoint triangulation of the points in S (current pixels S['p2'])
        against their reference observations; updates S in place. Returns #updated."""
        if S["p3"].shape[0] == 0:
            return 0
        T = np.asarray(T_WC, np.float64)
        c = T[:3, 3]
        d = (self.K_inv @ np.c_[S["p2"].astype(np.float64), np.ones(len(S["p2"]))].T).T @ T[:3, :3].T
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        r0, c0 = S["rr"], S["rc"]
        cosang = np.clip(np.sum(r0 * d, 1), -1.0, 1.0)
        alpha = np.degrees(np.arccos(cosang))
        cand = alpha >= self.tri_min_deg
        if self.tri_rule == "grow":
            cand &= alpha > S["tri"]
        if not np.any(cand):
            return 0
        idx = np.nonzero(cand)[0]
        r0, c0, d1, a = r0[idx], c0[idx], d[idx], cosang[idx]
        w0 = c0 - c
        b = a                              # r0 . d1 (unit rays)
        dd = np.sum(r0 * w0, 1)
        e = np.sum(d1 * w0, 1)
        den = 1.0 - b * b
        l0 = (b * e - dd) / den
        l1 = (e - b * dd) / den
        X = 0.5 * ((c0 + l0[:, None] * r0) + (c + l1[:, None] * d1))
        ok = (l0 > 0) & (l1 > 0) & np.all(np.isfinite(X), 1)
        # reprojection in the current view
        Rcw = T[:3, :3].T
        pc = (X - c) @ Rcw.T
        z1 = pc[:, 2]
        ok &= (z1 > float(getattr(self.config, "DEPTH_Z_MIN", 0.1))) & (z1 < float(getattr(self.config, "DEPTH_Z_MAX", 10.0)))
        uv1 = (pc @ self.K.T)
        uv1 = uv1[:, :2] / np.where(np.abs(uv1[:, 2:3]) > 1e-9, uv1[:, 2:3], 1e-9)
        ok &= np.linalg.norm(uv1 - S["p2"][idx], axis=1) < self.tri_max_px
        # reprojection in the reference view
        pr = np.einsum("nij,nj->ni", S["rR"][idx], X) + S["rt"][idx]
        uv0 = pr @ self.K.T
        uv0 = uv0[:, :2] / np.where(np.abs(uv0[:, 2:3]) > 1e-9, uv0[:, 2:3], 1e-9)
        ok &= (pr[:, 2] > 0) & (np.linalg.norm(uv0 - S["rpx"][idx], axis=1) < self.tri_max_px)
        sel = idx[ok]
        if self.tri_update:
            S["p3"][sel] = X[ok].astype(np.float32)
        # the latest valid triangulation is kept with the point (anchor position)
        S["xtri"][sel] = X[ok].astype(np.float32)
        S["tri"][sel] = alpha[sel].astype(np.float32)
        return int(sel.shape[0])

    # ----- public --------------------------------------------------------
    def initialize_map(self, features: FrameFeatures, depth_map=None) -> bool:
        if features.keypoints.shape[0] < self.config.MIN_TRACKED_POINTS:
            return False
        h, w = features.image_size
        kps = features.keypoints
        zs = self._depth_at_batch(depth_map, kps[:, 0], kps[:, 1], h, w)
        ok = np.isfinite(zs)
        if int(ok.sum()) < self.config.MIN_TRACKED_POINTS:
            return False
        idx = np.nonzero(ok)[0].astype(np.int64)
        if idx.shape[0] > self.config.MAX_FEATURES:      # spatially spread subset of the pool
            idx = idx[_spread(kps[idx], np.zeros((0, 2), np.float32), features.image_size, self.config.MAX_FEATURES)]
        # The map is born at identity pose, so camera frame == world frame (oracle runs: at
        # the ground-truth pose, so the map frame is the ground-truth frame).
        T0 = np.eye(4) if self.oracle_T is None else np.asarray(self.oracle_T, np.float64)
        self.S = self._new_points(kps[idx].astype(np.float64), zs[idx].astype(np.float64), idx, T0,
                                  features.descriptors[idx])
        self.last_features = features
        self.T_WC = T0.astype(np.float32)
        self.initialized = True
        self.map_generation += 1
        self._pending = None
        self.anchor_pts_3d = np.zeros((0, 3), np.float32)
        self.anchor_px = np.zeros((0, 2), np.float32)
        self.anchor_ids = np.zeros((0,), np.int64)
        logger.info(f"SLAM initialized with {idx.shape[0]} map points "
                    f"(gen {self.map_generation}, {self.backend.name}).")
        return True

    def track_pose(self, frame: np.ndarray, dynamic_bboxes=None, init_depth=None):
        """Steps 1-4. Returns (success, T_WC). On success the alignment anchors are in
        `anchor_*` and `spawn_points` must be called to finish the frame.
        `init_depth` (gauge depth of this frame) is used if the map has to be
        (re)initialised; with init_depth=None and REQUIRE_DEPTH_INIT the
        initialisation is postponed."""
        self._pending = None
        require = bool(getattr(self.config, "REQUIRE_DEPTH_INIT", False))

        def _bootstrap(feats):
            self.last_status = "INIT"
            if init_depth is None and require:
                return False, None
            self.initialize_map(feats, depth_map=init_depth)
            self._consecutive_failures = 0
            return False, None

        def _soft_fail(reason: str, feats):
            self._consecutive_failures += 1
            self.last_status = "LOST"
            if self._consecutive_failures >= self._failure_tolerance:
                logger.warning(f"{reason} (x{self._consecutive_failures}); reinitializing map.")
                self._consecutive_failures = 0
                if feats is not None and (init_depth is not None or not require):
                    self.initialize_map(feats, depth_map=init_depth)
                    self.last_status = "REINIT"
            else:
                logger.debug(f"{reason} ({self._consecutive_failures}/{self._failure_tolerance})")
            return False, None

        # 1. Detect features
        features = self.backend.detect(frame)
        if features.keypoints.shape[0] < self.config.MIN_TRACKED_POINTS:
            return _soft_fail("Too few features detected", None)

        # 1b. Drop features inside dynamic-object boxes, unless that would leave
        # too few to track (a person dominating the view): stable tracking with a
        # few moving outliers (rejected by PnP-RANSAC) beats losing the map.
        if dynamic_bboxes:
            filtered = _filter_features_outside_bboxes(features, dynamic_bboxes)
            min_keep = max(self.config.MIN_TRACKED_POINTS,
                           int(getattr(self.config, "DYNAMIC_MASK_MIN_FEATURES", 80)))
            if filtered.keypoints.shape[0] >= min_keep:
                features = filtered

        # 2. Bootstrap
        if not self.initialized or self.last_features is None or self.S["fi"].shape[0] == 0:
            return _bootstrap(features)

        # 3. Match the map points (descriptor of their last observation, at their last or
        # predicted pixel) against every keypoint detected in the new frame. The detection
        # pool (DETECT_FEATURES) is larger than the map (MAX_FEATURES), and a point that is
        # missed is kept for up to MAX_LOST_FRAMES frames, so tracks survive detection gaps.
        S = self.S
        key = hashlib.sha1(S["ids"].tobytes() + S["age"].tobytes() + S["p2"].tobytes()).hexdigest()
        prev = FrameFeatures(keypoints=S["p2"].astype(np.float32), descriptors=S["desc"],
                             image_size=features.image_size, raw_descriptors=None,
                             frame_id=self.last_features.frame_id, subset=("map", key))
        pairs = self.backend.match(prev, features)
        if pairs.shape[0] < self.config.MIN_TRACKED_POINTS:
            return _soft_fail("Insufficient inter-frame matches", features)
        pos = pairs[:, 0].astype(np.int64)                 # index into the tracked set
        curr = pairs[:, 1].astype(np.int64)
        # OpenCV's PnP-RANSAC draws its samples with a fixed internal seed; shuffling
        # the correspondences with the tracker's seeded RNG makes the seed control
        # the random sampling (run-to-run variation of RANSAC).
        perm = self.rng.permutation(pos.shape[0])
        pos, curr = pos[perm], curr[perm]
        pts_3d_arr = self.S["p3"][pos].astype(np.float32)
        pts_2d_arr = features.keypoints[curr].astype(np.float32)

        if self.oracle_T is not None:
            # ground-truth pose; outlier matches are removed by their distance to the epipolar
            # line of the point's reference observation (scale-free, independent of the map)
            T_WC_new = np.asarray(self.oracle_T, np.float64)
            inliers = self._epipolar_inliers(pos, pts_2d_arr, T_WC_new)
            if inliers.shape[0] < self.config.MIN_TRACKED_POINTS:
                return _soft_fail("Too few epipolar inliers", features)
        else:
            # PnP-RANSAC (8 px, 100 iterations, confidence 0.99) + LM refinement
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts_3d_arr, pts_2d_arr, self.K, None, iterationsCount=100,
                reprojectionError=8.0, confidence=0.99, flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not success or inliers is None or len(inliers) < self.config.MIN_TRACKED_POINTS:
                return _soft_fail("PnP failed or too few inliers", features)
            inliers = inliers.flatten()
            if len(inliers) >= 6:
                try:
                    rvec, tvec = cv2.solvePnPRefineLM(pts_3d_arr[inliers], pts_2d_arr[inliers], self.K, None, rvec, tvec)
                except cv2.error:
                    pass

            R_CW, _ = cv2.Rodrigues(rvec)
            T_CW = np.eye(4, dtype=np.float64)
            T_CW[:3, :3] = R_CW
            T_CW[:3, 3] = tvec.flatten()
            if not np.all(np.isfinite(T_CW)):
                return _soft_fail("PnP returned non-finite pose", features)
            try:
                T_WC_new = np.linalg.inv(T_CW)
            except np.linalg.LinAlgError:
                return _soft_fail("PnP pose is singular", features)
            if not np.all(np.isfinite(T_WC_new)):
                return _soft_fail("Inverted pose is non-finite", features)

        self.T_WC = T_WC_new.astype(np.float32)
        self._consecutive_failures = 0

        # 4. Inlier state (fresh observation), lost points, triangulation, anchors
        S_in = _take(self.S, pos[inliers])
        S_in["p2"] = pts_2d_arr[inliers]
        S_in["fi"] = curr[inliers]
        S_in["desc"] = features.descriptors[curr[inliers]]
        S_in["age"] = np.zeros(S_in["p3"].shape[0], np.int32)
        seen = np.zeros(self.S["p3"].shape[0], bool)
        seen[pos] = True                                   # matched (inlier or rejected outlier)
        lost = np.nonzero(~seen)[0]
        S_lost = _take(self.S, lost)
        S_lost["age"] = S_lost["age"] + 1
        keep = S_lost["age"] <= int(getattr(self.config, "MAX_LOST_FRAMES", 3))
        if lost.shape[0]:                                   # predicted pixel with the new pose
            pc = (S_lost["p3"].astype(np.float64) - T_WC_new[:3, 3]) @ T_WC_new[:3, :3]
            h_, w_ = features.image_size
            z_ = pc[:, 2]
            uv = (pc @ self.K.T)[:, :2] / np.where(z_[:, None] > 1e-6, z_[:, None], 1e-6)
            keep &= (z_ > 0.05) & (uv[:, 0] >= 0) & (uv[:, 0] < w_) & (uv[:, 1] >= 0) & (uv[:, 1] < h_)
            S_lost["p2"] = uv.astype(np.float32)
        S_lost = _take(S_lost, np.nonzero(keep)[0])
        S_lost["fi"] = np.full(S_lost["p3"].shape[0], -1, np.int64)
        self._triangulate(S_in, T_WC_new)
        self.n_inliers = int(S_in["p3"].shape[0])
        has_tri = np.all(np.isfinite(S_in["xtri"]), axis=1)
        self.n_triangulated = int(has_tri.sum())
        if self.anchor_mode == "triangulated":     # geometric anchors: the latest valid triangulation
            anchor = has_tri
            P_anchor = S_in["p3"] if self.tri_update else S_in["xtri"]
        else:                                      # all inliers at their map positions
            anchor = np.ones(self.n_inliers, bool)
            P_anchor = S_in["p3"]
        self.anchor_pts_3d = P_anchor[anchor].astype(np.float32)
        self.anchor_px = S_in["p2"][anchor]
        self.anchor_ids = S_in["ids"][anchor]
        self._pending = (features, S_in, S_lost)
        self.last_status = "TRACKING"
        return True, self.T_WC.copy()

    def _epipolar_inliers(self, pos, pts_2d, T_WC):
        """Indices (into pos) of matches whose current pixel lies within ORACLE_EPIPOLAR_PX of the
        epipolar line of the point's reference observation; matches whose reference camera is
        closer than ORACLE_MIN_BASELINE to the current one are kept (no constraint)."""
        S = self.S
        c0, r0 = S["rc"][pos].astype(np.float64), S["rr"][pos].astype(np.float64)
        c = T_WC[:3, 3]
        d = (self.K_inv @ np.c_[pts_2d.astype(np.float64), np.ones(len(pts_2d))].T).T @ T_WC[:3, :3].T
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        base = c - c0
        nrm = np.cross(base, r0)
        nn = np.linalg.norm(nrm, axis=1)
        sin_off = np.abs(np.sum(nrm * d, 1)) / np.maximum(nn, 1e-12)      # sine of the angle to the epipolar plane
        err_px = self.K[0, 0] * sin_off / np.sqrt(np.maximum(1.0 - sin_off ** 2, 1e-12))
        short = np.linalg.norm(base, axis=1) < self.oracle_min_baseline
        return np.nonzero(short | (err_px < self.oracle_epi_px))[0].astype(np.int64)

    def anchors_camera(self):
        """Alignment anchors of the last tracked frame: (pixels (N, 2), camera-frame depth (N,))."""
        if self.anchor_pts_3d.shape[0] == 0:
            return np.zeros((0, 2), np.float32), np.zeros((0,), np.float32)
        T_CW = np.linalg.inv(self.T_WC.astype(np.float64))
        zc = self.anchor_pts_3d.astype(np.float64) @ T_CW[2, :3] + T_CW[2, 3]
        return self.anchor_px.copy(), zc.astype(np.float32)

    def spawn_points(self, depth_map=None):
        """Step 5: create map points for keypoints without one, by back-projecting them
        with `depth_map` (NaN/0 = no point; None = no new points), then cap the tracked set."""
        if self._pending is None:
            return self.tracked_pts_3d.copy()
        features, S_in, S_lost = self._pending
        self._pending = None
        N = features.keypoints.shape[0]
        unmatched = np.ones(N, dtype=bool)
        if S_in["fi"].shape[0] > 0:
            unmatched[S_in["fi"]] = False
        S = _cat(S_in, S_lost)
        room = self.config.MAX_FEATURES - S["p3"].shape[0]
        if depth_map is not None and room > 0 and np.any(unmatched):
            h, w = features.image_size
            new_idx = np.nonzero(unmatched)[0].astype(np.int64)      # detection (score) order
            us, vs = features.keypoints[new_idx, 0], features.keypoints[new_idx, 1]
            zs = self._depth_at_batch(depth_map, us, vs, h, w)
            ok = np.isfinite(zs)
            new_idx, zs = new_idx[ok], zs[ok]
            if new_idx.shape[0]:
                pick = _spread(features.keypoints[new_idx], S["p2"], (h, w), room)
                new_idx, zs = new_idx[pick], zs[pick]
                S = _cat(S, self._new_points(features.keypoints[new_idx].astype(np.float64),
                                             zs.astype(np.float64), new_idx, self.T_WC,
                                             features.descriptors[new_idx]))
        self.S = S
        self.last_features = features
        return self.S["p3"].copy()

    def track(self, frame: np.ndarray, depth_map=None, dynamic_bboxes=None, init_depth=None):
        """Live path: track_pose + spawn_points with the most recent depth map.
        Returns (success, T_WC, tracked_pts_3d)."""
        ok, T = self.track_pose(frame, dynamic_bboxes=dynamic_bboxes,
                                init_depth=init_depth if init_depth is not None else depth_map)
        if not ok:
            return False, None, None
        pts = self.spawn_points(depth_map)
        return True, T, pts

    def update_points(self, pid_to_xyz: dict):
        """Write bundle-adjusted coordinates back into the tracked working set."""
        if not pid_to_xyz or self.S["ids"].shape[0] == 0:
            return 0
        n = 0
        for i, pid in enumerate(self.S["ids"].tolist()):
            xyz = pid_to_xyz.get(pid)
            if xyz is not None:
                self.S["p3"][i] = xyz
                n += 1
        return n


def _subset_features(features: FrameFeatures, idx: np.ndarray) -> FrameFeatures:
    """Keypoints/descriptors `idx` of a frame as a new FrameFeatures (same frame_id)."""
    raw = features.raw_descriptors
    if raw is not None:
        try:
            import torch
            if isinstance(raw, torch.Tensor):
                raw = raw[torch.from_numpy(np.asarray(idx, np.int64)).to(raw.device)]
            else:
                raw = raw[idx]
        except Exception as e:
            if isinstance(e, MemoryError) or "memory" in str(e).lower():
                raise  # out of host or GPU memory: never continue silently
            raw = None
    desc = features.descriptors[idx] if features.descriptors is not None else None
    return FrameFeatures(keypoints=features.keypoints[idx], descriptors=desc, image_size=features.image_size,
                         raw_descriptors=raw, frame_id=features.frame_id,
                         subset=(features.subset, "tracked", np.asarray(idx, np.int64).tobytes()))


def _spread(cand_px: np.ndarray, existing_px: np.ndarray, size, budget: int, grid=(8, 6)) -> np.ndarray:
    """Pick up to `budget` candidates (given in score order) so that new map points
    fill the least-occupied cells of a grid first. Returns candidate indices."""
    h, w = size
    gx, gy = grid
    cell = lambda p: (np.clip((p[:, 0] * gx / w).astype(int), 0, gx - 1) * gy
                      + np.clip((p[:, 1] * gy / h).astype(int), 0, gy - 1))
    occ = np.bincount(cell(existing_px), minlength=gx * gy) if existing_px.shape[0] else np.zeros(gx * gy, int)
    cc = cell(cand_px)
    chosen = []
    cap = occ.min() + 1
    taken = np.zeros(cand_px.shape[0], bool)
    while len(chosen) < budget and not taken.all():
        added = False
        for i in np.nonzero(~taken)[0]:
            if occ[cc[i]] < cap:
                chosen.append(i); taken[i] = True; occ[cc[i]] += 1; added = True
                if len(chosen) >= budget:
                    break
        cap += 1
        if not added and cap > occ.max() + 1:
            break
    return np.asarray(chosen, np.int64)


def _filter_features_outside_bboxes(features: FrameFeatures, bboxes):
    """Return a new FrameFeatures containing only keypoints that fall
    OUTSIDE every box in `bboxes`. Boxes are (x1, y1, x2, y2) pixel ints."""
    if features.keypoints.shape[0] == 0 or not bboxes:
        return features

    kps = features.keypoints
    keep = np.ones(kps.shape[0], dtype=bool)
    for (x1, y1, x2, y2) in bboxes:
        inside = (kps[:, 0] >= x1) & (kps[:, 0] <= x2) & (kps[:, 1] >= y1) & (kps[:, 1] <= y2)
        keep &= ~inside

    if keep.all():
        return features

    new_kps = kps[keep].copy()
    desc = features.descriptors
    new_desc = desc[keep] if (desc is not None and len(desc) == len(keep)) else desc

    new_raw = features.raw_descriptors
    if new_raw is not None:
        try:
            import torch
            if isinstance(new_raw, torch.Tensor):
                mask = torch.from_numpy(keep).to(device=new_raw.device)
                new_raw = new_raw[mask]
            else:
                new_raw = new_raw[keep]
        except Exception:
            new_raw = None

    # frame_id/subset identify the keypoint subset (used by the replay cache of matches)
    return FrameFeatures(keypoints=new_kps, descriptors=new_desc,
                         image_size=features.image_size, raw_descriptors=new_raw,
                         frame_id=features.frame_id,
                         subset=(features.subset, np.packbits(keep).tobytes()))
