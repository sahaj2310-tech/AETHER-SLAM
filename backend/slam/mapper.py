from __future__ import annotations

import time
import numpy as np
import logging
from backend.config import Config

logger = logging.getLogger("SLAMMapper")


class KeyFrame:
    def __init__(self, kf_id: int, T_WC: np.ndarray, pts_2d: np.ndarray,
                 pts_3d: np.ndarray, pt_ids: np.ndarray):
        self.id = kf_id
        self.T_WC = T_WC.copy()
        self.pts_2d = pts_2d.copy()
        self.pts_3d = pts_3d.copy()
        self.pt_ids = pt_ids.copy()


class SLAMMapper:
    def __init__(self, config: Config):
        self.config = config
        self.keyframes = []
        self.next_kf_id = 0

        # Voxel-keyed point store: voxel_key -> (point_id, xyz, rgb)
        self.point_by_id = {}      # int point_id -> [x, y, z, r, g, b]
        self.voxel_to_pid = {}     # voxel_tuple -> point_id

        # Loop closure events visible to the frontend (last N).
        self.loop_events = []
        self._last_kf_ts = 0.0

    def _voxel_key(self, pt):
        v = self.config.VOXEL_SIZE
        return (int(round(pt[0] / v)), int(round(pt[1] / v)), int(round(pt[2] / v)))

    def is_keyframe_needed(self, current_pose: np.ndarray, t: float | None = None) -> bool:
        """`t` is the frame timestamp (seconds); wall-clock time is used when omitted."""
        now = time.time() if t is None else float(t)
        if not self.keyframes:
            self._last_kf_ts = now
            return True

        # Time-rate limit: never insert two keyframes faster than KEYFRAME_MIN_DT.
        # This kills the "300 keyframes while stationary" pathology without
        # blocking real rotation: a steady pan past the room will still
        # produce a keyframe every KEYFRAME_MIN_DT seconds, which is plenty
        # to cover a full sweep.
        if (now - self._last_kf_ts) < getattr(self.config, "KEYFRAME_MIN_DT", 0.0):
            return False

        last_kf = self.keyframes[-1]
        T_last = last_kf.T_WC

        t_diff = float(np.linalg.norm(current_pose[:3, 3] - T_last[:3, 3]))

        R_diff = T_last[:3, :3].T @ current_pose[:3, :3]
        cos_theta = (np.trace(R_diff) - 1.0) / 2.0
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        r_diff_deg = float(np.rad2deg(np.arccos(cos_theta)))

        needed = (t_diff > self.config.KEYFRAME_MIN_TRANSLATION or
                  r_diff_deg > self.config.KEYFRAME_MIN_ROTATION)
        if needed:
            self._last_kf_ts = now
        return needed

    def add_keyframe(self, T_WC, pts_2d, pts_3d, pt_ids, frame_rgb):
        kf = KeyFrame(self.next_kf_id, T_WC, pts_2d, pts_3d, pt_ids)
        self.keyframes.append(kf)
        self.next_kf_id += 1

        # Cap keyframe history so a long scan doesn't grow RAM (and the
        # trajectory payload) without bound. BA only uses the recent window and
        # loop closure keeps its own capped record set, so dropping the oldest
        # keyframes is safe; the trajectory just shows the most recent path.
        max_kf = int(getattr(self.config, "MAX_KEYFRAMES", 600))
        if len(self.keyframes) > max_kf:
            self.keyframes = self.keyframes[-max_kf:]

        h, w, _ = frame_rgb.shape
        for i, pt_3d in enumerate(pts_3d):
            pid = int(pt_ids[i])
            u, v = int(round(pts_2d[i][0])), int(round(pts_2d[i][1]))
            u = max(0, min(w - 1, u))
            v = max(0, min(h - 1, v))
            bgr = frame_rgb[v, u]
            rgb = (int(bgr[2]), int(bgr[1]), int(bgr[0]))

            voxel = self._voxel_key(pt_3d)
            existing_pid = self.voxel_to_pid.get(voxel)
            if existing_pid is None:
                self.point_by_id[pid] = [float(pt_3d[0]), float(pt_3d[1]), float(pt_3d[2]),
                                         rgb[0], rgb[1], rgb[2]]
                self.voxel_to_pid[voxel] = pid
            else:
                # Update colour with simple average; coordinate stays at first observation
                old = self.point_by_id[existing_pid]
                old[3] = (old[3] + rgb[0]) // 2
                old[4] = (old[4] + rgb[1]) // 2
                old[5] = (old[5] + rgb[2]) // 2

        # Cap map size
        if len(self.point_by_id) > 100000:
            # Drop oldest half (lowest IDs)
            sorted_ids = sorted(self.point_by_id.keys())
            keep = set(sorted_ids[-50000:])
            self.point_by_id = {pid: v for pid, v in self.point_by_id.items() if pid in keep}
            self.voxel_to_pid = {vox: pid for vox, pid in self.voxel_to_pid.items() if pid in keep}

        logger.debug(f"Keyframe {kf.id} added. Map points: {len(self.point_by_id)}")
        return kf.id

    def update_point(self, pid: int, new_xyz):
        if pid in self.point_by_id:
            self.point_by_id[pid][0] = float(new_xyz[0])
            self.point_by_id[pid][1] = float(new_xyz[1])
            self.point_by_id[pid][2] = float(new_xyz[2])

    def record_loop_event(self, event):
        """Persist a loop-closure event for telemetry / visualization."""
        if event is None:
            return
        self.loop_events.append({
            "current_kf_id": event.current_kf_id,
            "matched_kf_id": event.matched_kf_id,
            "score": float(event.score),
            "n_matches": int(event.n_matches),
        })
        # keep only the last 20 events
        self.loop_events = self.loop_events[-20:]

    def get_map_data(self):
        trajectory = [kf.T_WC[:3, 3].tolist() for kf in self.keyframes]
        if not trajectory:
            trajectory = [[0.0, 0.0, 0.0]]

        if not self.point_by_id:
            return {"points": [], "colors": [], "trajectory": trajectory}

        all_vals = list(self.point_by_id.values())
        pts = [[v[0], v[1], v[2]] for v in all_vals]
        cols = [[v[3], v[4], v[5]] for v in all_vals]

        # Downsample for streaming: keep the per-tick WS payload (and the
        # JSON-serialisation cost on the broadcast thread) bounded.
        cap = int(getattr(self.config, "STREAM_MAX_POINTS", 2500))
        if len(pts) > cap:
            step = len(pts) // cap
            pts = pts[::step]
            cols = cols[::step]

        return {
            "points": pts,
            "colors": cols,
            "trajectory": trajectory,
            "loop_events": list(self.loop_events),
        }
