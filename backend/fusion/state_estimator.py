"""Single source of truth for the consolidated multi-modal state.

All perception threads write into this object; the WebSocket broadcaster
reads `get_consolidated_state()` once per tick to ship a payload.
"""

import numpy as np
import threading
import logging
from backend.config import Config
from backend.slam.mapper import SLAMMapper
from backend.semantic.semantic_map import SemanticMap

logger = logging.getLogger("StateEstimator")


class StateEstimator:
    def __init__(self, config: Config):
        self.config = config
        self.lock = threading.Lock()

        self.current_pose = np.eye(4, dtype=np.float32)
        # Smoothed copy used only for broadcasting. We expose this through
        # `get_consolidated_state()` so the UI camera frustum stops
        # jittering under PnP noise. SLAM internals still see the raw pose.
        self.broadcast_pose = np.eye(4, dtype=np.float32)
        self._pose_smooth_alpha = float(getattr(config, "POSE_SMOOTH_ALPHA", 0.35))
        self.trajectory = []
        self.map_points = []
        self.map_colors = []
        self.semantic_objects = []
        self.scene_graph = {"nodes": [], "edges": []}
        self.scene_description = []
        self.loop_events = []

        self.tracking_status = "INIT"

        # Microphone sound/voice-activity detection. `azimuth_deg`/`source`
        # are filled from the detected person when one is speaking (camera-
        # derived direction; a laptop has no usable mic array for true TDOA).
        self.audio = {"available": False, "energy": 0.0, "level": 0.0,
                      "vad_active": False, "type": "no_mic",
                      "azimuth_deg": 0.0, "source": "none"}

        # Closest detected person (for distance readout + audio direction +
        # masking them out of the environment reconstruction).
        self.person = {"present": False, "distance_m": 0.0,
                       "azimuth_deg": 0.0, "conf": 0.0, "position_3d": None}

        self.telemetry = {
            "fps": 0.0,
            "latency_ms": 0.0,
            "slam_ms": 0.0,
            "semantic_ms": 0.0,
            "tracked_points_count": 0,
            "keyframes_count": 0,
            "vram_mb": 0.0,
            "ram_pct": 0.0,
            "status": self.tracking_status,
            "feature_backend": "?",
        }

    def set_tracking_status(self, status: str):
        with self.lock:
            self.tracking_status = status
            self.telemetry["status"] = status

    def set_feature_backend(self, name: str):
        with self.lock:
            self.telemetry["feature_backend"] = name

    def update_pose_and_map(self, pose, mapper: SLAMMapper):
        with self.lock:
            if pose is not None and np.all(np.isfinite(pose)):
                self.current_pose = pose.copy()
                # EMA on translation; rotation is slerp-style via matrix blend
                # which is approximate but fine for visualization purposes.
                a = self._pose_smooth_alpha
                self.broadcast_pose[:3, 3] = (
                    (1 - a) * self.broadcast_pose[:3, 3] + a * pose[:3, 3]
                ).astype(np.float32)
                # Orthonormalise the rotation blend via SVD so it stays valid
                R_blend = (1 - a) * self.broadcast_pose[:3, :3] + a * pose[:3, :3]
                try:
                    U, _, Vt = np.linalg.svd(R_blend)
                    Rn = U @ Vt
                    if np.linalg.det(Rn) < 0:
                        U[:, -1] *= -1
                        Rn = U @ Vt
                    self.broadcast_pose[:3, :3] = Rn.astype(np.float32)
                except np.linalg.LinAlgError:
                    self.broadcast_pose[:3, :3] = pose[:3, :3].astype(np.float32)
            map_data = mapper.get_map_data()
            self.map_points = map_data["points"]
            self.map_colors = map_data["colors"]
            self.trajectory = map_data["trajectory"]
            self.loop_events = map_data.get("loop_events", [])
            self.telemetry["tracked_points_count"] = len(self.map_points)
            self.telemetry["keyframes_count"] = len(mapper.keyframes)

    def update_semantics(self, sem_map: SemanticMap):
        with self.lock:
            self.semantic_objects = sem_map.get_semantic_objects()
            self.scene_graph = sem_map.get_scene_graph()
            self.scene_description = sem_map.get_scene_description()

    def update_audio(self, event):
        if event is None:
            return
        with self.lock:
            # Fuse camera-derived direction: a speaking person is the source.
            if event.get("vad_active") and self.person.get("present"):
                event["azimuth_deg"] = self.person["azimuth_deg"]
                event["source"] = "person"
            else:
                event.setdefault("azimuth_deg", 0.0)
                event["source"] = "none"
            self.audio = event

    def update_person(self, person):
        with self.lock:
            self.person = person

    def update_telemetry(self, key: str, val):
        with self.lock:
            self.telemetry[key] = val

    def get_map_state(self):
        """Heavy payload (sparse point cloud + trajectory) broadcast at a
        lower rate than the light state, so JSON-serialising thousands of
        points doesn't hog the event loop / GIL every state tick."""
        with self.lock:
            return {
                "points": self.map_points,
                "colors": self.map_colors,
                "trajectory": self.trajectory,
            }

    def get_consolidated_state(self, include_map: bool = True):
        with self.lock:
            # Use the smoothed pose for broadcasting; it keeps the dashboard
            # frustum and trajectory ribbon visually stable. SLAM internals
            # still see the raw pose for accurate next-frame matching.
            pose = self.broadcast_pose
            cam_position = pose[:3, 3].tolist()
            R = pose[:3, :3]
            tr = np.trace(R)
            if tr > 0:
                S = np.sqrt(tr + 1.0) * 2
                qw = 0.25 * S
                qx = (R[2, 1] - R[1, 2]) / S
                qy = (R[0, 2] - R[2, 0]) / S
                qz = (R[1, 0] - R[0, 1]) / S
            else:
                qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
            q_norm = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz + 1e-9)
            orientation_q = [float(qw / q_norm), float(qx / q_norm),
                             float(qy / q_norm), float(qz / q_norm)]

            state = {
                "camera": {
                    "position": cam_position,
                    "orientation": orientation_q,
                    "matrix": pose.tolist(),
                },
                "objects": self.semantic_objects,
                "audio": self.audio,
                "person": self.person,
                "scene_graph": self.scene_graph,
                "scene_description": self.scene_description,
                "loop_events": self.loop_events,
                "telemetry": self.telemetry.copy(),
                "tracking_status": self.tracking_status,
            }
            if include_map:
                state["trajectory"] = self.trajectory
                state["points"] = self.map_points
                state["colors"] = self.map_colors
            return state


class SensorFusionEstimator(StateEstimator):
    pass
