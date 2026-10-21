"""YOLOv8-nano detection + DeepSORT multi-object tracking + 3D back-projection.

Pipeline per call:
  1. Run YOLOv8 inference (FP16 on CUDA when available).
  2. Hand the 2D detections to DeepSORT, which returns persistent track IDs
     using a MobileNet re-ID embedder + Kalman motion model.
  3. For each surviving track, back-project the masked (or bounding-box)
     pixels into the world frame using the dense depth map and the SLAM
     camera pose.

Why DeepSORT here:
  * Persistent IDs survive brief occlusions and YOLO score dips.
  * The track ID can be carried into the semantic map for stable identity
    across the entire session (used by the scene graph as the canonical
    node key).
"""

from __future__ import annotations

import logging
from typing import List, Dict, Optional

import cv2
import numpy as np

from backend.config import Config

logger = logging.getLogger("YOLODetector")

YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    logger.warning("ultralytics not found. Running mock detector.")

DEEPSORT_AVAILABLE = False
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    DEEPSORT_AVAILABLE = True
except ImportError:
    logger.warning("deep_sort_realtime not found. Tracking IDs will not persist.")


class YOLODetector:
    def __init__(self, config: Config, person_only: bool = False):
        self.config = config
        # person_only = lightweight mode: YOLO person boxes for distance /
        # audio-direction / mesh-masking, with NO DeepSORT re-ID and no
        # multi-class scene registry. Much cheaper than full semantics.
        self.person_only = person_only
        self.model = None
        self.device = "cpu"
        self.use_half = False
        self.img_size = int(config.YOLO_IMG_SIZE)

        if YOLO_AVAILABLE:
            try:
                logger.info(f"Loading YOLO model: {self.config.YOLO_MODEL_NAME}...")
                self.model = YOLO(self.config.YOLO_MODEL_NAME)
                import torch
                if torch.cuda.is_available():
                    self.model.to("cuda")
                    self.device = "cuda"
                    self.use_half = bool(self.config.YOLO_HALF_PRECISION)
                    logger.info(f"YOLO on CUDA (half={self.use_half}).")
                else:
                    logger.info("YOLO on CPU.")
            except Exception as e:
                logger.error(f"YOLO load failed: {e}. Mock detector.")
                self.model = None

        # DeepSORT (skipped entirely in person_only mode; it's pure overhead
        # when we only need raw person boxes for distance + direction).
        self.tracker = None
        if DEEPSORT_AVAILABLE and not self.person_only:
            try:
                self.tracker = DeepSort(
                    max_age=self.config.DEEPSORT_MAX_AGE,
                    n_init=self.config.DEEPSORT_N_INIT,
                    nms_max_overlap=1.0,
                    max_cosine_distance=0.3,
                    embedder="mobilenet",
                    embedder_gpu=(self.device == "cuda"),
                    half=(self.device == "cuda"),
                    bgr=True,
                )
                logger.info("DeepSORT tracker ready (mobilenet embedder).")
            except Exception as e:
                logger.error(f"DeepSORT init failed: {e}.")
                self.tracker = None

    # ---------------------------------------------------------------
    def detect_person_boxes(self, frame) -> List[Dict]:
        """Every person detection (class 0) with confidence >= PERSON_CONFIDENCE
        and area >= PERSON_MIN_BOX_FRAC of the frame: [{xyxy, conf}].
        Independent of depth, so the dynamic mask is the same for every
        alignment variant."""
        if self.model is None or not YOLO_AVAILABLE:
            return []
        conf_thr = float(getattr(self.config, "PERSON_CONFIDENCE", self.config.CONFIDENCE_THRESHOLD))
        try:
            results = self.model.predict(frame, conf=conf_thr, imgsz=self.img_size, half=self.use_half,
                                         device=self.device, classes=[0], verbose=False)
        except Exception as e:
            logger.error(f"YOLO person inference error: {e}")
            return []
        if not results or results[0].boxes is None:
            return []
        H, W = frame.shape[:2]
        min_area = float(getattr(self.config, "PERSON_MIN_BOX_FRAC", 0.012)) * (H * W)
        boxes = results[0].boxes
        xyxy = boxes.xyxy.detach().cpu().numpy()
        conf = boxes.conf.detach().cpu().numpy()
        cls = boxes.cls.detach().cpu().numpy()
        out = []
        for i in range(len(conf)):
            if int(cls[i]) != 0:
                continue
            x1, y1, x2, y2 = xyxy[i]
            x1 = int(max(0, min(W - 1, x1))); x2 = int(max(0, min(W - 1, x2)))
            y1 = int(max(0, min(H - 1, y1))); y2 = int(max(0, min(H - 1, y2)))
            if x2 - x1 < 4 or y2 - y1 < 4 or (x2 - x1) * (y2 - y1) < min_area:
                continue
            out.append({"xyxy": [x1, y1, x2, y2], "conf": float(conf[i])})
        return out

    def detect_persons(self, frame, depth_map) -> List[Dict]:
        """Lightweight person detection: returns each person as
        {xyxy, conf, distance_m, azimuth_rad}. distance is the median dense
        depth inside the box (metres, monocular scale); azimuth is the
        horizontal bearing of the box centre from the optical axis
        (positive = camera-right). No DeepSORT, no 3D registry."""
        if self.model is None or not YOLO_AVAILABLE or depth_map is None:
            return []
        # Stricter confidence for person reporting than the general threshold,
        # to avoid hallucinated boxes (a phantom "person" gives a phantom
        # distance). Temporal confirmation downstream filters the rest.
        conf_thr = float(getattr(self.config, "PERSON_CONFIDENCE",
                                 self.config.CONFIDENCE_THRESHOLD))
        try:
            results = self.model.predict(
                frame, conf=conf_thr,
                imgsz=self.img_size, half=self.use_half,
                device=self.device, classes=[0], verbose=False,
            )
        except Exception as e:
            logger.error(f"YOLO person inference error: {e}")
            return []
        if not results or results[0].boxes is None:
            return []

        H, W = depth_map.shape
        fx, cx = self.config.FX, self.config.CX
        # Reject implausibly small boxes (a real person occupies a fair chunk
        # of the frame); these are the usual false positives.
        min_area = float(getattr(self.config, "PERSON_MIN_BOX_FRAC", 0.012)) * (H * W)
        boxes = results[0].boxes
        out = []
        for i in range(len(boxes)):
            if int(boxes.cls[i].item()) != 0:
                continue
            x1, y1, x2, y2 = boxes.xyxy[i].detach().cpu().numpy().astype(int)
            x1 = max(0, min(W - 1, int(x1))); x2 = max(0, min(W - 1, int(x2)))
            y1 = max(0, min(H - 1, int(y1))); y2 = max(0, min(H - 1, int(y2)))
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            if (x2 - x1) * (y2 - y1) < min_area:
                continue
            roi = depth_map[y1:y2 + 1, x1:x2 + 1]
            valid = roi[(roi > 0.1) & (roi < 10.0)]
            if valid.size < 8:
                continue
            distance = float(np.median(valid))
            u_c = 0.5 * (x1 + x2)
            azimuth = float(np.arctan2(u_c - cx, fx))
            out.append({
                "xyxy": [int(x1), int(y1), int(x2), int(y2)],
                "conf": float(boxes.conf[i].item()),
                "distance_m": round(distance, 2),
                "azimuth_rad": azimuth,
            })
        # Closest person first.
        out.sort(key=lambda p: p["distance_m"])
        return out

    # ---------------------------------------------------------------
    def detect(self, frame, depth_map, T_WC) -> List[Dict]:
        if self.model is None or not YOLO_AVAILABLE:
            return []  # no model → no detections (never fabricate objects)

        try:
            results = self.model.predict(
                frame,
                conf=self.config.CONFIDENCE_THRESHOLD,
                imgsz=self.img_size,
                half=self.use_half,
                device=self.device,
                verbose=False,
            )
        except Exception as e:
            logger.error(f"YOLO inference error: {e}")
            return []

        if not results:
            return []

        result = results[0]
        boxes = result.boxes
        masks = result.masks

        if boxes is None or len(boxes) == 0:
            # Still tick DeepSORT so its Kalman states age
            if self.tracker is not None:
                self.tracker.update_tracks([], frame=frame)
            return []

        H, W = depth_map.shape

        # --- Build the input DeepSORT expects ---------------------------
        ds_detections = []  # list of ([x, y, w, h], conf, class_name)
        raw_dets = []
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            if cls_id not in self.config.CLASS_LABELS:
                continue
            label = self.config.CLASS_LABELS[cls_id]
            conf = float(boxes.conf[i].item())
            xyxy = boxes.xyxy[i].detach().cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            x1 = max(0, min(W - 1, int(x1)))
            y1 = max(0, min(H - 1, int(y1)))
            x2 = max(0, min(W - 1, int(x2)))
            y2 = max(0, min(H - 1, int(y2)))
            if x2 <= x1 or y2 <= y1:
                continue
            ds_detections.append(([x1, y1, x2 - x1, y2 - y1], conf, label))
            raw_dets.append({
                "label": label, "conf": conf,
                "xyxy": (x1, y1, x2, y2), "yolo_idx": i,
            })

        # --- DeepSORT update -------------------------------------------
        tracks = []
        if self.tracker is not None:
            tracks = self.tracker.update_tracks(ds_detections, frame=frame)

        # Index DeepSORT tracks by closest raw detection (IoU on bbox)
        track_to_raw = self._match_tracks_to_dets(tracks, raw_dets)

        fx, fy = self.config.FX, self.config.FY
        cx, cy = self.config.CX, self.config.CY
        detections = []

        for tr in tracks:
            if not tr.is_confirmed():
                continue
            tr_idx = track_to_raw.get(int(tr.track_id))
            if tr_idx is None:
                continue
            raw = raw_dets[tr_idx]
            x1, y1, x2, y2 = raw["xyxy"]

            mask_2d = None
            if masks is not None and raw["yolo_idx"] < len(masks.data):
                m = masks.data[raw["yolo_idx"]].detach().cpu().numpy()
                if m.size > 0:
                    mask_2d = (m > 0.5).astype(np.uint8)
                    if mask_2d.shape != (H, W):
                        mask_2d = cv2.resize(mask_2d, (W, H),
                                             interpolation=cv2.INTER_NEAREST)

            if mask_2d is not None and mask_2d.any():
                y_idx, x_idx = np.where(mask_2d)
            else:
                if x2 <= x1 or y2 <= y1:
                    continue
                yy, xx = np.meshgrid(np.arange(y1, y2 + 1),
                                     np.arange(x1, x2 + 1), indexing='ij')
                y_idx = yy.ravel()
                x_idx = xx.ravel()

            if x_idx.size == 0:
                continue

            object_depths = depth_map[y_idx, x_idx]
            valid = (object_depths > 0.2) & (object_depths < 10.0)
            if int(np.sum(valid)) < 5:
                continue

            vd = object_depths[valid]
            vy = y_idx[valid]
            vx = x_idx[valid]

            x_c = (vx - cx) * vd / fx
            y_c = (vy - cy) * vd / fy
            pts_cam = np.stack([x_c, y_c, vd, np.ones_like(vd)], axis=1)
            pts_world = (T_WC @ pts_cam.T).T[:, :3]

            centroid_3d = np.mean(pts_world, axis=0)
            min_b = np.min(pts_world, axis=0)
            max_b = np.max(pts_world, axis=0)
            bbox_dims = [max(float(d), 0.1) for d in (max_b - min_b)]

            detections.append({
                "track_id": int(tr.track_id),
                "label": raw["label"],
                "confidence": raw["conf"],
                "centroid_3d": centroid_3d.tolist(),
                "bbox_3d": bbox_dims,
                "xyxy": [int(x1), int(y1), int(x2), int(y2)],
                "depth_median": float(np.median(vd)),
            })

        return detections

    # ---------------------------------------------------------------
    @staticmethod
    def _match_tracks_to_dets(tracks, raw_dets):
        """Return {track_id -> index into raw_dets} by highest IoU."""
        out = {}
        if not tracks or not raw_dets:
            return out
        for tr in tracks:
            tx1, ty1, tx2, ty2 = tr.to_ltrb()
            best_iou = 0.0
            best_idx = None
            for i, d in enumerate(raw_dets):
                dx1, dy1, dx2, dy2 = d["xyxy"]
                iou = _iou_xyxy((tx1, ty1, tx2, ty2), (dx1, dy1, dx2, dy2))
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_idx is not None and best_iou > 0.3:
                out[int(tr.track_id)] = best_idx
        return out


def _iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ub = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = ua + ub - inter
    return float(inter / union) if union > 0 else 0.0


class PersonFilter:
    """Temporal confirmation + smoothing for the reported person, so the
    distance readout / 3D box never hallucinate or flicker.

    A candidate must be detected for `confirm` consecutive detection cycles
    before the person is reported `present`, and missed for `lose` cycles
    before it's dropped. Distance / azimuth / 3D position are EMA-smoothed so
    the number is steady. `xyxy` is the latest confirmed box (for masking)."""

    def __init__(self, confirm: int = 2, lose: int = 4, alpha: float = 0.35):
        self.confirm = int(confirm)
        self.lose = int(lose)
        self.alpha = float(alpha)
        self._hits = 0
        self._misses = 0
        self._present = False
        self._dist = 0.0
        self._az = 0.0
        self._conf = 0.0
        self._pos = None
        self.xyxy = None

    def update(self, cand) -> Dict:
        """cand: {distance_m, azimuth_deg, conf, xyxy, position_3d} or None."""
        a = self.alpha
        if cand is not None:
            self._hits += 1
            self._misses = 0
            self.xyxy = cand.get("xyxy")
            p = cand.get("position_3d")
            if self._present:
                self._dist = a * cand["distance_m"] + (1 - a) * self._dist
                self._az = a * cand["azimuth_deg"] + (1 - a) * self._az
                self._conf = a * cand["conf"] + (1 - a) * self._conf
                if p and self._pos:
                    self._pos = [a * p[k] + (1 - a) * self._pos[k] for k in range(3)]
                elif p:
                    self._pos = list(p)
            else:
                self._dist, self._az = cand["distance_m"], cand["azimuth_deg"]
                self._conf, self._pos = cand["conf"], (list(p) if p else None)
                if self._hits >= self.confirm:
                    self._present = True
        else:
            self._misses += 1
            self._hits = 0
            if self._misses >= self.lose:
                self._present = False
                self.xyxy = None
        return self.state()

    def state(self) -> Dict:
        if not self._present:
            return {"present": False, "distance_m": 0.0, "azimuth_deg": 0.0,
                    "conf": 0.0, "position_3d": None}
        return {"present": True, "distance_m": round(self._dist, 2),
                "azimuth_deg": round(self._az, 1), "conf": round(self._conf, 2),
                "position_3d": list(self._pos) if self._pos else None}

    def reset(self):
        self.__init__(self.confirm, self.lose, self.alpha)
