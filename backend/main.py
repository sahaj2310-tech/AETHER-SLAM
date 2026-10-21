"""FastAPI entrypoint + asynchronous threaded perception pipeline.

Stages (each a daemon thread, connected by capacity-one overwrite buffers):

    Capture ─►(frame_q)─► SLAM ─►(slam_q)─► Semantic + alignment ─►(tsdf_q)─► Fusion
                                                                                 │
                                                       mesh extraction thread ◄──┘
                                       state estimator ──► WebSocket broadcasts
                                                           (state / map / video / mesh)

SLAM      = DISK + LightGlue tracking, PnP-RANSAC + LM, parallax triangulation of
            the alignment anchors, keyframes, loop detection, windowed bundle adjustment.
Semantic  = YOLOv8n-seg person boxes + DPT-Hybrid relative inverse depth, aligned to
            the map by confidence-gated scale-shift filtering (CG-SSF).
Fusion    = uncertainty-weighted TSDF on the GPU; marching cubes on its own thread.

Capture is a webcam or a recorded sequence replayed in real time (AETHER_SOURCE).
Every GPU stage runs on its own CUDA stream. /api/perf reports measured
throughput, stage service times and the capture->fusion age of every frame.
"""

import base64
import logging
import os
import threading
import time
from contextlib import asynccontextmanager, nullcontext
from typing import Optional

import asyncio
import cv2
import numpy as np
import psutil
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# torch.load monkey-patch (ultralytics/MiDaS-friendly)
try:
    import torch
    _orig_load = torch.load

    def _patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_load(*args, **kwargs)
    torch.load = _patched_load
except ImportError:
    torch = None

from backend.config import Config
from backend.slam.tracker import SLAMTracker
from backend.slam.mapper import SLAMMapper
from backend.slam.optimization import run_local_bundle_adjustment, run_local_bundle_adjustment_lm
from backend.slam.loop_closure import LoopClosureDetector
from backend.semantic.detector import YOLODetector, PersonFilter
from backend.semantic.depth import DepthEstimator
from backend.semantic.dense_fusion import DenseFusion
from backend.semantic.alignment import make_aligner
from backend.semantic.semantic_map import SemanticMap
from backend.semantic import query as query_engine
from backend.audio.audio_detector import AudioDetector
from backend.fusion.state_estimator import SensorFusionEstimator
from backend.pipeline.orchestrator import LatestQueue, Stage
from backend.pipeline.synchronous import align_config_from
from backend.perf import PerfMonitor
from backend.sources import RecordedSequence
from backend import persistence

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("Backend")

# --- Global config + state ------------------------------------------------
config = Config()
recorded_source: Optional[RecordedSequence] = None
if config.CAPTURE_SOURCE and config.CAPTURE_SOURCE != "webcam":
    try:
        recorded_source = RecordedSequence.open(config.CAPTURE_SOURCE, config.IMAGE_WIDTH)
        config.set_camera(recorded_source.K, *recorded_source.size)
        logger.info(f"Capture source: recorded {recorded_source.kind} '{recorded_source.name}' "
                    f"({len(recorded_source)} frames, {recorded_source.size[0]}x{recorded_source.size[1]}).")
    except Exception as e:
        logger.error(f"Cannot open capture source {config.CAPTURE_SOURCE!r}: {e}. Falling back to webcam.")
        recorded_source = None

fusion_estimator = SensorFusionEstimator(config)
perf = PerfMonitor()
connection_manager: list[WebSocket] = []

# Inter-stage queues (always size 1, drop-oldest)
frame_q = LatestQueue(maxsize=1)          # Capture -> SLAM
slam_result_q = LatestQueue(maxsize=1)    # SLAM -> Semantic
tsdf_input_q = LatestQueue(maxsize=1)     # Semantic -> Dense fusion thread

latest_video_frame: Optional[str] = None
video_frame_lock = threading.Lock()

# Reset coordination: a monotonically-increasing generation counter observed
# independently by every stage (no flag-clearing races).
_reset_lock = threading.Lock()
reset_generation = 0


def request_reset():
    global reset_generation
    with _reset_lock:
        reset_generation += 1
    logger.info("Reset requested (generation %d).", reset_generation)


shared_semantic_map: Optional[SemanticMap] = None
# Latest aligned depth (numpy, 0 = invalid) for creating new map points, and the
# gauge depth used if the SLAM stage has to (re)initialise the map.
shared_spawn_depth = None
shared_gauge_depth = None
shared_dense_fusion: Optional[DenseFusion] = None
shared_mapper = None
# Dynamic-object (person) boxes published by the semantic stage.
shared_dynamic_bboxes: list = []
_dynamic_bboxes_lock = threading.Lock()
# Serialises *model construction* across stages (concurrent CUDA init can crash).
_gpu_init_lock = threading.Lock()

# --- CUDA tuning ---------------------------------------------------------
if torch is not None and torch.cuda.is_available():
    if config.CUDNN_BENCHMARK:
        torch.backends.cudnn.benchmark = True
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass
    if config.GPU_MEMORY_CAP_GB:
        total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        frac = min(1.0, config.GPU_MEMORY_CAP_GB / total)
        torch.cuda.set_per_process_memory_fraction(frac, 0)
        logger.info(f"GPU memory capped at {config.GPU_MEMORY_CAP_GB:.1f} GB of {total:.1f} GB.")
    logger.info("cudnn.benchmark + TF32 enabled.")


def _stream_ctx():
    """Each GPU stage issues its kernels on its own CUDA stream."""
    if torch is not None and torch.cuda.is_available():
        return torch.cuda.stream(torch.cuda.Stream())
    return nullcontext()


def _stream_sync():
    if torch is not None and torch.cuda.is_available():
        torch.cuda.current_stream().synchronize()


# =========================================================================
# Stages
# =========================================================================

def _capture_stage(stop_event, stage):
    """Webcam, or a recorded sequence replayed at its own frame timing."""
    if recorded_source is not None:
        return _replay_capture(stop_event, stage)
    cap = None
    consecutive_failures = 0
    local_reset_gen = reset_generation
    index = 0
    while not stop_event.is_set():
        t0 = time.time()
        if reset_generation != local_reset_gen:
            local_reset_gen = reset_generation
            frame_q.get_latest(timeout=0.0); slam_result_q.get_latest(timeout=0.0); tsdf_input_q.get_latest(timeout=0.0)
        if cap is None:
            logger.info(f"Opening webcam (idx={config.CAMERA_INDEX}, CAP_DSHOW)...")
            cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(config.CAMERA_INDEX)
            if not cap.isOpened():
                logger.error(f"No webcam found at index {config.CAMERA_INDEX}.")
                fusion_estimator.set_tracking_status("NO_CAMERA")
                cap = None
                time.sleep(1.0)
                continue
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.IMAGE_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.IMAGE_HEIGHT)
                cap.set(cv2.CAP_PROP_FPS, config.CAPTURE_FPS)
            except Exception:
                pass
        ret, frame = cap.read()
        if not ret or frame is None:
            consecutive_failures += 1
            if consecutive_failures >= 5:
                logger.error("Webcam read failed repeatedly. Re-opening...")
                cap.release()
                cap = None
                fusion_estimator.set_tracking_status("NO_CAMERA")
            time.sleep(0.05)
            continue
        consecutive_failures = 0
        frame = cv2.resize(frame, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT))
        now = time.perf_counter()
        frame_q.put_latest({"frame": frame, "captured_at": now, "t": time.time(), "index": index})
        index += 1
        perf.record("capture", 1e3 * (time.time() - t0))
        stage.record_latency(time.time() - t0)
        target_dt = 1.0 / config.CAPTURE_FPS
        elapsed = time.time() - t0
        if elapsed < target_dt:
            time.sleep(target_dt - elapsed)
    if cap is not None:
        cap.release()


def _replay_capture(stop_event, stage):
    src = recorded_source
    n = len(src)
    ts = np.asarray(src.timestamps, np.float64)
    loop = 0
    while not stop_event.is_set():
        t_wall0 = time.perf_counter()
        for i in range(n):
            if stop_event.is_set():
                return
            due = t_wall0 + (ts[i] - ts[0])
            delay = due - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            t0 = time.perf_counter()
            frame = src.frame(i)
            now = time.perf_counter()
            frame_q.put_latest({"frame": frame, "captured_at": now, "t": float(ts[i] + loop * (ts[-1] - ts[0] + 1.0)),
                                "index": i})
            perf.record("capture", 1e3 * (now - t0))
            stage.record_latency(now - t0)
        if not config.CAPTURE_LOOP:
            logger.info("Recorded sequence finished.")
            return
        loop += 1
        request_reset()        # a new pass starts a new map


def _slam_stage(stop_event, stage):
    global shared_mapper
    with _gpu_init_lock:
        tracker = SLAMTracker(config)
    mapper = SLAMMapper(config)
    shared_mapper = mapper
    loop_detector = LoopClosureDetector(tracker.backend, min_kf_gap=config.LOOP_MIN_KF_GAP,
                                        min_score=config.LOOP_MIN_SCORE, min_matches=config.LOOP_MIN_MATCHES)
    lba_counter = 0
    local_reset_gen = reset_generation
    fusion_estimator.set_feature_backend(tracker.backend.name)
    with _stream_ctx():
        while not stop_event.is_set():
            if reset_generation != local_reset_gen:
                local_reset_gen = reset_generation
                tracker = SLAMTracker(config, feature_backend=tracker.backend)
                mapper = SLAMMapper(config)
                shared_mapper = mapper
                loop_detector = LoopClosureDetector(tracker.backend, min_kf_gap=config.LOOP_MIN_KF_GAP,
                                                    min_score=config.LOOP_MIN_SCORE,
                                                    min_matches=config.LOOP_MIN_MATCHES)
                lba_counter = 0
                fusion_estimator.set_tracking_status("RESET")

            item = frame_q.get_latest(timeout=0.05)
            if item is None:
                continue
            t0 = time.perf_counter()
            frame = item["frame"]
            with _dynamic_bboxes_lock:
                dyn_snapshot = list(shared_dynamic_bboxes)
            success, T_WC, _ = tracker.track(frame, depth_map=shared_spawn_depth, dynamic_bboxes=dyn_snapshot,
                                             init_depth=shared_gauge_depth)
            anchors_px, anchors_z = tracker.anchors_camera() if success else (None, None)
            status = "TRACKING" if success else ("LOST" if tracker.initialized else "INIT")

            if success and mapper.is_keyframe_needed(tracker.T_WC, item["t"]):
                mapper.add_keyframe(tracker.T_WC, tracker.tracked_pts_2d, tracker.tracked_pts_3d,
                                    tracker.tracked_pt_ids, frame)
                try:
                    ev = loop_detector.check(mapper.keyframes[-1].id, tracker.last_features, t=item["t"])
                    if ev is not None:
                        mapper.record_loop_event(ev)
                    loop_detector.add_keyframe(mapper.keyframes[-1].id, tracker.last_features)
                except Exception as e:
                    logger.debug(f"Loop check skipped: {e}")
                lba_counter += 1
                if config.LBA_ENABLED and lba_counter % max(1, config.LBA_EVERY_KF) == 0:
                    try:
                        if config.LBA_METHOD == "lm":
                            run_local_bundle_adjustment_lm(mapper, tracker, window_size=config.LBA_WINDOW_SIZE,
                                                           max_nfev=config.LBA_MAX_NFEV, huber_px=config.LBA_HUBER_PX)
                        else:
                            run_local_bundle_adjustment(mapper, tracker, window_size=config.LBA_WINDOW_SIZE)
                    except Exception as e:
                        logger.error(f"BA error: {e}")

            eff_T = tracker.T_WC.copy()
            fusion_estimator.update_pose_and_map(eff_T, mapper)
            fusion_estimator.set_tracking_status(status)
            _stream_sync()
            slam_result_q.put_latest({
                "T_WC": eff_T, "success": success, "frame": frame,
                "anchors_px": anchors_px, "anchors_z": anchors_z,
                "tracked_pts_2d": tracker.tracked_pts_2d.copy(),
                "map_gen": tracker.map_generation, "captured_at": item["captured_at"], "t": item["t"],
                "need_init": (not tracker.initialized) or tracker._consecutive_failures >= tracker._failure_tolerance - 1,
            })
            dt = time.perf_counter() - t0
            perf.record("slam", 1e3 * dt)
            stage.record_latency(dt)
            fusion_estimator.update_telemetry("slam_ms", round(stage.latency_ms, 1))
            fusion_estimator.update_telemetry("anchors", int(tracker.anchor_px.shape[0]))
            fusion_estimator.update_telemetry("inliers", int(tracker.n_inliers))


def _box_mask(shape, boxes, device):
    valid = torch.ones(shape, dtype=torch.bool, device=device)
    h, w = shape
    for (x1, y1, x2, y2) in boxes:
        x1 = max(0, int(x1)); y1 = max(0, int(y1)); x2 = min(w - 1, int(x2)); y2 = min(h - 1, int(y2))
        if x2 > x1 and y2 > y1:
            valid[y1:y2 + 1, x1:x2 + 1] = False
    return valid


def _semantic_stage(stop_event, stage):
    global shared_semantic_map, shared_spawn_depth, shared_gauge_depth
    detect_enabled = bool(getattr(config, "SEMANTIC_DETECTION_ENABLED", False))
    person_enabled = bool(getattr(config, "PERSON_TRACKING_ENABLED", True)) and not detect_enabled
    detector = None
    with _gpu_init_lock:
        if detect_enabled:
            detector = YOLODetector(config)
        elif person_enabled:
            detector = YOLODetector(config, person_only=True)
        depth_estimator = DepthEstimator(config)
    aligner = make_aligner(config.ALIGNMENT_VARIANT, align_config_from(config))
    mask_persons = config.ALIGNMENT_VARIANT != "cgssf_nomask"
    semantic_map = SemanticMap(config)
    if not detect_enabled:
        semantic_map.reset()
    shared_semantic_map = semantic_map
    pad = int(getattr(config, "DYNAMIC_MASK_PADDING_PX", 8))
    person_every = max(1, int(getattr(config, "PERSON_DETECT_EVERY", 1)))
    person_filter = PersonFilter(confirm=int(getattr(config, "PERSON_CONFIRM_CYCLES", 2)),
                                 lose=int(getattr(config, "PERSON_LOSE_CYCLES", 4)),
                                 alpha=float(getattr(config, "PERSON_SMOOTH_ALPHA", 0.35)))
    local_reset_gen = reset_generation
    last_map_gen = None
    last_prune = time.time()
    sem_cycle = 0
    boxes = []
    last_nu, last_nu_pose = None, None

    with _stream_ctx():
        while not stop_event.is_set():
            if reset_generation != local_reset_gen:
                local_reset_gen = reset_generation
                semantic_map.reset(); person_filter.reset(); aligner.reset()
                shared_spawn_depth = None; shared_gauge_depth = None
                last_map_gen = None; last_nu = None

            item = slam_result_q.get_latest(timeout=0.05)
            if item is None:
                continue
            t0 = time.perf_counter()
            sem_cycle += 1
            frame = item["frame"]
            T_WC = item["T_WC"]
            map_gen = item.get("map_gen")
            if last_map_gen is not None and map_gen != last_map_gen:
                aligner.reset(); person_filter.reset()
            last_map_gen = map_gen

            # --- person boxes (cadence): masking + person read-out ---
            detections = []
            if detector is not None and sem_cycle % person_every == 0:
                if detect_enabled:
                    detections = detector.detect(frame, shared_spawn_depth if shared_spawn_depth is not None
                                                 else np.ones(frame.shape[:2], np.float32), T_WC)
                    dets = [d for d in detections if d["label"] in getattr(config, "DYNAMIC_OBJECT_CLASSES", {"person"})]
                    boxes = [(d["xyxy"][0] - pad, d["xyxy"][1] - pad, d["xyxy"][2] + pad, d["xyxy"][3] + pad) for d in dets]
                    semantic_map.update_map(detections)
                    fusion_estimator.update_semantics(semantic_map)
                    if (time.time() - last_prune) > 5.0:
                        semantic_map.prune(); last_prune = time.time()
                else:
                    dets = detector.detect_person_boxes(frame)
                    boxes = [(d["xyxy"][0] - pad, d["xyxy"][1] - pad, d["xyxy"][2] + pad, d["xyxy"][3] + pad) for d in dets]
            if not mask_persons:
                boxes_mask = []
            else:
                boxes_mask = boxes
            with _dynamic_bboxes_lock:
                shared_dynamic_bboxes[:] = boxes_mask

            # --- depth network (optionally skipped while static) ---
            reuse = False
            if config.DEPTH_MOTION_SKIP and last_nu is not None and last_nu_pose is not None:
                dt_ = float(np.linalg.norm(T_WC[:3, 3] - last_nu_pose[:3, 3]))
                Rr = last_nu_pose[:3, :3].T @ T_WC[:3, :3]
                dr = float(np.arccos(np.clip((np.trace(Rr) - 1) / 2, -1, 1)))
                reuse = dt_ < config.DEPTH_SKIP_TRANS and dr < config.DEPTH_SKIP_ROT_RAD
            nu = last_nu if reuse else depth_estimator.predict(frame)
            if nu is not None and not reuse:
                last_nu, last_nu_pose = nu, T_WC.copy()

            res = None
            if nu is not None:
                valid = _box_mask(nu.shape, boxes_mask, nu.device)
                if item.get("need_init") or shared_gauge_depth is None:
                    shared_gauge_depth = depth_estimator.gauge_depth(nu, valid)
                if item.get("success") and item.get("anchors_px") is not None:
                    px, zc = item["anchors_px"], item["anchors_z"]
                    if boxes_mask and px.shape[0]:
                        keep = np.ones(px.shape[0], bool)
                        for (x1, y1, x2, y2) in boxes_mask:
                            keep &= ~((px[:, 0] >= x1) & (px[:, 0] <= x2) & (px[:, 1] >= y1) & (px[:, 1] <= y2))
                        px, zc = px[keep], zc[keep]
                    res = aligner.step(nu, valid, px, zc.astype(np.float64), item["t"], T_WC=T_WC)
                    if res.z_spawn is not None:
                        shared_spawn_depth = res.z_spawn.cpu().numpy()
                    th = aligner.theta
                    fusion_estimator.update_telemetry("align_status", res.status)
                    fusion_estimator.update_telemetry("align_scale", None if th is None else round(th[0], 4))
                    fusion_estimator.update_telemetry("align_shift", None if th is None else round(th[1], 4))

            # --- person read-out (closest confirmed person, distance from aligned depth) ---
            if person_enabled and detector is not None:
                if sem_cycle % person_every == 0:
                    cand = None
                    zmap = res.z_eval.cpu().numpy() if (res is not None and res.z_eval is not None) else None
                    if boxes and zmap is not None:
                        best = None
                        for (x1, y1, x2, y2) in boxes:
                            x1c, y1c = max(0, int(x1 + pad)), max(0, int(y1 + pad))
                            x2c, y2c = int(x2 - pad), int(y2 - pad)
                            roi = zmap[y1c:y2c + 1, x1c:x2c + 1]
                            vals = roi[(roi > 0.1) & (roi < 10.0)]
                            if vals.size >= 8:
                                d = float(np.median(vals))
                                if best is None or d < best[0]:
                                    best = (d, (x1c, y1c, x2c, y2c))
                        if best is not None:
                            pz, (px1, py1, px2, py2) = best
                            uc, vc = 0.5 * (px1 + px2), 0.5 * (py1 + py2)
                            cam_pt = np.array([(uc - config.CX) * pz / config.FX, (vc - config.CY) * pz / config.FY, pz, 1.0])
                            pos_w = (T_WC @ cam_pt)[:3]
                            cand = {"distance_m": round(pz, 2), "azimuth_deg": float(np.degrees(np.arctan2(uc - config.CX, config.FX))),
                                    "conf": 1.0, "xyxy": [px1, py1, px2, py2], "position_3d": [float(v) for v in pos_w]}
                    person_state = person_filter.update(cand)
                else:
                    person_state = person_filter.state()
                fusion_estimator.update_person(person_state)
                if person_state["present"] and person_filter.xyxy is not None:
                    detections = [{"xyxy": list(person_filter.xyxy), "label": f"person {person_state['distance_m']:.1f}"}]

            # --- hand the aligned, weighted depth to fusion (only while tracking) ---
            if (res is not None and res.z_fuse is not None and item.get("success")
                    and fusion_estimator.tracking_status == "TRACKING"):
                _stream_sync()
                tsdf_input_q.put_latest({"frame": frame, "depth": res.z_fuse, "weight": res.w_fuse, "T_WC": T_WC,
                                         "map_gen": map_gen, "captured_at": item["captured_at"]})

            annotated = _annotate(frame, item.get("tracked_pts_2d"), detections)
            _publish_video(annotated)
            dt = time.perf_counter() - t0
            perf.record("semantic", 1e3 * dt)
            stage.record_latency(dt)
            fusion_estimator.update_telemetry("semantic_ms", round(stage.latency_ms, 1))


def _dense_fusion_stage(stop_event, stage):
    """Motion-gated, uncertainty-weighted TSDF integration on the GPU."""
    global shared_dense_fusion
    if not config.DENSE_FUSION_ENABLED:
        return
    dense_fusion = DenseFusion(config)
    shared_dense_fusion = dense_fusion
    if not dense_fusion.enabled:
        logger.warning("Open3D/torch unavailable; dense fusion stage idle.")
        return
    local_reset_gen = reset_generation
    last_map_gen = None
    last_integ_pose = None
    with _stream_ctx():
        while not stop_event.is_set():
            if reset_generation != local_reset_gen:
                local_reset_gen = reset_generation
                dense_fusion.reset(); last_map_gen = None; last_integ_pose = None
            item = tsdf_input_q.get_latest(timeout=0.1)
            if item is None:
                continue
            t0 = time.perf_counter()
            map_gen = item.get("map_gen")
            if last_map_gen is not None and map_gen != last_map_gen:
                dense_fusion.reset(); last_integ_pose = None
            last_map_gen = map_gen
            T_WC = item["T_WC"]
            moved = last_integ_pose is None
            if not moved:
                dt = float(np.linalg.norm(T_WC[:3, 3] - last_integ_pose[:3, 3]))
                R = last_integ_pose[:3, :3].T @ T_WC[:3, :3]
                dr = float(np.arccos(float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))
                moved = dt > config.TSDF_MIN_MOVE_TRANS_M or dr > config.TSDF_MIN_MOVE_ROT_RAD
            if moved and dense_fusion.integrate(item["frame"], item["depth"], item["weight"], T_WC):
                last_integ_pose = T_WC.copy()
            _stream_sync()
            done = time.perf_counter()
            perf.record("fusion", 1e3 * (done - t0))
            perf.record_age(1e3 * (done - item["captured_at"]))
            stage.record_latency(done - t0)
            fusion_estimator.update_telemetry("tsdf_ms", round(stage.latency_ms, 1))


def _mesh_stage(stop_event, stage):
    """Marching cubes on its own thread so it never blocks fusion or tracking."""
    while not stop_event.is_set():
        time.sleep(max(0.2, config.TSDF_MESH_COOLDOWN_S))
        df = shared_dense_fusion
        if df is None or not df.enabled:
            continue
        t0 = time.perf_counter()
        snap = df.extract_mesh(cooldown_s=config.TSDF_MESH_COOLDOWN_S)
        if snap is not None:
            perf.record("mesh", 1e3 * (time.perf_counter() - t0))
            stage.record_latency(time.perf_counter() - t0)


def _audio_stage(stop_event, stage):
    """Microphone sound / voice-activity detection (self-healing)."""
    if not getattr(config, "AUDIO_DETECTION_ENABLED", True):
        return
    detector = AudioDetector(config)
    while not stop_event.is_set():
        t0 = time.time()
        detector.ensure_open()
        fusion_estimator.update_audio(detector.event())
        stage.record_latency(time.time() - t0)
        time.sleep(0.08)
    detector.close()


def _telemetry_stage(stop_event, stage):
    """Measured throughput + RAM/VRAM sampler. Not on the perception hot path."""
    _hb = 0
    while not stop_event.is_set():
        time.sleep(0.5)
        _hb += 1
        fps = perf.fps("semantic")
        fusion_estimator.update_telemetry("fps", round(fps, 1))
        fusion_estimator.update_telemetry("slam_fps", round(perf.fps("slam"), 1))
        fusion_estimator.update_telemetry("ram_pct", round(psutil.virtual_memory().percent, 1))
        if torch is not None and torch.cuda.is_available():
            fusion_estimator.update_telemetry("vram_mb", round(torch.cuda.memory_allocated() / (1024 * 1024), 1))
        if _hb % 6 == 0:
            logger.info("perf: fps=%.1f slam_fps=%.1f slam=%.0fms semantic=%.0fms clients=%d",
                        fps, perf.fps("slam"), fusion_estimator.telemetry.get("slam_ms", 0.0),
                        fusion_estimator.telemetry.get("semantic_ms", 0.0), len(connection_manager))


# =========================================================================
# Annotation + video publish
# =========================================================================

def _annotate(frame, tracked_pts_2d, detections):
    out = frame.copy()
    if tracked_pts_2d is not None and len(tracked_pts_2d) > 0:
        for pt in tracked_pts_2d:
            cv2.circle(out, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)
    for det in detections:
        x1, y1, x2, y2 = det["xyxy"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 96, 32), 2)
        tid = det.get("track_id")
        label = det["label"] if tid in (None, "?") else f"{det['label']} #{tid}"
        cv2.putText(out, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 96, 32), 1)
    return out


def _publish_video(annotated_frame):
    global latest_video_frame
    ok, buf = cv2.imencode(".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), config.VIDEO_JPEG_QUALITY])
    if not ok:
        return
    with video_frame_lock:
        latest_video_frame = base64.b64encode(buf).decode("utf-8")


# =========================================================================
# FastAPI app + lifespan
# =========================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    stages = [
        Stage("capture", _capture_stage),
        Stage("slam", _slam_stage),
        Stage("semantic", _semantic_stage),
        Stage("fusion", _dense_fusion_stage),
        Stage("mesh", _mesh_stage),
        Stage("audio", _audio_stage),
        Stage("telemetry", _telemetry_stage),
    ]
    for s in stages:
        s.start()
    tasks = [asyncio.create_task(c()) for c in (_broadcast_state_loop, _broadcast_map_loop,
                                                _broadcast_video_loop, _broadcast_mesh_loop)]
    logger.info("Pipeline stages started.")
    try:
        yield
    finally:
        for t in tasks:
            t.cancel()
        for s in stages:
            s.stop()
        try:
            if shared_semantic_map is not None:
                shared_semantic_map.save_persistent()
            persistence.save_session(config, semantic_map=shared_semantic_map, mapper=shared_mapper,
                                     dense_fusion=shared_dense_fusion)
        except Exception as e:
            logger.warning(f"shutdown save failed: {e}")
        logger.info("Pipeline stopped.")


app = FastAPI(title="AETHER-SLAM API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                   allow_methods=["*"], allow_headers=["*"])


class ConfigUpdate(BaseModel):
    confidence_threshold: Optional[float] = None


class SpatialQuery(BaseModel):
    text: str


@app.get("/api/status")
def get_status():
    return {
        "status": "online",
        "cuda_available": torch is not None and torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if (torch is not None and torch.cuda.is_available()) else None,
        "feature_backend": fusion_estimator.telemetry.get("feature_backend", "?"),
        "capture_source": ("webcam" if recorded_source is None else f"{recorded_source.kind}:{recorded_source.name}"),
        "alignment": config.ALIGNMENT_VARIANT,
        "depth_model": config.DEPTH_MODEL_TYPE,
        "dense_fusion": (shared_dense_fusion.stats() if shared_dense_fusion is not None else {"enabled": False}),
    }


@app.get("/api/perf")
def get_perf():
    """Measured runtime: stage service times, throughput, capture->fusion age, GPU memory."""
    return perf.snapshot()


@app.post("/api/perf/reset")
def reset_perf():
    perf.reset()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    return {"status": "ok"}


@app.post("/api/config")
def update_config(data: ConfigUpdate):
    if data.confidence_threshold is not None:
        config.CONFIDENCE_THRESHOLD = data.confidence_threshold
    return {"status": "success", "confidence_threshold": config.CONFIDENCE_THRESHOLD}


@app.get("/api/export/mesh")
def export_mesh():
    """Export the current reconstruction (full resolution) as a PLY mesh."""
    if shared_dense_fusion is None or not shared_dense_fusion.enabled:
        return {"status": "error", "message": "dense fusion not running"}
    try:
        import open3d as o3d
        mesh = shared_dense_fusion.full_mesh()
        if mesh is None or len(mesh.vertices) == 0:
            return {"status": "empty", "message": "No mesh yet - move the camera slowly to scan first."}
        out_dir = os.path.join(config.DATA_DIR, "export")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "room_mesh.ply")
        o3d.io.write_triangle_mesh(path, mesh)
        return FileResponse(path, media_type="application/octet-stream", filename="room_mesh.ply",
                            headers={"Content-Disposition": "attachment; filename=room_mesh.ply"})
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/save")
def api_save():
    path = persistence.save_session(config, semantic_map=shared_semantic_map, mapper=shared_mapper,
                                    dense_fusion=shared_dense_fusion)
    if path is None:
        return {"status": "error"}
    return {"status": "success", "path": path,
            "objects": len(shared_semantic_map.get_semantic_objects()) if shared_semantic_map else 0,
            "keyframes": len(shared_mapper.keyframes) if shared_mapper else 0}


@app.get("/api/load")
def api_load():
    data = persistence.load_session(config)
    if data is None:
        return {"status": "no_session"}
    return {"status": "ok", **data}


@app.post("/api/query")
def spatial_query(q: SpatialQuery):
    if shared_semantic_map is None:
        return {"intent": "not_ready", "summary": "Scene map not initialised yet.",
                "objects": [], "relations": [], "confidence": 0.0}
    return query_engine.answer(q.text, shared_semantic_map).to_dict()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connection_manager.append(websocket)
    logger.info(f"Client connected. Total: {len(connection_manager)}")
    try:
        while True:
            data = await websocket.receive_text()
            if data == "reset":
                request_reset()
    except WebSocketDisconnect:
        if websocket in connection_manager:
            connection_manager.remove(websocket)


async def _send_to_all(payload):
    dead = []
    for ws in connection_manager:
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in connection_manager:
            connection_manager.remove(ws)


async def _broadcast_state_loop():
    period = 1.0 / config.STATE_BROADCAST_HZ
    while True:
        if connection_manager:
            state = fusion_estimator.get_consolidated_state(include_map=False)
            state["msg"] = "state"
            await _send_to_all(state)
        await asyncio.sleep(period)


async def _broadcast_map_loop():
    period = 1.0 / max(0.5, getattr(config, "MAP_BROADCAST_HZ", 4))
    while True:
        if connection_manager:
            payload = fusion_estimator.get_map_state()
            payload["msg"] = "map"
            await _send_to_all(payload)
        await asyncio.sleep(period)


async def _broadcast_video_loop():
    period = 1.0 / config.VIDEO_BROADCAST_HZ
    while True:
        if connection_manager:
            with video_frame_lock:
                frame_b64 = latest_video_frame
            if frame_b64 is not None:
                await _send_to_all({"msg": "video", "video_frame": frame_b64})
        await asyncio.sleep(period)


async def _broadcast_mesh_loop():
    period = 1.0 / max(0.05, config.MESH_BROADCAST_HZ)
    last_extracted_at = 0.0
    while True:
        if connection_manager and shared_dense_fusion is not None:
            snap = shared_dense_fusion.latest_mesh()
            if snap.n_vertices > 0 and snap.extracted_at > last_extracted_at:
                last_extracted_at = snap.extracted_at
                payload = snap.to_dict()
                payload["msg"] = "mesh"
                await _send_to_all(payload)
        await asyncio.sleep(period)


# --- Static frontend ----------------------------------------------------
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.exists(static_dir):
    logger.info(f"Mounting static: {static_dir}")

    @app.get("/")
    def serve_index():
        return FileResponse(os.path.join(static_dir, "index.html"),
                            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"})

    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


if __name__ == "__main__":
    # Run in THIS process via Server.run() (uvicorn.run's supervisor would spawn
    # a second process with a second copy of every model on the GPU).
    server = uvicorn.Server(uvicorn.Config(app, host=config.HOST, port=config.PORT, log_level="info"))
    server.run()
