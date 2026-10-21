import os
import numpy as np


class Config:
    # --- Server Configuration ---
    HOST = "0.0.0.0"
    PORT = int(os.environ.get("AETHER_PORT", "8000"))
    DEBUG = False

    # --- Capture ---
    # CAPTURE_SOURCE: "webcam" (CAMERA_INDEX), or a recorded sequence replayed
    # in real time as a virtual camera: a TUM RGB-D folder (rgb.txt), a Replica
    # (NICE-SLAM render) scene folder, or a video file. Override with the
    # environment variable AETHER_SOURCE. Never synthetic frames.
    CAPTURE_SOURCE = os.environ.get("AETHER_SOURCE", "webcam")
    CAPTURE_LOOP = True          # restart a recorded sequence when it ends
    CAPTURE_FPS = 30
    CAMERA_INDEX = 0

    # --- Camera (pinhole) ---
    IMAGE_WIDTH = 640
    IMAGE_HEIGHT = 480
    FX = 525.0
    FY = 525.0
    CX = 320.0
    CY = 240.0

    def __init__(self):
        self._update_intrinsics_cache()

    def _update_intrinsics_cache(self):
        self._intrinsics_cache = np.array([
            [self.FX, 0.0,     self.CX],
            [0.0,     self.FY, self.CY],
            [0.0,     0.0,     1.0]
        ], dtype=np.float32)

    def set_camera(self, K, width: int, height: int):
        """Use the intrinsics of a recorded source (after undistortion / resizing)."""
        self.FX, self.FY = float(K[0][0]), float(K[1][1])
        self.CX, self.CY = float(K[0][2]), float(K[1][2])
        self.IMAGE_WIDTH, self.IMAGE_HEIGHT = int(width), int(height)
        self._update_intrinsics_cache()

    @property
    def INTRINSICS_MATRIX(self):
        return self._intrinsics_cache

    # --- SLAM ---
    DETECT_FEATURES = 1024              # DISK keypoints detected per frame (matching pool)
    MAX_FEATURES = 256                  # tracked map points (matched against the whole
                                        # pool, so LightGlue runs 256 x 1024)
    MAX_LOST_FRAMES = 3                 # a map point missed in a frame is kept (at its
                                        # predicted pixel) for re-matching for this many frames
    KEYFRAME_MIN_TRANSLATION = 0.15     # raised: ignores small pose jitter
    KEYFRAME_MIN_ROTATION = 10.0        # kept: useful while panning the room
    KEYFRAME_MIN_DT = 0.25              # seconds; rate-limit keyframes
    MIN_TRACKED_POINTS = 15
    LBA_WINDOW_SIZE = 10
    LBA_ENABLED = False                 # windowed bundle adjustment (off: no measurable gain
                                        # on TUM fr1, see results/design_tests)
    LBA_METHOD = "lm"                   # "lm": sparse trust-region (Levenberg-Marquardt
                                        # type) solver; "adam": legacy first-order pass
    LBA_EVERY_KF = 1                    # run BA every N-th keyframe
    LBA_HUBER_PX = 2.0
    LBA_MAX_NFEV = 15
    # Map-point triangulation: a point created from the aligned depth keeps its
    # first (reference) observation and is re-estimated by two-view midpoint
    # triangulation once the parallax reaches TRI_MIN_PARALLAX_DEG (accepted if
    # in front of both cameras and within TRI_MAX_REPROJ_PX in both views).
    TRI_MIN_PARALLAX_DEG = 2.0
    TRI_MAX_REPROJ_PX = 2.0
    TRI_RULE = "latest"                 # re-triangulate with the latest view ("grow": only if parallax grew)
    TRI_UPDATE_POINTS = False           # True: triangulation also moves the map point (causes
                                        # scale drift); default: triangulated depths are anchors only
    # Anchors of depth alignment: "triangulated" PnP inliers (default) or "all"
    # PnP inliers (including points that only carry back-projected depth).
    ANCHOR_MODE = "triangulated"
    # Anchor-restriction test (evaluation only; 0 = off): in every second block of ANCHOR_BAND_BLOCK
    # frames, only the anchors whose network output lies within +-ANCHOR_BAND of the median over the
    # frame's anchors reach the alignment, which makes the frame only partially observable.
    ANCHOR_BAND = 0.0
    ANCHOR_BAND_BLOCK = 50
    MAX_KEYFRAMES = 600                 # cap keyframe history (RAM + trajectory)
    VOXEL_SIZE = 0.05
    # Monocular bootstrap: when no dense depth is available yet, assume
    # a far depth at the top of the frame (walls/ceiling) and a near
    # depth at the bottom (floor). This breaks the coplanar-init PnP
    # degeneracy that otherwise sends the tracker into a re-init loop.
    INIT_DEPTH_TOP_M = 4.0
    INIT_DEPTH_BOTTOM_M = 1.0
    # How many consecutive PnP failures before we destroy the map. Higher =
    # the map (and the dense reconstruction built on it) survives brief
    # tracking dropouts instead of resetting and smearing the mesh.
    TRACK_FAILURE_TOLERANCE = 12
    # Pose smoothing: low-pass the broadcast pose so the 3D camera
    # frustum and the trajectory ribbon don't jitter under PnP noise.
    # SLAM internally still uses the raw pose; this only affects what's
    # shipped to the frontend.
    POSE_SMOOTH_ALPHA = 0.35
    # "auto" → try DISK+LightGlue, fall back to ORB+matcher
    # other values: "disk_lightglue", "orb_lk"
    FEATURE_BACKEND = "auto"
    FEATURE_HALF_PRECISION = True       # FP16 DISK on CUDA
    LIGHTGLUE_ADAPTIVE = True           # LightGlue early stopping + point pruning (speed);
                                        # False = full network (more matches, longer tracks)

    # --- Loop closure ---
    LOOP_MIN_KF_GAP = 10
    LOOP_MIN_SCORE = 0.55
    LOOP_MIN_MATCHES = 35

    # --- Depth motion-skip (reuse last depth when pose is near-identical) ---
    DEPTH_SKIP_TRANS = 0.05             # meters
    DEPTH_SKIP_ROT_RAD = 0.087          # ~5 degrees

    # --- YOLO (nano-only target) ---
    # Object detection is OPTIONAL and the heaviest extra load (YOLO-seg +
    # DeepSORT re-ID, both on the GPU). Default OFF for a smoother real-time
    # reconstruction on thermally-limited hardware: the dense mesh + SLAM are
    # unaffected; you only lose the object boxes / registry / scene graph.
    # Set True to get them back.
    SEMANTIC_DETECTION_ENABLED = False
    # Lightweight person tracking (active unless full semantics is on): runs
    # YOLO person-only (no DeepSORT) to report person↔camera distance, give
    # audio its source direction, and mask the person out of the environment
    # reconstruction. Set False for pure reconstruction (no YOLO at all).
    PERSON_TRACKING_ENABLED = True
    # Anti-hallucination for the person readout: stricter confidence, a minimum
    # box size, and temporal confirmation (must be seen `CONFIRM` detection
    # cycles before showing, missed `LOSE` cycles before clearing) + EMA smoothing.
    PERSON_CONFIDENCE = 0.6
    PERSON_MIN_BOX_FRAC = 0.012   # min person-box area as a fraction of the frame
    PERSON_CONFIRM_CYCLES = 2
    PERSON_LOSE_CYCLES = 4
    PERSON_SMOOTH_ALPHA = 0.35    # EMA weight for distance/azimuth/position
    YOLO_MODEL_NAME = "yolov8n-seg.pt"
    CONFIDENCE_THRESHOLD = 0.55         # raised: kills indoor false positives
    YOLO_IMG_SIZE = 320          # 320 is the real-time sweet spot for nano-seg;
                                 # 416 nearly doubled the GPU cost for little gain.
    YOLO_HALF_PRECISION = True
    # class 11 ("stop sign") removed; it produced false positives indoors.
    CLASS_LABELS = {
        0: "person",
        56: "chair",
        60: "diningtable",
        62: "tvmonitor",
        63: "laptop",
        64: "mouse",
        65: "remote",
        66: "keyboard",
        67: "cell phone",
        73: "book",
        74: "clock",
        75: "vase",
    }

    # --- DeepSORT ---
    DEEPSORT_MAX_AGE = 30
    DEEPSORT_N_INIT = 2

    # --- Dynamic-object filter ---
    # Classes whose detections should mask SLAM features (people are the
    # dominant moving object in indoor scenes: features on a walking
    # person would otherwise corrupt the trajectory).
    DYNAMIC_OBJECT_CLASSES = {"person"}
    DYNAMIC_MASK_PADDING_PX = 8
    # When a person dominates the webcam (the common "user at their desk"
    # case), masking out every person-feature can leave too few points to
    # match frame-to-frame, which sends the tracker into an endless reinit
    # loop and starves the dense reconstruction. If fewer than this many
    # features would survive the mask, we keep the unmasked set instead:
    # PnP-RANSAC rejects the moving outliers, and stable tracking with mild
    # contamination beats losing the map entirely.
    DYNAMIC_MASK_MIN_FEATURES = 80

    # --- Pipeline cadence (the big perf lever) ---
    # The 4 GB GPU can't run DISK+LightGlue (tracking), DPT (depth) and YOLO
    # (person) every frame. SLAM runs full rate (pose continuity matters), but
    # depth and person-detection are run every Nth semantic frame and reused
    # in between; quality is preserved (the TSDF integrates motion-gated and a
    # person moves slowly) while freeing the GPU so tracking stays fast/stable.
    # DEPTH_EVERY=1: attempt depth every frame, but the estimator's own
    # motion-skip cache returns instantly when the camera is still, so depth
    # only actually recomputes while you're moving (exactly when mapping needs
    # fresh depth). SLAM is the bottleneck, so this barely affects FPS but
    # greatly improves mapping during pans. Raise it to trade mapping for speed.
    DEPTH_EVERY = 1              # run DPT depth every Nth semantic frame
    PERSON_DETECT_EVERY = 3      # run YOLO person detection every Nth frame

    # --- Depth network ---
    # Relative inverse depth from MiDaS DPT-Hybrid (FP16). Alternatives:
    # "DPT_Large", "MiDaS_small", "DepthAnythingV2_Small" (needs transformers).
    DEPTH_MODEL_TYPE = "DPT_Hybrid"
    DEPTH_HALF_PRECISION = True
    # Skip the depth network while the camera is (nearly) static and reuse the
    # last prediction; alignment still runs every frame. Off by default.
    DEPTH_MOTION_SKIP = False

    # --- Depth alignment: confidence-gated scale-shift filtering (CG-SSF) ---
    # Variant (see backend/semantic/alignment.py): "cgssf" is the full method;
    # the others are the ablations of the paper (Tables II and III and the supplementary material).
    ALIGNMENT_VARIANT = "cgssf"
    DEPTH_Z_MIN = 0.1                # valid aligned depth [z_min, z_max] (map units)
    DEPTH_Z_MAX = 10.0
    # Map gauge: at (re)initialisation the median depth of the frame is set to
    # this many map units (s = 1/D0, b = 0); see SLAMTracker.
    GAUGE_MEDIAN_DEPTH = 2.0
    REQUIRE_DEPTH_INIT = True        # postpone map initialisation until depth exists
    CGSSF_N_MIN = 8                  # N_min
    CGSSF_KAPPA_MIN = 0.01           # kappa_min (observability)
    CGSSF_HUBER_C = 1.345            # Huber threshold (robust sigma units)
    CGSSF_IRLS_ITERS = 5
    CGSSF_GATE_CONF = 0.99           # 1 - gamma
    CGSSF_RESET_AFTER = 5            # M_r
    CGSSF_GATE_WINDOW = 30           # |W|: the gate threshold is scaled by the median of the last |W|
                                     # normalized innovations (0: fixed chi-square threshold)
    # Process noise indexed by the change of view (rotation angle in rad + translation / median
    # scene depth); "time" restores the earlier time-indexed model (ablation (f)).
    CGSSF_Q_MODE = "view"
    # Process noise eta_s, eta_b (per sqrt(unit of view change)), calibrated once on the held-out
    # TUM fr2/xyz sequence with tip/calibrate_eta.py --q-mode view (see results/calibration_view).
    CGSSF_ETA_S = 0.284
    CGSSF_ETA_B = 0.196
    # Measurement-noise floor sigma_s, sigma_b (relative to s), calibrated with eta.
    CGSSF_SIGMA_S = 0.02
    CGSSF_SIGMA_B = 0.02
    # Time-indexed process noise (ablation (f), the earlier design): calibrated on fr2/xyz against
    # elapsed time (results/calibration/calibration.json); eta per sqrt(second).
    CGSSF_ETA_S_TIME = 0.11
    CGSSF_ETA_B_TIME = 0.083
    CGSSF_SIGMA_S_TIME = 0.018
    CGSSF_SIGMA_B_TIME = 0.017
    # Designs of the closest methods (evaluation only, backend/semantic/alignment.py): window length of
    # the sliding-window joint fit (l) and the noise of the scale-only filter (m), chosen on the held-out
    # fr2/xyz sequence with ground-truth poses (results/design_tuning).
    ALIGN_WINDOW_FRAMES = 10         # W in {5, 10, 20, 40}: fr2/xyz AbsRel 13.5 / 11.2 / 11.3 / 11.8 %
    SCALE_KF_ETA = 0.071             # eta in {0.071, 0.284, 1.136}: fr2/xyz AbsRel 9.8 / 10.1 / 10.3 % (metric DAv2-S)
    SCALE_KF_SIGMA = 0.02

    # --- Audio detection (sound / voice activity only; no direction) ---
    AUDIO_DETECTION_ENABLED = True
    AUDIO_SAMPLE_RATE = 44100    # fallback; the real device's native rate is used
    AUDIO_BLOCKSIZE = 1024
    VAD_ABS_FLOOR = 0.002        # absolute RMS floor below which it's "idle"
                                 # (low: some laptop mics have very low gain)
    VAD_MARGIN = 4.0             # fire when RMS > noise_floor * this
    VAD_LOUD_RMS = 0.08          # above this → "loud" (clap/shout) vs "voice"
    VAD_LEVEL_FULL_RMS = 0.06    # RMS that maps to a full (1.0) UI meter

    # --- Streaming ---
    VIDEO_BROADCAST_HZ = 12
    STATE_BROADCAST_HZ = 15      # 15 Hz is plenty for the 3D view and roughly
                                 # halves the JSON-serialisation load that was
                                 # stealing the GIL from the perception threads.
    VIDEO_JPEG_QUALITY = 80      # crisper live overlay
    STREAM_MAX_POINTS = 2500     # cap point-cloud size shipped per map tick
    MAP_BROADCAST_HZ = 4         # heavy point-cloud/trajectory payload rate
                                 # (kept well below the 15 Hz light-state rate
                                 # so serialising thousands of points doesn't
                                 # starve the perception threads).
    MESH_BROADCAST_HZ = 1.0      # push the live reconstruction ~1×/second

    # --- Dense TSDF fusion (Open3D) -----------------------------------
    # Optional but ENABLED by default. Set to False to skip dense fusion
    # entirely (saves ~50–200 MB RAM + a couple of CPU ms per integrated
    # frame). The sparse SLAM map keeps working either way.
    DENSE_FUSION_ENABLED = True
    # GPU sparse voxel-block TSDF with per-pixel weights (tsdf_gpu.py); marching
    # cubes by Open3D on the CPU. Distances are in map units.
    TSDF_VOXEL_M = 0.03          # voxel size delta
    TSDF_TRUNC_M = 0.09          # truncation mu (3 x voxel)
    TSDF_MAX_DEPTH_M = 4.0       # pixels beyond this are not fused (far
                                 # monocular depth is the least reliable)
    # Only fold a new frame into the volume once the camera has actually
    # moved to a fresh viewpoint. Re-integrating a near-static view at
    # jittery monocular poses just smears the surface; gating on motion keeps
    # the reconstruction crisp and builds it up as you pan the camera.
    TSDF_MIN_MOVE_TRANS_M = 0.03
    TSDF_MIN_MOVE_ROT_RAD = 0.05
    TSDF_MESH_COOLDOWN_S = 1.2   # min seconds between marching-cubes passes
    TSDF_MESH_MAX_VERTICES = 28000  # decimate above this to keep payload small

    # --- CUDA tuning ---
    CUDNN_BENCHMARK = True
    # Optional device-memory cap in GB for the whole process (torch caching
    # allocator), e.g. 4.0 to emulate a 4-GB GPU. None = no cap.
    GPU_MEMORY_CAP_GB = (float(os.environ["AETHER_GPU_CAP_GB"])
                         if os.environ.get("AETHER_GPU_CAP_GB") else None)

    # --- Paths ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    os.makedirs(DATA_DIR, exist_ok=True)
