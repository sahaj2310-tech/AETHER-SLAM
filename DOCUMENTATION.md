# AETHER-SLAM: Technical Documentation

**A live 3D room scanner from a single camera.** Point an ordinary camera at a
space (or replay a recorded sequence), move it around, and the system builds a
coloured 3D mesh of the real environment live, then lets you export it
as a `.ply`. Underneath is a small visual-odometry front end, a monocular depth
network, and **confidence-gated scale-shift filtering (CG-SSF)**, an estimator
that turns the network's relative depth into temporally consistent, fusion-ready
depth with per-pixel uncertainty. Everything runs in one process with a GPU
footprint that fits a 4 GB card.

This document explains what the system does, the concepts and math behind it,
the architecture, every source file, the runtime data flow, configuration, APIs,
how it is evaluated, the measured performance on the current hardware, and the
design constraints.

---

## Table of contents

1. [Executive summary](#1-executive-summary)
2. [Usability & use cases](#2-usability--use-cases)
3. [Glossary](#3-glossary)
4. [System architecture](#4-system-architecture)
5. [The processing pipeline, step by step](#5-the-processing-pipeline-step-by-step)
6. [The mathematics](#6-the-mathematics)
7. [File-by-file reference (backend)](#7-file-by-file-reference-backend)
8. [File-by-file reference (frontend)](#8-file-by-file-reference-frontend)
9. [Evaluation tooling (`tip/`)](#9-evaluation-tooling-tip)
10. [Configuration reference](#10-configuration-reference)
11. [HTTP / WebSocket API](#11-http--websocket-api)
12. [Measured performance & benchmark results](#12-measured-performance--benchmark-results)
13. [What is new in v3](#13-what-is-new-in-v3)
14. [Design constraints](#14-design-constraints)
15. [Build & run](#15-build--run)

---

## 1. Executive summary

AETHER-SLAM turns a **single RGB camera** into:

- a **live dense 3D reconstruction** (a coloured triangle mesh),
- a **camera trajectory** and a sparse map from learned-feature visual odometry,
- **temporally consistent depth** for every frame, aligned to the map by CG-SSF,
- a **person detector** that removes people from tracking, alignment and the mesh,
  and reports the closest person's distance,
- a **sound/voice-activity detector**, an **auto-estimated floor plan**, and
- an **exportable `.ply` model**.

It is a **single Python process** with a **multi-threaded perception pipeline**
(one CUDA stream per GPU stage) feeding a **FastAPI + WebSocket** server and a
**React + Three.js** dashboard. No NeRF, no Gaussian splatting, no
microservices.

The scientific core (also the subject of the accompanying paper) is the depth
alignment: a closed-form robust fit with covariance, an observability test, a
Kalman filter with rank-one updates and a χ² innovation gate, a calibrated noise
model, and uncertainty-weighted TSDF fusion. Exponential smoothing and "reuse the
last good scale", the rules of the earlier prototype, are special cases of it.

---

## 2. Usability & use cases

- **Quick 3D capture of a room** without a depth camera, with the mesh exported
  to Blender / MeshLab.
- **Rough spatial measurement:** floor area and room extents from a casual scan
  (relative scale; see §14).
- **Robotics / AR prototyping:** a self-contained, readable monocular
  odometry + depth + fusion stack with a live pose and mesh stream.
- **Teaching computer vision:** every concept (features, PnP, triangulation,
  monocular depth, Kalman filtering, TSDF fusion) is in one small code base, with
  a live visualization and an evaluation harness on public benchmarks.

**How to use it:** launch the dashboard, then move the camera **slowly sideways
and forward** (translation, not just rotation), watch the mesh fill in, and click
**EXPORT .PLY**. Without a webcam, replay a recorded sequence (§15).

---

## 3. Glossary

**Monocular SLAM / visual odometry (VO).** Estimating the camera's motion (and a
map) from a single camera. Geometry is recovered only **up to an unknown global
similarity** (scale), the *map gauge*, and **pure rotation gives no parallax**.

**Map gauge.** The unit of length of the map. AETHER-SLAM fixes it at every map
(re)initialization by giving the first frame a median depth of `D0 = 2` map units.

**DISK / LightGlue.** A learned keypoint detector+descriptor (DISK) and a learned
attention-based matcher (LightGlue), both from `kornia`.

**PnP-RANSAC + LM.** Pose from 2D–3D correspondences with random sampling
(outlier rejection), refined by Levenberg-Marquardt on the inliers.

**Parallax triangulation.** Every map point remembers the ray of the observation
at which it was created. Once the angle between that ray and the current ray
reaches 2°, the point is triangulated from the two views (midpoint method) and
checked (in front of both cameras, ≤ 2 px reprojection error in both). These
triangulated points are the **anchors** of the depth alignment: geometry that
does not come from the depth network.

**Relative (affine-invariant) inverse depth.** Robust monocular depth networks
(MiDaS / DPT) predict inverse depth only up to an unknown scale `s` and shift `b`
per image: `ρ ≈ s·ν̃ + b`.

**CG-SSF (confidence-gated scale-shift filtering).** The estimator that tracks
`θ = [s, b]` over time: robust per-frame fit with covariance → observability test
(full / rank-one / no update) → Kalman predict → χ² gate → update; plus
re-initialization after `M_r` consistent rejections.

**Observability κ.** The squared coefficient of variation of the anchors'
disparities. When all anchors lie at nearly the same depth (a wall), `κ → 0`:
only the *mean* inverse depth is determined, so the filter applies a **rank-one**
update along that direction instead of a full one.

**Noise floor R₀.** Errors common to all anchors of a frame (pose error,
non-affine network inconsistency) are invisible to the per-frame covariance; a
calibrated floor `R₀ = s²·diag(σ_s², σ_b²)` is added to it.

**View-indexed process noise.** Between frames the parameters follow a random walk
whose variance grows with the change of view `Δm` (rotation angle in rad + camera
translation / median scene depth), not with elapsed time: a static camera keeps
its alignment, a fast one changes it quickly (`CGSSF_Q_MODE = "view"`; `"time"` is
the earlier time-indexed model, kept as variant `cgssf_timeq`).

**Normalized gate.** A measurement is rejected if its normalized innovation squared
`d²` exceeds `β·χ²_m(1−γ)`, where `β = max(1, median of d²/median(χ²_m) over the last 30
measurements)`. For a consistent filter `β ≈ 1` (the nominal χ² test); when a sequence moves
faster than the calibration sequence and the innovations are larger than predicted, `β`
raises the threshold so that genuine changes are not rejected, while a short burst of outliers
does not move the median (`CGSSF_GATE_WINDOW = 30`; 0 restores the fixed threshold, variant
`cgssf_fixedgate`).

**Variogram calibration.** The process noise `η` and the floor `σ` are calibrated
once on a held-out sequence (TUM fr2/xyz) by fitting `D(L) = 2σ² + η²·m̄_L` to the
robust variance of parameter changes over lags `L = 1..10`, where `m̄_L` is the mean
view change accumulated over `L` frames (`tip/calibrate_eta.py --q-mode view`, four
passes).

**TSDF.** Truncated signed distance function: a voxel volume that stores, per
voxel, a weighted average of the signed distance to the observed surface.
**Marching cubes** extracts the mesh.

**Uncertainty-weighted fusion.** CG-SSF gives every pixel a depth standard
deviation `σ_z`; pixels with `σ_z > μ` (the truncation) are discarded and the rest
are fused with weight `ω = min{1, (δ/σ_z)²}` (δ = voxel size).

**Capacity-one overwrite buffer.** A queue of size 1 whose producer overwrites
the old item; every stage always works on the newest data and nothing backs up.
It bounds the age of every output (Proposition 2 of the paper).

---

## 4. System architecture

```
                 webcam  or  recorded sequence replayed at its own timing (AETHER_SOURCE)
                                         │
                                  [Capture thread]
                                         │  frame_q (size-1, overwrite)
                                         ▼
   [SLAM thread, CUDA stream 1]  DISK (1024 kps) + LightGlue vs. ≤256 map points
                                 → PnP-RANSAC + LM → parallax triangulation (anchors)
                                 → new points from the latest aligned depth
                                 → keyframes, loop detection (logged)
                                         │  slam_result_q (pose, anchors, frame)
                                         ▼
   [Semantic thread, CUDA stream 2]  YOLOv8n-seg person boxes (mask)
                                     DPT-Hybrid relative inverse depth ν
                                     CG-SSF alignment → depth ẑ, σ_z, weights ω
                                         │  tsdf_input_q (depth, weights, pose)
                                         ▼
   [Fusion thread, CUDA stream 3]  weighted sparse voxel-block TSDF (GPU)
                                         │
   [Mesh thread]  marching cubes (Open3D, CPU) every ~1.2 s → decimated snapshot
   [Audio thread] microphone VAD      [Telemetry thread] measured fps, RAM, VRAM
                                         ▼
                        state estimator ──► asyncio broadcasters ──► WebSocket
                        (state 15 Hz · map 4 Hz · video 12 Hz · mesh 1 Hz)  ──► React + Three.js
```

**Design principles**

- **Freshness over completeness.** Size-1 overwrite buffers between stages: a slow
  stage skips frames instead of building a backlog; staleness is bounded by
  `2·ΣLᵢ` (stage service times) and measured live at `/api/perf`.
- **One process, one CUDA context.** Models are loaded once (serialized by a lock
  to avoid concurrent CUDA initialization); tensors are passed by reference; each
  GPU stage has its own CUDA stream so tracking and depth inference overlap.
- **Heavy work off the hot path.** Marching cubes runs on its own thread.
- **Generation-based reset.** Manual resets (`reset_generation`) and map
  re-initializations (`map_generation`) make every consumer drop stale state.

---

## 5. The processing pipeline, step by step

1. **Capture.** Webcam (MJPG 640×480 @ 30 Hz) or a recorded sequence (TUM RGB-D,
   Replica/NICE-SLAM renders, or a video file) replayed in real time at its own
   timestamps, undistorted and resized to 640 px width with matching intrinsics.

2. **Track** (`SLAMTracker.track`).
   - DISK detects 1024 keypoints. Keypoints inside person boxes are dropped
     (unless fewer than 80 would remain).
   - The ≤256 map points, each carrying the descriptor and pixel of its last
     observation, are matched with LightGlue against all 1024 keypoints. A point
     missed in a frame is kept for up to 3 frames at its predicted pixel.
   - PnP-RANSAC (8 px, 100 iterations) + LM refinement gives the pose.
   - Inliers are re-triangulated against their reference observation once the
     parallax reaches 2°; inliers with a valid triangulation become **anchors**.
   - New points are created for unmatched keypoints from the most recent aligned
     depth, spread over an 8×6 grid, up to the budget of 256.
   - Keyframes on motion; loop candidates are detected and verified (logged, not
     used for optimization). Windowed bundle adjustment exists but is off by
     default (no measurable gain).
   - Re-initialization after 12 consecutive failures (a new gauge, a new map).

3. **Perceive + align** (semantic thread).
   - YOLOv8n-seg person boxes (confidence ≥ 0.6, padded 8 px) → dynamic mask.
   - DPT-Hybrid (FP16) → relative inverse depth `ν`.
   - CG-SSF uses the anchors outside the mask to update `θ = [s, b]` and produces
     the aligned depth `ẑ`, its standard deviation `σ_z`, and fusion weights `ω`.
   - The range-valid aligned depth is published for the tracker (new points).

4. **Fuse** (fusion thread). Motion-gated (3 cm / 0.05 rad) integration of the
   weighted depth into a sparse voxel-block TSDF on the GPU (3-cm voxels,
   9-cm truncation, ≤ 4 map units). The mesh thread extracts, decimates
   (≤ 28k vertices for streaming) and measures the room every ~1.2 s.

5. **Hear** (audio thread). Self-healing microphone VAD; when a person is present
   and speaking, the audio source direction points at them.

6. **Broadcast.** Light state at 15 Hz, map at 4 Hz, annotated video at 12 Hz,
   mesh at ≤1 Hz.

---

## 6. The mathematics

**Observation model** (inverse-depth domain, network output normalized by its
median): `ρ(u) = s·ν̃(u) + b + ε(u)`.

**Robust closed-form fit** (Huber IRLS, 5 iterations, `c = 1.345`): for weights
`w_j`, `ŝ = S_νρ / S_νν`, `b̂ = ρ̄ − ŝ·ν̄`, with covariance
`Σ = σ̂²/S_νν · [[1, −ν̄], [−ν̄, ν̄² + S_νν/W]]`.

**Observability:** `κ = S_νν / (W·ν̄²)`. Full update if `N ≥ 8` and `κ ≥ 0.01`,
rank-one (`y = ρ̄`, `H = [ν̄, 1]`) if only `N ≥ 8`, none otherwise.

**Filter:** random walk `Q = s²·diag(η_s², η_b²)·Δm` (change of view; `Δt` in
`cgssf_timeq`); measurement covariance `R = Σ + R₀` (full) or `σ̂²/W + H R₀ Hᵀ`
(rank-one); normalized χ² gate at 99 % (`β`, see the glossary); Kalman update; re-initialization after 5 consistent rejections of one kind (full-rank:
median θ; rank-one: shift re-anchored to the median mean inverse depth).

**Per-pixel uncertainty and weights:**
`σ_ρ² = hᵀPh + σ̂²`, `ẑ = 1/ρ̂`, `σ_z = ẑ²·σ_ρ`, discard `σ_z > μ`,
`ω = min{1, (δ/σ_z)²}`, TSDF update `D ← (W·D + ω·min(sdf/μ, 1)) / (W + ω)`,
`W ← W + ω`.

**Special cases (Proposition 1):** with constant `R = r·I`, `Q = q·I` and no gate
the filter is an EMA with `α = 1 − λ`, `q/r = λ²/(1 − λ)`; a rejected measurement
is "reuse the last parameters". The prototype's `α = 0.7` ⇔ `q/r ≈ 0.13`.

**Staleness (Proposition 2):** with size-1 overwrite buffers, the age of a stage's
output satisfies `a_i ≤ a_{i−1} + 2·L_i`, hence `a_n ≤ a_0 + 2·Σ L_i`.

**Triangulation (midpoint):** for rays `c₀ + λ₀r₀` and `c₁ + λ₁r₁`, the closest
points are found in closed form and averaged; accepted if `λ₀, λ₁ > 0`, the depth
is in `[0.1, 10]` and both reprojection errors are ≤ 2 px.

---

## 7. File-by-file reference (backend)

```
backend/
├── config.py                  every tunable (camera, front end, CG-SSF, TSDF, streaming, CUDA)
├── main.py                    FastAPI app, threaded stages, broadcasters, /api/* endpoints
├── perf.py                    PerfMonitor: stage service times, throughput, capture→fusion age
├── sources.py                 TUM / Replica / video sources (undistortion, intrinsics, GT access)
├── persistence.py             session save / load (JSON + PLY)
├── audio/audio_detector.py    microphone VAD (self-healing)
├── fusion/state_estimator.py  single source of truth for the dashboard state
├── pipeline/orchestrator.py   LatestQueue (size-1 overwrite) + Stage (thread wrapper)
├── pipeline/synchronous.py    synchronous replay (benchmarks): NetworkBundle cache + SynchronousPipeline
├── semantic/cg_ssf.py         CG-SSF estimator, robust fit, variogram calibration
├── semantic/alignment.py      all alignment variants of the ablation (prototype … CG-SSF)
├── semantic/depth.py          depth-network wrapper (DPT-Hybrid/Large, MiDaS-small, Depth Anything V2)
├── semantic/tsdf_gpu.py       sparse voxel-block TSDF on the GPU with per-pixel weights
├── semantic/dense_fusion.py   live wrapper: integration, mesh extraction, room dimensions
├── semantic/detector.py       YOLOv8n-seg person boxes + PersonFilter (+ optional DeepSORT mode)
├── semantic/query.py, semantic_map.py   optional object registry / scene graph mode
└── slam/
    ├── feature_backend.py     DISK + LightGlue (and ORB fallback)
    ├── tracker.py             map points with descriptors, PnP, triangulation, anchors
    ├── mapper.py              keyframes + voxel-hashed point cloud for the dashboard
    ├── optimization.py        windowed bundle adjustment (sparse LM; legacy Adam), off by default
    └── loop_closure.py        descriptor-signature loop detection with geometric verification
```

Highlights:

- **`semantic/cg_ssf.py`**: `robust_affine_fit` (Eqs. 2–4 of the paper),
  `CGSSF.update` (Eqs. 5–7, gating, rank-one updates, both re-initialization
  rules), `CGSSF.depth_torch` (Eqs. 10–12: per-pixel depth, σ_z and weights on
  the GPU) and `calibrate_noise` (variogram fit, Eq. 8).
- **`semantic/alignment.py`**: one interface for every estimator of the paper's
  ablations, variants (a)–(k): `proto_depthls`, `proto_ema` (the exact old
  prototype: depth-domain LS on `1/(ν̂+0.05)`, EMA 0.7, reuse, per-pixel EMA),
  `inv_ls`, `inv_irls`, `irls_ema`, `cgssf_timeq`, `cgssf_norank1`, `cgssf_nogate`,
  `cgssf_fixedgate`, `cgssf_uniform`, `cgssf`, and `cgssf_nomask`; the designs of the closest methods (paper Table IV):
  `win_irls` (sliding-window joint fit over the anchors of the last W frames),
  `scale_kf` (scale-only filter with the CG-SSF gate and noise model, for metric
  priors) and `metric_raw` (a metric network's depth without alignment).
- **`semantic/tsdf_gpu.py`**: blocks of 8³ voxels allocated on demand and found
  with `torch.searchsorted` on packed block keys; integration fully on the GPU;
  the in-block voxel order matches Open3D's `VoxelBlockGrid`, so marching cubes
  is delegated to Open3D without re-ordering.
- **`slam/tracker.py`**: struct-of-arrays point state (position, last pixel,
  descriptor, age, reference observation, triangulation); `track_pose` (detect,
  match, PnP, triangulate, anchors) and `spawn_points` are split so the
  synchronous replay can align the current frame's depth in between.
- **`pipeline/synchronous.py`**: runs the stages in order without dropping
  frames; caches network outputs per frame index (float16-rounded, identical for
  every run) so that many variants / seeds of a sequence share one network pass.

---

## 8. File-by-file reference (frontend)

`frontend/src/App.jsx` (dashboard: WebSocket client, telemetry incl. the
**Depth Alignment (CG-SSF)** panel showing update status, anchors/inliers, s and b; the
3D viewport; person range; audio meter; floor plan; export/save/reset),
`components/Viewport3D.jsx` (Three.js scene: point cloud, trajectory, frustum,
dense mesh, person box, audio ray), `components/LandingPage.jsx` (landing page),
`components/ErrorBoundary.jsx`. Built with Vite into `backend/static/`.

---

## 9. Evaluation tooling (`tip/`)

| Script | Purpose |
| :--- | :--- |
| `run_suite.py` | Synchronous replay of a TUM / Replica sequence for many variants × seeds; in-process evaluation (ATE with Sim(3), depth AbsRel/δ₁/SC, Replica Acc/Comp/CR/F); writes `runs/<variant>/<sequence>/seed<k>/`. `--oracle-poses`: ground-truth camera poses (alignment and fusion without front-end errors); `--unc-eval`: quality of the per-pixel uncertainty σ_z (sparsification, AUSE/AURG, coverage); `--extra-fusion`: additional fusion volumes for the trade-off; `--diag`: gate diagnostics; `--config KEY=VALUE`: configuration overrides; `--band-variants`: the listed variants also in the anchor-restriction test (tag suffix `_band`); `--plan "MODEL|SUFFIX|VARIANTS|BAND_VARIANTS|KEY=VAL;…"`: several depth networks and settings in one process, sharing the cached keypoints, matches and person boxes (verified bit-identical to separate processes) |
| `run_benchmarks.sh` | The first benchmark (TUM fr1 ×9, fr3 ×3, Replica ×8; baselines (a)–(e)) |
| `run_revision.sh` | The runs of the revision, one job at a time: `tum_normal`, `replica_normal`, `tum_oracle`, `stx_oracle` (TUM fr3 structure vs. texture), `replica_oracle`, `band_oracle` and `band_replica` (anchor-restriction test on TUM and on Replica with reconstruction), `backbones`, `design_tune` (held-out choice of W and of the scale filter's noise on fr2/xyz), `design_tum` and `design_replica` (alignment designs of the closest methods with DPT-Hybrid and the relative and metric Depth Anything V2-S, normal and anchor-restricted) |
| `oracle_tools.py` | Ground-truth poses and depth per frame, gate diagnostics, uncertainty evaluation (`sparsification`) |
| `calibrate_eta.py` | Held-out calibration of η and σ on TUM fr2/xyz (fixed-point iterations) |
| `collect_results.py`, `collect_extra.py` | Tables III–VI, per-sequence tables, provenance, paired Wilcoxon tests (Holm-Bonferroni, bootstrap intervals), gate diagnostics, fusion trade-off, uncertainty quality, anchor-restriction test, share of partially observable frames |
| `runtime_bench.py` | Component latencies/memory + live end-to-end runs (uncapped and 4 GB cap) |
| `simulate_alignment.py` | Controlled simulation (20 runs) behind Table II / Fig. 2 |
| `eval_common.py`, `eval_depth.py`, `eval_recon.py` | ATE, depth and reconstruction metrics |
| `provenance.py` | Code hash (no git checkout) and environment description |

Protocol details (failures, per-segment scales after re-initialization,
deterministic network caching, seeds) are documented in `run_suite.py` and in
the paper's Section VI-A.

---

## 10. Configuration reference

| Setting | Default | Meaning |
| :--- | :--- | :--- |
| `CAPTURE_SOURCE` (env `AETHER_SOURCE`) | `webcam` | or a TUM / Replica folder / video file replayed live |
| `PORT` (env `AETHER_PORT`) | `8000` | server port (`run.bat --port`) |
| `DETECT_FEATURES` / `MAX_FEATURES` | 1024 / 256 | DISK keypoints per frame / tracked map points |
| `MAX_LOST_FRAMES` | 3 | frames a missed map point is kept for re-matching |
| `TRI_MIN_PARALLAX_DEG` / `TRI_MAX_REPROJ_PX` | 2.0 / 2.0 | triangulation of anchors |
| `ANCHOR_MODE` | `triangulated` | or `all` PnP inliers |
| `ANCHOR_BAND` / `ANCHOR_BAND_BLOCK` | 0.0 / 50 | anchor-restriction test (evaluation only; 0 = off): in every second block of 50 frames, only anchors within ±`ANCHOR_BAND` of the median network output reach the alignment |
| `LBA_ENABLED` / `LBA_METHOD` | False / `lm` | windowed bundle adjustment |
| `DEPTH_MODEL_TYPE` | `DPT_Hybrid` | `DPT_Large`, `MiDaS_small`, `DepthAnythingV2_Small`, `DepthAnythingV2_MetricIndoor_Small` (metric; its depth is inverted to metric inverse depth) |
| `ALIGNMENT_VARIANT` | `cgssf` | any variant of `alignment.VARIANTS` |
| `GAUGE_MEDIAN_DEPTH` | 2.0 | `D0` |
| `CGSSF_Q_MODE` | `view` | process noise per change of view (`time`: per second) |
| `CGSSF_ETA_S/B`, `CGSSF_SIGMA_S/B` | 0.284, 0.196, 0.02, 0.02 | calibrated noise (fr2/xyz, view-indexed) |
| `CGSSF_ETA_S/B_TIME`, `CGSSF_SIGMA_S/B_TIME` | 0.11, 0.083, 0.018, 0.017 | time-indexed variant `cgssf_timeq` |
| `CGSSF_N_MIN`, `CGSSF_KAPPA_MIN`, `CGSSF_GATE_CONF`, `CGSSF_RESET_AFTER`, `CGSSF_GATE_WINDOW` | 8, 0.01, 0.99, 5, 30 | filter |
| `ALIGN_WINDOW_FRAMES`, `SCALE_KF_ETA`, `SCALE_KF_SIGMA` | 10, 0.071, 0.02 | baselines of the paper's Table IV (evaluation only): window of `win_irls`; noise of `scale_kf` (chosen on the held-out fr2/xyz sequence) |
| `TSDF_VOXEL_M` / `TSDF_TRUNC_M` / `TSDF_MAX_DEPTH_M` | 0.03 / 0.09 / 4.0 | fusion (map units) |
| `PERSON_DETECT_EVERY` / `DEPTH_MOTION_SKIP` | 3 / False | cadences |
| `GPU_MEMORY_CAP_GB` (env `AETHER_GPU_CAP_GB`) | none | cap the process's GPU memory (e.g. 4) |

---

## 11. HTTP / WebSocket API

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/ws` | WS | `state` / `map` / `video` / `mesh` stream; send `"reset"` to restart |
| `/api/status` | GET | GPU, capture source, alignment variant, depth model, fusion stats |
| `/api/perf` | GET | measured stage service times (median/p95/max), throughput, capture→fusion age, Prop. 2 bound, GPU memory |
| `/api/perf/reset` | POST | restart the statistics (e.g. after warm-up) |
| `/api/export/mesh` | GET | full-resolution `room_mesh.ply` |
| `/api/save`, `/api/load` | POST / GET | persist / read the session |
| `/api/config`, `/api/query` | POST | runtime parameters / scene query (optional mode) |

---

## 12. Measured performance & benchmark results

<!-- PERF -->

All numbers below were measured on **NVIDIA GeForce RTX 3060** (12 GB), Intel(R) Core(TM) i5-14400F, 16 GB RAM, Windows 11, PyTorch 2.5.1 + CUDA 12.1. They are generated by `docs/fill_perf.py` from the measured results; the run logs, the aggregated results and the provenance of every benchmark number are available from the author on request.

### Live end-to-end (TUM fr1/room replayed at 30 Hz through the full server)

| | tracking FPS | frames reaching fusion / s | tracking (median ms) | depth + CG-SSF (median ms) | fusion (median ms) | frame age at fusion (median / p95 ms) | peak GPU memory |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| uncapped | 10.7 | 8.7 | 66.4 | 74.4 | 1.7 | 220 / 396 | 0.92 GB |
| 4 GB cap | 11.1 | 9.1 | 68.1 | 75.7 | 4.6 | 234 / 448 | 0.91 GB |

![live runtime](docs/live_runtime.png)

### Per-component latency (isolated, median after GPU sync)

| component | latency (ms) | memory (MB) | TUM fr1 AbsRel of the full system (seed 0) |
| :-- | --: | --: | --: |
| DISK (1024) + LightGlue | 36.1 | 451.7 | – |
| YOLOv8n-seg (320 px) | 6.6 | 79.0 | – |
| DPT-Hybrid (default) | 38.2 | 527.1 | 41.4 % |
| DPT-Large | 50.0 | 1318.7 | 37.9 % |
| MiDaS-small | 12.7 | 87.2 | 42.3 % |
| Depth Anything V2-S | 25.2 | 167.5 | 37.2 % |
| CG-SSF update (CPU) | 0.25 | – | – |
| depth, σ_z, weights (GPU) | 0.66 | – | – |
| weighted TSDF integration (GPU) | 3.9 | 234.8 | – |
| marching cubes (CPU thread) | 82.8 | – | – |

![model latency](docs/model_latency.png)

### Full-system ablation (TUM RGB-D fr1 × 9, Replica × 8, mean over seeds 0, 1, 2)

| variant | TUM AbsRel % ↓ | TUM δ₁ % ↑ | TUM SC ×10⁻² ↓ | Replica Acc. cm ↓ | Replica Comp. cm ↓ | Replica F % ↑ | Replica ATE cm ↓ |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| (a) Depth-domain LS (prototype fit) | 42.0 | 25.0 | 38.9 | 51.1 | 63.2 | 6.5 | 46.6 |
| (b) (a) + EMA + reuse (prototype) | 44.7 | 20.3 | 38.7 | 51.1 | 65.4 | 6.5 | 47.9 |
| (c) Inverse-domain LS | 73.7 | 31.9 | 50.6 | 216.0 | 23.1 | 9.6 | 25.0 |
| (d) Inverse-domain IRLS | 66.7 | 34.1 | 49.3 | 212.2 | 23.4 | 9.6 | 23.5 |
| (e) (d) + EMA + reuse | 37.5 | 31.8 | 28.8 | 82.8 | 22.3 | 10.6 | 22.5 |
| (f) CG-SSF, time-indexed process noise | 40.4 | 27.4 | 27.0 | 31.1 | 37.0 | 13.1 | 22.6 |
| (g) CG-SSF w/o rank-one | 40.8 | 27.3 | 26.8 | 66.7 | 23.1 | 10.4 | 22.8 |
| (h) CG-SSF w/o gate | 39.1 | 29.3 | 24.3 | 68.0 | 22.9 | 10.3 | 22.6 |
| (i) CG-SSF, fixed gate threshold | 38.9 | 27.7 | 29.1 | 31.1 | 36.9 | 12.3 | 23.4 |
| (j) CG-SSF, uniform weights | 40.3 | 27.9 | 26.9 | 68.0 | 23.6 | 10.0 | 22.9 |
| (k) CG-SSF, weighted (full) | 40.3 | 27.9 | 26.9 | 30.3 | 38.6 | 13.2 | 22.9 |

Paired Wilcoxon tests over sequences (full CG-SSF vs. the variant): scale consistency vs. prototype (b) p = 0.008, vs. (e) p = 1.000; AbsRel vs. (e) p = 0.164; Replica accuracy vs. uniform weights (j) p = 0.016, completion vs. (j) p = 0.078.

**Reading:** uncertainty-weighted fusion (k) more than halves the Replica accuracy error of uniform weights (j), 68.0 cm to 30.3 cm, at the cost of completion. In the full system, the lightweight front end chosen for the 4 GB budget (no bundle adjustment, no loop closure) dominates the remaining error, and the filtered variants (e) to (k) differ little in depth accuracy; the paper compares the estimators with ground-truth poses.

![ablation](docs/ablation.png)

### Trajectory (ATE RMSE after Sim(3) alignment)

- TUM fr1: 28.0 cm mean over the 7/9 sequences tracked without failure (fr1/room and fr1/rpy re-initialize). Published learned dense methods: 3.0–5.4 cm averaged over all nine (as reported in MASt3R-SLAM, Table 1).
- TUM fr3 dynamic (with / without person masking): walking_xyz 15.7 / 20.1 cm, walking_halfsphere 28.1 / 26.8 cm, sitting_xyz 14.3 / 10.6 cm.
- Replica: 22.9 cm; reconstruction Acc. 30.33 cm, Comp. 38.56 cm, completion ratio 14.1 %, F-score 13.2 % (published dense methods: 2.96–5.03 cm Acc., 64.7–83.7 % CR, measured by their authors on an RTX 3090 / A100; GlORIE-SLAM uses 15.2 GB of GPU memory).

<!-- /PERF -->

---

## 13. What is new in v3

- **CG-SSF integrated** in the live system and the benchmarks (it existed only
  as a reference script before); the old "least squares + EMA + reuse" path is
  kept as an ablation variant that reproduces the prototype exactly.
- **Geometric anchors:** parallax triangulation of map points; the alignment no
  longer fits the network to its own back-projected depth.
- **Gauge fixed from the network's depth** at (re)initialization (instead of an
  image-row heuristic).
- **Longer tracks:** 1024-keypoint detection pool, descriptors stored with the
  map points, lost points kept for 3 frames.
- **Weighted GPU TSDF** (per-pixel weights, GPU integration, Open3D meshing).
- **Per-stage CUDA streams**, a dedicated mesh thread, measured throughput and
  staleness at `/api/perf`.
- **Recorded-sequence camera** for machines without a webcam, and a
  deterministic synchronous replay with an evaluation harness.

---

## 14. Design constraints

- **Relative scale.** Monocular geometry has no metric scale; distances and room
  dimensions are in map units (`D0` = 2 units ≈ a typical indoor viewing
  distance), not metres.
- **Lightweight front end for a 4 GB budget.** Frame-to-map visual odometry without
  global optimization keeps the whole system within a 4 GB GPU; it can lose track
  on large textureless surfaces with fast motion. Learned SLAM systems with dense
  or patch-based bundle adjustment track more accurately with several times the
  memory (published GPU memory: DPV-SLAM 4.0 GB, DROID-SLAM 8.5 GB, GlORIE-SLAM
  15.2 GB).
- **Translate the camera.** Pure rotation gives no parallax and maps poorly.
- **Uncertainty gating trades completeness for accuracy:** far, uncertain pixels
  are not fused, so surfaces seen only from far away stay empty.
- **Loop closures are detected and logged** but not used to correct drift.
- **Audio direction is camera-derived** (toward the detected person).

---

## 15. Build & run

**Windows (native):** on the first run, `run.bat` creates `venv` with a signed
python.org Python 3.10–3.12, installs the packages and CUDA PyTorch 2.5.1 (NVIDIA
GPU) and builds the React frontend into `backend/static/`; later runs skip whatever
is up to date (packages are reinstalled only when `requirements.txt` changes, the
frontend only when its sources change). It then starts the server and opens
`http://localhost:8000` when it answers. Options: `--replay <folder|video>`,
`--cap4` (4 GB GPU emulation), `--port <n>` (also env `AETHER_PORT`), `--setup`,
`--rebuild`, `--no-browser`, `--help`. If the venv's base Python cannot start (e.g.
blocked by Windows Smart App Control, which blocks unsigned interpreters such as
uv-managed Python), `run.bat` repoints `venv\pyvenv.cfg` to a signed Python 3.10.

**Manual:** `venv\Scripts\python -m backend.main`, then open
`http://localhost:8000` and click **LAUNCH DASHBOARD**.

**Without a webcam, replay a recorded sequence at its original timing:**

```bat
set AETHER_SOURCE=C:\path\to\rgbd_dataset_freiburg1_desk
venv\Scripts\python -m backend.main
```

(TUM RGB-D folders with `rgb.txt`, NICE-SLAM Replica scene folders with
`results/` and `traj.txt`, or a video file.)

**Emulate a 4 GB GPU:** `set AETHER_GPU_CAP_GB=4` before starting.

**Docker (Linux + NVIDIA Container Toolkit):**
`docker build -t aether-slam .` then
`docker run --gpus all -p 8000:8000 --device /dev/video0 aether-slam`
(or `-e AETHER_SOURCE=/data/<sequence> -v <data>:/data`).

**Benchmarks:** see `tip/run_benchmarks.sh` (datasets: TUM RGB-D from
`cvg.cit.tum.de`, Replica renders + culled meshes from the NICE-SLAM release,
mirrored on Hugging Face as `voviktyl/Replica-SLAM`).

> If the `venv` was created on another machine and its base Python is missing,
> install the same minor version (e.g. `uv python install 3.10`) and point
> `venv\pyvenv.cfg`'s `home` at it; the installed packages keep working.

---

*Built by **Sahaj Sinha**. AETHER-SLAM turns a single camera into a live,
exportable 3D model of the world around it.*
