"""Synchronous (frame-by-frame, no frame dropping) execution of the AETHER-SLAM
pipeline, used to replay recorded sequences for evaluation.

Per frame, in order:
  1. person boxes (YOLOv8n-seg, every PERSON_DETECT_EVERY frames, reused in between)
  2. depth network -> relative inverse depth nu_k
  3. tracking: DISK + LightGlue against the previous frame, PnP-RANSAC + LM
     (map (re)initialisation with the gauge depth of the current frame)
  4. alignment of nu_k to the map with the PnP-inlier anchors (CG-SSF or an ablation)
  5. new map points from the aligned depth of the current frame
  6. keyframe insertion, loop detection, windowed bundle adjustment
  7. motion-gated TSDF integration with the alignment's per-pixel weights

The networks are shared between runs through `NetworkBundle`; with `cache=True`
their outputs for a frame index (and LightGlue matches for a pair of keypoint
subsets) are memoised, so several variants / seeds of one sequence run the
networks once. Outputs are identical to recomputation up to GPU arithmetic.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from backend.config import Config
from backend.semantic.alignment import AlignConfig, _sample, make_aligner
from backend.semantic.cg_ssf import CGSSFConfig
from backend.semantic.tsdf_gpu import WeightedTSDF
from backend.slam.loop_closure import LoopClosureDetector
from backend.slam.mapper import SLAMMapper
from backend.slam.optimization import run_local_bundle_adjustment, run_local_bundle_adjustment_lm
from backend.slam.tracker import SLAMTracker


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class CachedFeatureBackend:
    """Memoises DISK detections per frame index and LightGlue matches per pair of
    keypoint subsets. `current` must be set to the frame index before detect()."""

    def __init__(self, backend, enabled=True):
        self.backend = backend
        self.name = backend.name
        self.enabled = enabled
        self.current = None
        self._det = {}
        self._match = {}
        self.misses = {"detect": 0, "match": 0}

    def detect(self, image):
        key = self.current
        if self.enabled and key is not None and key in self._det:
            return self._det[key]
        f = self.backend.detect(image)
        f.frame_id, f.subset = key, None
        self.misses["detect"] += 1
        if self.enabled and key is not None:
            self._det[key] = f
        return f

    def match(self, f1, f2):
        key = (f1.frame_id, f1.subset, f2.frame_id, f2.subset)
        if self.enabled and f1.frame_id is not None and f2.frame_id is not None and key in self._match:
            return self._match[key]
        m = self.backend.match(f1, f2)
        self.misses["match"] += 1
        if self.enabled and f1.frame_id is not None and f2.frame_id is not None:
            self._match[key] = m
        return m

    def clear(self):
        self._det.clear(); self._match.clear()


class NetworkBundle:
    """The three networks, loaded once and shared by consecutive runs."""

    def __init__(self, config: Config, depth_model: str | None = None, person=True, cache=True):
        from backend.slam.feature_backend import make_feature_backend
        from backend.semantic.depth import DepthEstimator
        self.config = config
        self.features = CachedFeatureBackend(make_feature_backend(config), enabled=cache)
        self.depth = DepthEstimator(config, model_type=depth_model)
        self.person = None
        if person:
            from backend.semantic.detector import YOLODetector
            self.person = YOLODetector(config, person_only=True)
        self.cache = cache
        self._nu = {}
        self._boxes = {}

    def nu(self, index, frame):
        if self.cache and index in self._nu:
            return self._nu[index].to(self.depth.device, non_blocking=True).float(), 0.0
        _sync(); t0 = time.perf_counter()
        nu = self.depth.predict(frame)
        _sync(); dt = 1e3 * (time.perf_counter() - t0)
        if self.cache and nu is not None:
            # every run (including the first) sees the same float16-rounded output
            self._nu[index] = nu.half().cpu().pin_memory() if torch.cuda.is_available() else nu.half()
            nu = self._nu[index].to(self.depth.device, non_blocking=True).float()
        return nu, dt

    def person_boxes(self, index, frame):
        if self.person is None:
            return [], 0.0
        if self.cache and index in self._boxes:
            return self._boxes[index], 0.0
        _sync(); t0 = time.perf_counter()
        b = self.person.detect_person_boxes(frame)
        _sync(); dt = 1e3 * (time.perf_counter() - t0)
        if self.cache:
            self._boxes[index] = b
        return b, dt

    def clear(self):
        self._nu.clear(); self._boxes.clear(); self.features.clear()

    def set_depth_model(self, depth_model: str | None):
        """Replace the depth network (several networks evaluated in one process): the cached depth outputs
        are dropped; the cached keypoints, matches, and person boxes do not depend on it and are kept."""
        from backend.semantic.depth import DepthEstimator
        depth_model = depth_model or self.config.DEPTH_MODEL_TYPE
        if depth_model == self.depth.model_type:
            return
        self.depth.model = None
        self._nu.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.depth = DepthEstimator(self.config, model_type=depth_model)


def align_config_from(config: Config) -> AlignConfig:
    return AlignConfig(
        z_min=config.DEPTH_Z_MIN, z_max=config.DEPTH_Z_MAX,
        voxel=config.TSDF_VOXEL_M, trunc=config.TSDF_TRUNC_M,
        cgssf=CGSSFConfig(n_min=config.CGSSF_N_MIN, kappa_min=config.CGSSF_KAPPA_MIN,
                          huber_c=config.CGSSF_HUBER_C, irls_iters=config.CGSSF_IRLS_ITERS,
                          eta_s=config.CGSSF_ETA_S, eta_b=config.CGSSF_ETA_B,
                          sigma_s=config.CGSSF_SIGMA_S, sigma_b=config.CGSSF_SIGMA_B,
                          gate_conf=config.CGSSF_GATE_CONF, reset_after=config.CGSSF_RESET_AFTER,
                          gate_window=getattr(config, "CGSSF_GATE_WINDOW", 0),
                          q_mode=getattr(config, "CGSSF_Q_MODE", "time"),
                          z_min=config.DEPTH_Z_MIN, z_max=config.DEPTH_Z_MAX),
        cgssf_time=CGSSFConfig(n_min=config.CGSSF_N_MIN, kappa_min=config.CGSSF_KAPPA_MIN,
                               huber_c=config.CGSSF_HUBER_C, irls_iters=config.CGSSF_IRLS_ITERS,
                               eta_s=getattr(config, "CGSSF_ETA_S_TIME", config.CGSSF_ETA_S),
                               eta_b=getattr(config, "CGSSF_ETA_B_TIME", config.CGSSF_ETA_B),
                               sigma_s=getattr(config, "CGSSF_SIGMA_S_TIME", config.CGSSF_SIGMA_S),
                               sigma_b=getattr(config, "CGSSF_SIGMA_B_TIME", config.CGSSF_SIGMA_B),
                               gate_conf=config.CGSSF_GATE_CONF, reset_after=config.CGSSF_RESET_AFTER,
                               gate_window=getattr(config, "CGSSF_GATE_WINDOW", 0),
                               q_mode="time", z_min=config.DEPTH_Z_MIN, z_max=config.DEPTH_Z_MAX),
        window=int(getattr(config, "ALIGN_WINDOW_FRAMES", 10)),
        scale_eta=float(getattr(config, "SCALE_KF_ETA", config.CGSSF_ETA_S)),
        scale_sigma=float(getattr(config, "SCALE_KF_SIGMA", config.CGSSF_SIGMA_S)))


@dataclass
class StepResult:
    pose: np.ndarray | None
    z_eval: torch.Tensor | None
    status: str
    align_status: str
    timing_ms: dict = field(default_factory=dict)
    map_generation: int = 0
    n_anchors: int = 0
    diag: dict = field(default_factory=dict)


class SynchronousPipeline:
    def __init__(self, config: Config, nets: NetworkBundle, variant: str = "cgssf", seed: int = 0,
                 mask_persons: bool | None = None, oracle_poses: bool = False, extra_fusion=None):
        self.config = config
        self.nets = nets
        self.variant = variant
        self.seed = seed
        self.mask_persons = (variant != "cgssf_nomask") if mask_persons is None else mask_persons
        np.random.seed(seed)
        torch.manual_seed(seed)
        try:
            import cv2
            cv2.setRNGSeed(seed)
        except Exception:
            pass
        self.tracker = SLAMTracker(config, feature_backend=nets.features, seed=seed)
        self.mapper = SLAMMapper(config)
        self.loop = LoopClosureDetector(nets.features, min_kf_gap=config.LOOP_MIN_KF_GAP,
                                        min_score=config.LOOP_MIN_SCORE, min_matches=config.LOOP_MIN_MATCHES)
        self.aligner = make_aligner(variant, align_config_from(config), seed=seed)
        self.tsdf = WeightedTSDF(config.TSDF_VOXEL_M, config.TSDF_TRUNC_M, max_depth=config.TSDF_MAX_DEPTH_M)
        self.K = config.INTRINSICS_MATRIX.astype(np.float64)
        self._boxes = []
        self._frame_count = 0
        self._last_gen = None
        self._last_integ_pose = None
        self._lba_counter = 0
        self.n_reinit = 0
        self.status_counts = {}
        self.loop_events = 0
        # Ground-truth-pose evaluation: the caller passes the ground-truth camera pose of every
        # frame; the map frame is the (metric) ground-truth frame and is never reset.
        self.oracle = bool(oracle_poses)
        # Additional fusion volumes (evaluation of the accuracy-completeness trade-off), each
        # (name, kind, value): 'unc' = uncertainty weights and sigma_z <= value * mu,
        # 'uni' = unit weights and sigma_z <= value * mu, 'depth' = unit weights and z <= value.
        self.extra = [(n_, k_, float(v_), WeightedTSDF(config.TSDF_VOXEL_M, config.TSDF_TRUNC_M,
                                                      max_depth=config.TSDF_MAX_DEPTH_M))
                      for n_, k_, v_ in (extra_fusion or [])]

    def _box_list(self, dets):
        pad = int(getattr(self.config, "DYNAMIC_MASK_PADDING_PX", 8))
        return [(d["xyxy"][0] - pad, d["xyxy"][1] - pad, d["xyxy"][2] + pad, d["xyxy"][3] + pad) for d in dets]

    def _valid_mask(self, shape, device):
        valid = torch.ones(shape, dtype=torch.bool, device=device)
        h, w = shape
        for (x1, y1, x2, y2) in self._boxes:
            x1 = max(0, int(x1)); y1 = max(0, int(y1)); x2 = min(w - 1, int(x2)); y2 = min(h - 1, int(y2))
            if x2 > x1 and y2 > y1:
                valid[y1:y2 + 1, x1:x2 + 1] = False
        return valid

    def _anchors_outside_boxes(self, px, z):
        if not self._boxes or px.shape[0] == 0:
            return px, z
        keep = np.ones(px.shape[0], bool)
        for (x1, y1, x2, y2) in self._boxes:
            keep &= ~((px[:, 0] >= x1) & (px[:, 0] <= x2) & (px[:, 1] >= y1) & (px[:, 1] <= y2))
        return px[keep], z[keep]

    def _restrict_anchors(self, nu, px, z, index):
        """Anchor-restriction test (config.ANCHOR_BAND > 0; evaluation only): in every second block of
        ANCHOR_BAND_BLOCK frames, keep the anchors whose network output lies within +-ANCHOR_BAND of the
        median over the frame's anchors. The same anchors reach every estimator."""
        band = float(getattr(self.config, "ANCHOR_BAND", 0.0))
        block = max(1, int(getattr(self.config, "ANCHOR_BAND_BLOCK", 50)))
        if band <= 0 or px.shape[0] == 0 or (index // block) % 2 == 0:
            return px, z
        x = _sample(nu, px)
        pos = x > 0
        if not pos.any():
            return px, z
        keep = np.abs(x / np.median(x[pos]) - 1.0) <= band
        return px[keep], z[keep]

    @torch.no_grad()
    def step(self, frame_bgr: np.ndarray, t: float, index: int, gt_pose=None, gt_depth=None) -> StepResult:
        cfg = self.config
        timing = {}
        k = self._frame_count
        self._frame_count += 1
        if self.oracle:
            if gt_pose is None:            # no ground-truth pose for this frame: not processed
                return StepResult(pose=None, z_eval=None, status="NO_GT", align_status="none",
                                  map_generation=self.tracker.map_generation)
            self.tracker.oracle_T = np.asarray(gt_pose, np.float64)

        # 1. person boxes (cadence)
        every = max(1, int(getattr(cfg, "PERSON_DETECT_EVERY", 1)))
        if self.mask_persons and k % every == 0:
            dets, timing["seg"] = self.nets.person_boxes(index, frame_bgr)
            self._boxes = self._box_list(dets)
        elif not self.mask_persons:
            self._boxes = []

        # 2. depth network
        nu, timing["depth"] = self.nets.nu(index, frame_bgr)
        valid = self._valid_mask(nu.shape, nu.device) if nu is not None else None

        # 3. tracking
        _sync(); t0 = time.perf_counter()
        self.nets.features.current = index
        need_init = (not self.tracker.initialized
                     or self.tracker._consecutive_failures >= self.tracker._failure_tolerance - 1)
        init_depth = self.nets.depth.gauge_depth(nu, valid) if (need_init and nu is not None) else None
        gen_before = self.tracker.map_generation
        ok, T = self.tracker.track_pose(frame_bgr, dynamic_bboxes=self._boxes or None, init_depth=init_depth)
        _sync(); timing["track"] = 1e3 * (time.perf_counter() - t0)
        if self.tracker.map_generation != gen_before and gen_before > 0:
            self.n_reinit += 1
        if self._last_gen is not None and self.tracker.map_generation != self._last_gen and not self.oracle:
            self.aligner.reset(); self.tsdf.reset(); self._last_integ_pose = None
            for e in self.extra:
                e[3].reset()
        self._last_gen = self.tracker.map_generation

        res = None
        if ok and nu is not None:
            # 4. alignment
            t0 = time.perf_counter()
            px, zc = self.tracker.anchors_camera()
            px, zc = self._anchors_outside_boxes(px, zc)
            px, zc = self._restrict_anchors(nu, px, zc, index)
            if self.variant == "oracle_align":
                res = self.aligner.step(nu, valid, px, zc.astype(np.float64), t, T_WC=T, gt_depth=gt_depth)
            else:
                res = self.aligner.step(nu, valid, px, zc.astype(np.float64), t, T_WC=T)
            self.status_counts[res.status] = self.status_counts.get(res.status, 0) + 1
            # 5. new map points from the current aligned depth
            zs = res.z_spawn.cpu().numpy() if res.z_spawn is not None else None
            _sync(); timing["align"] = 1e3 * (time.perf_counter() - t0)
            self.tracker.spawn_points(zs)
            # 6. keyframes, loop detection, local BA
            t0 = time.perf_counter()
            if self.mapper.is_keyframe_needed(self.tracker.T_WC, t):
                self.mapper.add_keyframe(self.tracker.T_WC, self.tracker.tracked_pts_2d, self.tracker.tracked_pts_3d,
                                         self.tracker.tracked_pt_ids, frame_bgr)
                try:
                    ev = self.loop.check(self.mapper.keyframes[-1].id, self.tracker.last_features, t=t)
                    if ev is not None:
                        self.mapper.record_loop_event(ev); self.loop_events += 1
                    self.loop.add_keyframe(self.mapper.keyframes[-1].id, self.tracker.last_features)
                except Exception as e:
                    if isinstance(e, MemoryError) or "memory" in str(e).lower():
                        raise  # out of host or GPU memory: never continue silently
                self._lba_counter += 1
                if cfg.LBA_ENABLED and self._lba_counter % max(1, cfg.LBA_EVERY_KF) == 0:
                    if cfg.LBA_METHOD == "lm":
                        run_local_bundle_adjustment_lm(self.mapper, self.tracker, window_size=cfg.LBA_WINDOW_SIZE,
                                                       max_nfev=cfg.LBA_MAX_NFEV, huber_px=cfg.LBA_HUBER_PX)
                    else:
                        run_local_bundle_adjustment(self.mapper, self.tracker, window_size=cfg.LBA_WINDOW_SIZE)
            _sync(); timing["map"] = 1e3 * (time.perf_counter() - t0)
            # 7. motion-gated weighted TSDF integration
            t0 = time.perf_counter()
            Tc = self.tracker.T_WC
            moved = self._last_integ_pose is None
            if not moved:
                dtr = float(np.linalg.norm(Tc[:3, 3] - self._last_integ_pose[:3, 3]))
                Rr = self._last_integ_pose[:3, :3].T @ Tc[:3, :3]
                dr = float(np.arccos(np.clip((np.trace(Rr) - 1) / 2, -1, 1)))
                moved = dtr > cfg.TSDF_MIN_MOVE_TRANS_M or dr > cfg.TSDF_MIN_MOVE_ROT_RAD
            if moved and res.z_fuse is not None:
                rgb = torch.from_numpy(np.ascontiguousarray(frame_bgr[:, :, ::-1])).to(nu.device)
                if self.tsdf.integrate(res.z_fuse, res.w_fuse, rgb, self.K, Tc):
                    self._last_integ_pose = Tc.copy()
                if self.extra and getattr(self.aligner, "last_maps", None) is not None:
                    self._integrate_extra(rgb, valid, Tc)
            _sync(); timing["fuse"] = 1e3 * (time.perf_counter() - t0)
        elif ok:
            self.tracker.spawn_points(None)

        pose = self.tracker.T_WC.copy() if ok else None
        diag = {}
        if res is not None:
            f = getattr(self.aligner, "f", None)
            diag = dict(theta_out=self.aligner.theta, theta_fit=getattr(self.aligner, "last_fit", None),
                        med=getattr(self.aligner, "last_med", None), kappa=res.kappa,
                        d2=(f.last_d2 if f is not None else float("nan")))
        return StepResult(pose=pose, z_eval=(res.z_eval if res is not None else None),
                          status=self.tracker.last_status,
                          align_status=(res.status if res is not None else "none"),
                          timing_ms=timing, map_generation=self.tracker.map_generation,
                          n_anchors=(res.n_anchors if res is not None else 0), diag=diag)

    def _integrate_extra(self, rgb, valid, Tc):
        c = self.aligner.cfg
        z, sig = self.aligner.last_maps
        base = torch.isfinite(z) & (z >= c.z_min) & (z <= c.z_max)
        if valid is not None:
            base = base & valid
        for _, kind, val, vol in self.extra:
            if kind in ("unc", "uni"):
                keep = base & (sig <= val * c.trunc) if np.isfinite(val) else base
                if kind == "unc":
                    w = torch.clamp((c.voxel / sig.clamp(min=1e-12)) ** 2, max=1.0)
                else:
                    w = torch.ones_like(z)
            elif kind == "depth":
                keep = base & (z <= val)
                w = torch.ones_like(z)
            else:
                raise ValueError(kind)
            vol.integrate(torch.where(keep, z, torch.zeros_like(z)), torch.where(keep, w, torch.zeros_like(w)),
                          rgb, self.K, Tc)

    def extract_extra_meshes(self):
        return {name: vol.extract_mesh() for name, _, _, vol in self.extra}

    def extract_mesh(self):
        return self.tsdf.extract_mesh()
