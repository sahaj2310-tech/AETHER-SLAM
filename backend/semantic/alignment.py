"""Alignment of the depth network's relative inverse depth to the SLAM map gauge.

Every variant consumes
  * nu        -- raw network output (relative inverse depth), torch tensor (H, W) on the GPU,
  * valid     -- torch bool (H, W): pixels outside the dynamic-object (person) boxes,
  * anchors   -- pixel coordinates (N, 2) of tracked map points outside the boxes and
                 their camera-frame depth (N,) in map units,
and returns an AlignResult:
  * z_eval    -- aligned depth at every pixel, clipped to [z_min, z_max]   (evaluation)
  * z_fuse    -- depth handed to TSDF fusion, 0 where a pixel is not fused
  * w_fuse    -- per-pixel integration weight, 0 where a pixel is not fused
  * z_spawn   -- depth used to create new map points (range-valid, outside the boxes)

Variants (labels of Tables II-IV of the paper):
  proto_depthls  (a) prototype fit: least squares in the depth domain on
                     d_raw = 1/(nu_hat + 0.05), nu_hat = min-max normalised nu, no smoothing
  proto_ema      (b) the prototype: (a) + EMA of the parameters (0.7 old / 0.3 new),
                     reuse of the last parameters, and per-pixel EMA of the depth map
  inv_ls         (c) least squares in the inverse-depth domain, nu~ = nu / median(nu)
  inv_irls       (d) Huber IRLS in the inverse-depth domain (robust_affine_fit)
  irls_ema       (e) (d) + EMA (0.7) + reuse
  cgssf_timeq    (f) CG-SSF with process noise indexed by elapsed time (its own calibration)
  cgssf_norank1  (g) CG-SSF without rank-one updates (partially observable frames are skipped)
  cgssf_nogate   (h) CG-SSF without the chi-square gate
  cgssf_fixedgate (i) CG-SSF with the fixed threshold chi2_m(1 - gamma)
  cgssf_uniform  (j) CG-SSF, unit fusion weights
  cgssf          (k) CG-SSF with uncertainty-weighted fusion (full system)
  cgssf_nomask   full system with person masking disabled (handled by the caller)
  cgssf_eta4, cgssf_eta025  (k) with the calibrated process noise scaled by 4 or 1/4 (sensitivity)
  oracle_align   per-frame Huber fit of nu~ to ground-truth inverse depth (upper bound for any
                 per-frame affine alignment; needs ground-truth depth, i.e. ground-truth-pose runs)
Designs of the closest methods on the same inputs (ground-truth-pose runs):
  win_irls       (l) sliding-window joint fit: Huber IRLS of (s, b) over the anchors of the last W
                     frames (no covariance, gate, or treatment of partial observability)
  scale_kf       (m) scale-only recursive filter (rho = s nu~, no shift): per-frame Huber fit of s
                     with its variance, view-indexed random walk, normalised chi-square gate (1 dof)
                     and re-initialization as in CG-SSF; intended for metric priors
  metric_raw     a metric network's depth without alignment (rho = nu; metric map only)
(f), (i), (k), cgssf_nomask and the sensitivity variants fuse with uncertainty weights; all other
variants fuse every range-valid pixel with unit weight.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace

import numpy as np
import torch

from backend.semantic.cg_ssf import (_CHI2, _CHI2_MED, CGSSF, CGSSFConfig, _mad_sigma, robust_affine_fit,
                                     view_change)

VARIANTS = {
    "proto_depthls": "(a) depth-domain least squares (prototype fit), no smoothing",
    "proto_ema": "(b) depth-domain least squares + EMA + reuse + per-pixel EMA (prototype)",
    "inv_ls": "(c) inverse-domain least squares",
    "inv_irls": "(d) inverse-domain IRLS",
    "irls_ema": "(e) inverse-domain IRLS + EMA + reuse",
    "cgssf_timeq": "(f) CG-SSF with time-indexed process noise (weighted fusion)",
    "cgssf_norank1": "(g) CG-SSF without rank-one update",
    "cgssf_nogate": "(h) CG-SSF without gate",
    "cgssf_fixedgate": "(i) CG-SSF with a fixed chi-square threshold (weighted fusion)",
    "cgssf_uniform": "(j) CG-SSF, uniform fusion weights",
    "cgssf": "(k) CG-SSF with uncertainty-weighted fusion (full system)",
    "cgssf_nomask": "full system without person masking",
    "cgssf_eta4": "(k) with the calibrated process noise scaled by 4 (sensitivity)",
    "cgssf_eta025": "(k) with the calibrated process noise scaled by 1/4 (sensitivity)",
    "oracle_align": "per-frame robust fit to ground-truth depth (upper bound; ground-truth-pose runs only)",
    "win_irls": "(l) sliding-window joint robust fit over the anchors of the last W frames",
    "scale_kf": "(m) scale-only recursive filter with gate (design of the closest methods; metric priors)",
    "metric_raw": "metric network output without alignment (ground-truth-pose runs only)",
}


@dataclass
class AlignConfig:
    z_min: float = 0.1
    z_max: float = 10.0
    voxel: float = 0.03            # TSDF voxel size delta (map units), Eq. (12)
    trunc: float = 0.09            # TSDF truncation mu (map units), gating sigma_z <= mu
    ema_alpha: float = 0.7         # weight of the previous parameters in (b) and (e)
    proto_min_anchors: int = 10    # prototype: minimum in-view anchors for a fit
    proto_pixel_alpha: float = 0.5           # prototype per-pixel EMA (weight of the new frame)
    proto_pixel_reset_trans: float = 0.20    # reset the per-pixel EMA on > 0.2 map units ...
    proto_pixel_reset_rot: float = 0.26      # ... or > 15 deg between consecutive frames
    cgssf: CGSSFConfig = field(default_factory=CGSSFConfig)
    # time-indexed process noise with its own calibration (ablation (f), the earlier design)
    cgssf_time: CGSSFConfig = field(default_factory=CGSSFConfig)
    window: int = 10               # (l): frames pooled by the sliding-window fit
    scale_eta: float = 0.071       # (m): process noise of s (relative to s, per sqrt(unit of view change))
    scale_sigma: float = 0.02      # (m): measurement-noise floor of s (relative to s)


@dataclass
class AlignResult:
    z_eval: torch.Tensor | None = None
    z_fuse: torch.Tensor | None = None
    w_fuse: torch.Tensor | None = None
    z_spawn: torch.Tensor | None = None
    status: str = "none"
    theta: tuple | None = None
    n_anchors: int = 0
    kappa: float = float("nan")


def _sample(img: torch.Tensor, uv: np.ndarray) -> np.ndarray:
    """Nearest-pixel lookup of img at (u, v) pixel coordinates; returns float64 numpy."""
    if uv.shape[0] == 0:
        return np.zeros((0,), np.float64)
    h, w = img.shape
    u = np.clip(np.rint(uv[:, 0]).astype(np.int64), 0, w - 1)
    v = np.clip(np.rint(uv[:, 1]).astype(np.int64), 0, h - 1)
    ui = torch.from_numpy(u).to(img.device)
    vi = torch.from_numpy(v).to(img.device)
    return img[vi, ui].double().cpu().numpy()


def _median_value(nu: torch.Tensor, valid: torch.Tensor | None) -> float:
    sel = nu[valid] if valid is not None else nu.reshape(-1)
    sel = sel[sel > 0]
    return float(sel.median()) if sel.numel() > 0 else 1.0


def _median_normalise(nu: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
    return nu / max(_median_value(nu, valid), 1e-12)


def _rot_angle(R: np.ndarray) -> float:
    return float(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))


class BaseAligner:
    name = "base"
    weighted = False

    def __init__(self, cfg: AlignConfig, seed: int = 0):
        self.cfg = cfg
        self.seed = seed
        self.reset()

    def reset(self):
        self.theta = None

    # diagnostics of the last step: raw per-frame fit (before smoothing/filtering) and the
    # median used to normalise nu (nu~ = nu / last_med)
    last_fit = None
    last_med = None

    @property
    def ready(self) -> bool:
        return self.theta is not None

    def _range_outputs(self, z: torch.Tensor, valid: torch.Tensor | None, res: AlignResult):
        """Uniform-weight fusion of every range-valid pixel outside the boxes."""
        c = self.cfg
        rv = torch.isfinite(z) & (z >= c.z_min) & (z <= c.z_max)
        if valid is not None:
            rv = rv & valid
        res.z_fuse = torch.where(rv, z, torch.zeros_like(z))
        res.w_fuse = rv.to(z.dtype)
        res.z_spawn = res.z_fuse
        return res


# ----------------------------------------------------------------------------- (a), (b)
class PrototypeAligner(BaseAligner):
    """Depth-domain least squares of the earlier prototype (backend/semantic/depth.py, v1)."""
    name = "proto"

    def __init__(self, cfg: AlignConfig, seed: int = 0, ema: bool = False):
        self.ema = ema
        super().__init__(cfg, seed)

    def reset(self):
        self.theta = None            # (a, c): z = a * d_raw + c
        self._smoothed = None
        self._last_pose = None

    def step(self, nu, valid, anchors_uv, anchors_z, t, T_WC=None) -> AlignResult:
        c = self.cfg
        lo, hi = float(nu.amin()), float(nu.amax())
        nu_hat = (nu - lo) / (hi - lo + 1e-6)
        d_raw = 1.0 / (nu_hat + 0.05)
        status = "reuse"
        ok = (anchors_z > c.z_min) & (anchors_z < c.z_max)
        n = int(ok.sum())
        if n >= c.proto_min_anchors:
            d = _sample(d_raw, anchors_uv[ok])
            A = np.stack([d, np.ones_like(d)], 1)
            try:
                (a, b), *_ = np.linalg.lstsq(A, anchors_z[ok].astype(np.float64), rcond=None)
                if 0.1 < a < 20.0 and -10.0 < b < 10.0:
                    if self.ema and self.theta is not None:
                        al = c.ema_alpha
                        a = al * self.theta[0] + (1 - al) * a
                        b = al * self.theta[1] + (1 - al) * b
                    self.theta = (float(a), float(b))
                    status = "fit"
            except np.linalg.LinAlgError:
                pass
        if self.theta is not None:
            z = torch.clamp(self.theta[0] * d_raw + self.theta[1], c.z_min, c.z_max)
        else:                         # never aligned yet: generic relative-depth scaling
            z = torch.clamp(1.0 + d_raw / float(d_raw.amax()) * 4.0, c.z_min, c.z_max)
            status = "heuristic"
        if self.ema:                  # per-pixel temporal EMA, reset on large motion
            reset = self._smoothed is None or T_WC is None or self._last_pose is None
            if not reset:
                dt = float(np.linalg.norm(T_WC[:3, 3] - self._last_pose[:3, 3]))
                dr = _rot_angle(self._last_pose[:3, :3].T @ T_WC[:3, :3])
                reset = dt > c.proto_pixel_reset_trans or dr > c.proto_pixel_reset_rot
            if reset or self._smoothed.shape != z.shape:
                self._smoothed = z
            else:
                self._smoothed = c.proto_pixel_alpha * z + (1 - c.proto_pixel_alpha) * self._smoothed
            z = self._smoothed
            if T_WC is not None:
                self._last_pose = np.array(T_WC, copy=True)
        res = AlignResult(z_eval=z, status=status, theta=self.theta, n_anchors=n)
        return self._range_outputs(z, valid, res)


# ----------------------------------------------------------------------------- (c), (d), (e)
class InverseAligner(BaseAligner):
    """Per-frame fits in the inverse-depth domain, optionally with EMA + reuse."""
    name = "inverse"

    def __init__(self, cfg: AlignConfig, seed: int = 0, robust: bool = False, ema: bool = False):
        self.robust, self.ema = robust, ema
        super().__init__(cfg, seed)

    def reset(self):
        self.theta = None
        self.rng = np.random.default_rng(self.seed)

    def step(self, nu, valid, anchors_uv, anchors_z, t, T_WC=None) -> AlignResult:
        c = self.cfg
        self.last_med = max(_median_value(nu, valid), 1e-12)
        nu_t = nu / self.last_med
        self.last_fit = None
        ok = anchors_z > 1e-6
        n = int(ok.sum())
        status = "reuse"
        kappa = float("nan")
        n_min = c.cgssf.n_min if self.ema else 2
        if n >= max(n_min, 2):
            x = _sample(nu_t, anchors_uv[ok])
            rho = 1.0 / anchors_z[ok].astype(np.float64)
            if self.robust:
                init = tuple(self.theta) if (self.ema and self.theta is not None) else None
                fit = robust_affine_fit(x, rho, c.cgssf, init=init, rng=self.rng)
                th, kappa = fit["theta"], float(fit["kappa"])
            else:
                A = np.stack([x, np.ones_like(x)], 1)
                th, *_ = np.linalg.lstsq(A, rho, rcond=None)
            if np.all(np.isfinite(th)):
                self.last_fit = (float(th[0]), float(th[1]))
                if self.ema and self.theta is not None:
                    th = c.ema_alpha * np.asarray(self.theta) + (1 - c.ema_alpha) * np.asarray(th)
                self.theta = (float(th[0]), float(th[1]))
                status = "fit"
        if self.theta is None:
            return AlignResult(status="skipped", n_anchors=n, kappa=kappa)
        s, b = self.theta
        rho_img = nu_t * s + b
        z_raw = torch.where(rho_img > 0, 1.0 / rho_img.clamp(min=1e-12), torch.full_like(rho_img, float("inf")))
        z_eval = torch.clamp(1.0 / rho_img.clamp(min=1e-6), c.z_min, c.z_max)
        res = AlignResult(z_eval=z_eval, status=status, theta=self.theta, n_anchors=n, kappa=kappa)
        return self._range_outputs(z_raw, valid, res)


# ----------------------------------------------------------------------------- (f)-(i)
class CGSSFAligner(BaseAligner):
    name = "cgssf"

    def __init__(self, cfg: AlignConfig, seed: int = 0, weighted: bool = True, **switches):
        self.weighted = weighted
        self.switches = switches
        super().__init__(cfg, seed)

    def reset(self):
        cc = self.cfg.cgssf
        cfg = CGSSFConfig(**{**cc.__dict__, **self.switches})
        self.f = CGSSF(cfg, seed=self.seed)
        self._last_T = None

    @property
    def theta(self):
        return None if self.f.theta is None else (float(self.f.theta[0]), float(self.f.theta[1]))

    @theta.setter
    def theta(self, _):
        pass

    def step(self, nu, valid, anchors_uv, anchors_z, t, T_WC=None) -> AlignResult:
        c = self.cfg
        self.last_med = max(_median_value(nu, valid), 1e-12)
        nu_t = nu / self.last_med
        self.last_maps = None
        ok = anchors_z > 1e-6
        x = _sample(nu_t, anchors_uv[ok])
        rho = 1.0 / anchors_z[ok].astype(np.float64)
        dm = None
        if self.f.cfg.q_mode == "view" and T_WC is not None:
            dm = view_change(self._last_T, T_WC, self.f.median_depth())
        if T_WC is not None:
            self._last_T = np.array(T_WC, np.float64, copy=True)
        status = self.f.update(x, rho, t, view_change=dm)
        n = int(ok.sum())
        if self.f.theta is None:
            return AlignResult(status=status, n_anchors=n, kappa=self.f.last_kappa)
        z, sig_z, rv, w = self.f.depth_torch(nu_t, voxel=c.voxel, trunc=c.trunc)
        self.last_maps = (z, sig_z)       # for additional fusion volumes (evaluation only)
        z_eval = torch.clamp(torch.where(torch.isfinite(z), z, torch.full_like(z, c.z_max)), c.z_min, c.z_max)
        res = AlignResult(z_eval=z_eval, status=status, theta=self.theta, n_anchors=n, kappa=self.f.last_kappa)
        if not self.weighted:
            return self._range_outputs(z, valid, res)
        keep = rv & valid if valid is not None else rv
        res.z_fuse = torch.where(keep, z, torch.zeros_like(z))
        res.w_fuse = torch.where(keep, w, torch.zeros_like(w))
        # New map points use every range-valid pixel (no uncertainty gating), as in (h).
        sv = torch.isfinite(z) & (z >= c.z_min) & (z <= c.z_max)
        if valid is not None:
            sv = sv & valid
        res.z_spawn = torch.where(sv, z, torch.zeros_like(z))
        return res

    def gauge_parameters(self):
        return self.f.P, self.f.sigma2

    @property
    def last_fit(self):
        lf = self.f.last_fit
        return None if lf is None else (float(lf["theta"][0]), float(lf["theta"][1]))

    @last_fit.setter
    def last_fit(self, _):
        pass


# ----------------------------------------------------------------------------- oracle
class OracleAligner(BaseAligner):
    """Per-frame Huber fit of nu~ to the ground-truth inverse depth of the frame: the best
    per-frame affine alignment of this network (upper bound). Ground-truth-pose runs only,
    where the map gauge is metric."""
    name = "oracle"
    max_pixels = 20000

    def reset(self):
        self.theta = None
        self.rng = np.random.default_rng(self.seed)

    def step(self, nu, valid, anchors_uv, anchors_z, t, T_WC=None, gt_depth=None) -> AlignResult:
        c = self.cfg
        self.last_med = max(_median_value(nu, valid), 1e-12)
        nu_t = nu / self.last_med
        self.last_fit = None
        status = "reuse"
        n = 0
        if gt_depth is not None:
            g = gt_depth.to(nu.device, torch.float32)
            m = (g > c.z_min) & (g < c.z_max) & torch.isfinite(g)
            if valid is not None:
                m = m & valid
            idx = torch.nonzero(m.reshape(-1)).squeeze(1)
            n = int(idx.numel())
            if n >= 100:
                if n > self.max_pixels:
                    pick = torch.from_numpy(self.rng.choice(n, self.max_pixels, replace=False)).to(idx.device)
                    idx = idx[pick]
                x = nu_t.reshape(-1)[idx].double().cpu().numpy()
                rho = (1.0 / g.reshape(-1)[idx]).double().cpu().numpy()
                fit = robust_affine_fit(x, rho, c.cgssf, init=self.theta, rng=self.rng)
                if np.all(np.isfinite(fit["theta"])) and fit["theta"][0] > 0:
                    self.theta = (float(fit["theta"][0]), float(fit["theta"][1]))
                    self.last_fit = self.theta
                    status = "fit"
        if self.theta is None:
            return AlignResult(status="skipped", n_anchors=n)
        s_, b_ = self.theta
        rho_img = nu_t * s_ + b_
        z_raw = torch.where(rho_img > 0, 1.0 / rho_img.clamp(min=1e-12), torch.full_like(rho_img, float("inf")))
        z_eval = torch.clamp(1.0 / rho_img.clamp(min=1e-6), c.z_min, c.z_max)
        res = AlignResult(z_eval=z_eval, status=status, theta=self.theta, n_anchors=n)
        return self._range_outputs(z_raw, valid, res)


# ----------------------------------------------------------------------------- (l)
class WindowAligner(BaseAligner):
    """Sliding-window joint fit: Huber IRLS of (s, b) over the anchors of the last W frames, each
    sampled in its own normalised output nu~, i.e. parameters held constant over the window. Pools the
    anchors of several frames like a multi-frame (joint) alignment, without a covariance, a gate, or a
    treatment of partially observable frames."""
    name = "window"

    def reset(self):
        self.theta = None
        self.rng = np.random.default_rng(self.seed)
        self._win = deque(maxlen=max(1, int(self.cfg.window)))

    def step(self, nu, valid, anchors_uv, anchors_z, t, T_WC=None) -> AlignResult:
        c = self.cfg
        self.last_med = max(_median_value(nu, valid), 1e-12)
        nu_t = nu / self.last_med
        self.last_fit = None
        ok = anchors_z > 1e-6
        n = int(ok.sum())
        self._win.append((_sample(nu_t, anchors_uv[ok]), 1.0 / anchors_z[ok].astype(np.float64)))
        x = np.concatenate([w_[0] for w_ in self._win])
        rho = np.concatenate([w_[1] for w_ in self._win])
        status, kappa = "reuse", float("nan")
        if x.size >= max(c.cgssf.n_min, 2):
            fit = robust_affine_fit(x, rho, c.cgssf, init=self.theta, rng=self.rng)
            th = fit["theta"]
            if np.all(np.isfinite(th)):
                self.theta = (float(th[0]), float(th[1]))
                self.last_fit = self.theta
                kappa = float(fit["kappa"])
                status = "fit"
        if self.theta is None:
            return AlignResult(status="skipped", n_anchors=n, kappa=kappa)
        s, b = self.theta
        rho_img = nu_t * s + b
        z_raw = torch.where(rho_img > 0, 1.0 / rho_img.clamp(min=1e-12), torch.full_like(rho_img, float("inf")))
        z_eval = torch.clamp(1.0 / rho_img.clamp(min=1e-6), c.z_min, c.z_max)
        res = AlignResult(z_eval=z_eval, status=status, theta=self.theta, n_anchors=n, kappa=kappa)
        return self._range_outputs(z_raw, valid, res)


# ----------------------------------------------------------------------------- (m)
class ScaleFilterAligner(BaseAligner):
    """Scale-only recursive filter, the alignment design of the closest methods with metric priors:
    rho = s nu~ (no shift). Per frame a Huber IRLS fit of s with variance sigma^2 / sum(w nu~^2); a random
    walk indexed by the view change; the normalised chi-square gate (one degree of freedom) with
    re-initialization after M_r consistent rejections; the same thresholds as CG-SSF."""
    name = "scale"

    def reset(self):
        self.theta = None              # (s, 0.0)
        self.P = None                  # variance of s
        self.sigma2 = None
        self.rng = np.random.default_rng(self.seed)
        self._u = deque(maxlen=max(self.cfg.cgssf.gate_window, 1))
        self._rejected = []
        self._last_T = None
        self.last_d2 = float("nan")

    def _fit(self, x, rho, s0):
        cc = self.cfg.cgssf
        s = s0 if s0 is not None else float(np.median(rho / np.maximum(x, 1e-9)))
        w = np.ones_like(x)
        sxx = 0.0
        for _ in range(cc.irls_iters):
            r = rho - s * x
            a = np.abs(r) / _mad_sigma(r)
            w = np.where(a <= cc.huber_c, 1.0, cc.huber_c / np.maximum(a, 1e-12))
            sxx = float((w * x * x).sum())
            if sxx <= 1e-15:
                return None
            s = float((w * x * rho).sum() / sxx)
        r = rho - s * x
        sig2 = float((w * r * r).sum() / max(w.sum() - 1.0, 1.0))
        return s, sig2 / sxx, sig2

    def _consistent(self):
        rr = np.array(self._rejected)
        med = np.median(rr[:, 0])
        ok = int(np.sum((rr[:, 0] - med) ** 2 / np.maximum(rr[:, 1], 1e-18) <= _CHI2[1][self.cfg.cgssf.gate_conf]))
        return ok >= len(rr) // 2 + 1

    def step(self, nu, valid, anchors_uv, anchors_z, t, T_WC=None) -> AlignResult:
        c, cc = self.cfg, self.cfg.cgssf
        self.last_med = max(_median_value(nu, valid), 1e-12)
        nu_t = nu / self.last_med
        self.last_fit = None
        self.last_d2 = float("nan")
        ok = anchors_z > 1e-6
        n = int(ok.sum())
        if self.theta is not None and T_WC is not None and self._last_T is not None:   # prediction
            dm = view_change(self._last_T, T_WC, float(np.clip(1.0 / max(self.theta[0], 1e-9), c.z_min, c.z_max)))
            if dm is not None and np.isfinite(dm):
                self.P += self.theta[0] ** 2 * c.scale_eta ** 2 * max(dm, 0.0)
        if T_WC is not None:
            self._last_T = np.array(T_WC, np.float64, copy=True)
        status = "skipped" if self.theta is None else "reuse"
        f = None
        if n >= cc.n_min:
            f = self._fit(_sample(nu_t, anchors_uv[ok]), 1.0 / anchors_z[ok].astype(np.float64),
                          None if self.theta is None else self.theta[0])
        if f is not None and f[0] > 0 and np.isfinite(f[1]):
            s_m, var_m, sig2 = f
            self.last_fit = (s_m, 0.0)
            if self.theta is None:
                self.theta, self.P, self.sigma2 = (s_m, 0.0), var_m, sig2
                status = "init"
            else:
                s = self.theta[0]
                R = var_m + (s * c.scale_sigma) ** 2
                e = s_m - s
                S = self.P + R
                d2 = e * e / S
                self.last_d2 = d2
                lam = 1.0
                if cc.gate_window > 0 and len(self._u) >= max(1, cc.gate_window // 2):
                    lam = max(1.0, float(np.median(self._u)))
                if cc.gate_window > 0:
                    self._u.append(d2 / _CHI2_MED[1])
                if cc.use_gate and d2 > lam * _CHI2[1][cc.gate_conf]:
                    self._rejected.append((s_m, R, sig2))
                    status = "rejected"
                    if len(self._rejected) >= cc.reset_after and self._consistent():
                        rr = np.array(self._rejected)
                        self.theta = (float(np.median(rr[:, 0])), 0.0)
                        self.P, self.sigma2 = float(np.median(rr[:, 1])), float(np.median(rr[:, 2]))
                        self._rejected.clear()
                        status = "reset"
                else:
                    G = self.P / S
                    self.theta = (s + G * e, 0.0)
                    self.P = (1.0 - G) * self.P
                    self.sigma2 = sig2
                    self._rejected.clear()
                    status = "full"
        if self.theta is None:
            return AlignResult(status=status, n_anchors=n)
        rho_img = nu_t * self.theta[0]
        z_raw = torch.where(rho_img > 0, 1.0 / rho_img.clamp(min=1e-12), torch.full_like(rho_img, float("inf")))
        z_eval = torch.clamp(1.0 / rho_img.clamp(min=1e-6), c.z_min, c.z_max)
        res = AlignResult(z_eval=z_eval, status=status, theta=self.theta, n_anchors=n)
        return self._range_outputs(z_raw, valid, res)


class MetricRawAligner(BaseAligner):
    """A metric network's depth without alignment (rho = nu, i.e. s = median(nu), b = 0). Meaningful only
    in ground-truth-pose runs, where the map is metric."""
    name = "metric_raw"

    def step(self, nu, valid, anchors_uv, anchors_z, t, T_WC=None) -> AlignResult:
        c = self.cfg
        self.last_med = max(_median_value(nu, valid), 1e-12)
        self.theta = (self.last_med, 0.0)
        self.last_fit = None
        z_raw = torch.where(nu > 0, 1.0 / nu.clamp(min=1e-12), torch.full_like(nu, float("inf")))
        z_eval = torch.clamp(1.0 / nu.clamp(min=1e-6), c.z_min, c.z_max)
        res = AlignResult(z_eval=z_eval, status="fixed", theta=self.theta, n_anchors=int((anchors_z > 1e-6).sum()))
        return self._range_outputs(z_raw, valid, res)


def make_aligner(variant: str, cfg: AlignConfig, seed: int = 0) -> BaseAligner:
    if variant == "win_irls":
        return WindowAligner(cfg, seed)
    if variant == "scale_kf":
        return ScaleFilterAligner(cfg, seed)
    if variant == "metric_raw":
        return MetricRawAligner(cfg, seed)
    if variant == "proto_depthls":
        return PrototypeAligner(cfg, seed, ema=False)
    if variant == "proto_ema":
        return PrototypeAligner(cfg, seed, ema=True)
    if variant == "inv_ls":
        return InverseAligner(cfg, seed, robust=False, ema=False)
    if variant == "inv_irls":
        return InverseAligner(cfg, seed, robust=True, ema=False)
    if variant == "irls_ema":
        return InverseAligner(cfg, seed, robust=True, ema=True)
    if variant == "cgssf_norank1":
        return CGSSFAligner(cfg, seed, weighted=False, use_partial=False)
    if variant == "cgssf_nogate":
        return CGSSFAligner(cfg, seed, weighted=False, use_gate=False)
    if variant == "cgssf_uniform":
        return CGSSFAligner(cfg, seed, weighted=False)
    if variant in ("cgssf", "cgssf_nomask"):
        return CGSSFAligner(cfg, seed, weighted=True)
    if variant == "oracle_align":
        return OracleAligner(cfg, seed)
    if variant == "cgssf_timeq":
        return CGSSFAligner(replace(cfg, cgssf=cfg.cgssf_time), seed, weighted=True, q_mode="time")
    if variant == "cgssf_fixedgate":
        return CGSSFAligner(cfg, seed, weighted=True, gate_window=0)
    if variant in ("cgssf_eta4", "cgssf_eta025"):  # sensitivity: process noise scaled by 4 or 1/4
        k = 4.0 if variant == "cgssf_eta4" else 0.25
        return CGSSFAligner(replace(cfg, cgssf=replace(cfg.cgssf, eta_s=k * cfg.cgssf.eta_s, eta_b=k * cfg.cgssf.eta_b)),
                            seed, weighted=True)
    raise ValueError(f"unknown alignment variant {variant!r}; choose from {sorted(VARIANTS)}")
