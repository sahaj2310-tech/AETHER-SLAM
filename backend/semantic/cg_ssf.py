"""
Confidence-Gated Scale-Shift Filtering (CG-SSF).

Aligns an affine-invariant (relative) inverse-depth map nu_k(u) from a monocular
depth network to the gauge of the SLAM map, using sparse map points as anchors.

    rho_k(u) = s_k * nu~_k(u) + b_k + eps,      nu~ = nu / median(nu)       (Eq. 1)

Per frame:  robust closed-form fit (IRLS-Huber) + covariance          (Eqs. 2-3)
            observability test kappa = S_nn / (W * nubar^2)          (Eq. 4)
            Kalman predict / chi-square gate / full or rank-one update (Eqs. 5-7);
            the gate threshold is scaled by the median of the recent normalized innovations;
            the measurement covariance is Sigma_k + R0, where the floor
            R0 = s^2 diag(sigma_s^2, sigma_b^2) models errors common to all
            anchors of a frame (pose error, unmodelled network inconsistency)
            per-pixel depth + std + TSDF weight                       (Eqs. 10-12)

The filter itself is pure NumPy and O(N) per frame in the number of anchors;
`depth_torch` evaluates Eqs. (10)-(12) for a whole image on the GPU.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

# chi-square quantiles (1 - gamma) for 1 and 2 degrees of freedom
_CHI2 = {1: {0.95: 3.841459, 0.99: 6.634897, 0.999: 10.827566},
         2: {0.95: 5.991465, 0.99: 9.210340, 0.999: 13.815511}}
# chi-square medians for 1 and 2 degrees of freedom (normalize the innovations for the gate scale)
_CHI2_MED = {1: 0.454936, 2: 1.386294}


@dataclass
class CGSSFConfig:
    n_min: int = 8              # minimum anchors for any update
    kappa_min: float = 0.01     # minimum squared coefficient of variation of anchor disparities
    huber_c: float = 1.345      # Huber threshold (in robust-sigma units)
    irls_iters: int = 5
    ransac_iters: int = 64      # only used when no prior exists
    eta_s: float = 0.05         # process noise of s, relative to s, per sqrt(second)
    eta_b: float = 0.05         # process noise of b, relative to s, per sqrt(second)
    sigma_s: float = 0.0        # measurement-noise floor of s, relative to s
    sigma_b: float = 0.0        # measurement-noise floor of b, relative to s
    gate_conf: float = 0.99     # 1 - gamma
    reset_after: int = 5        # M_r consecutive rejected full-rank measurements
    gate_window: int = 30       # |W|: recent innovations whose median scales the gate threshold (0: fixed)
    z_min: float = 0.1
    z_max: float = 10.0
    use_gate: bool = True       # ablation switches
    use_partial: bool = True
    # process-noise index: "time" -> Q = s^2 diag(eta^2) dt (eta per sqrt(second));
    # "view" -> Q = s^2 diag(eta^2) dm, dm = view change between frames (rotation angle in rad
    # + camera translation / median scene depth), eta per sqrt(unit of view change)
    q_mode: str = "time"


def normalize_disparity(nu: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """nu~ = nu / median(nu) over valid pixels."""
    v = nu[valid] if valid is not None else nu
    med = np.median(v[v > 0]) if np.any(v > 0) else 1.0
    return nu / max(med, 1e-12)


def _wls(nu, rho, w):
    W = w.sum()
    nb = (w * nu).sum() / W
    rb = (w * rho).sum() / W
    dn = nu - nb
    Snn = (w * dn * dn).sum()
    Snr = (w * dn * (rho - rb)).sum()
    s = Snr / Snn if Snn > 1e-15 else 0.0
    b = rb - s * nb
    return s, b, W, nb, rb, Snn


def _mad_sigma(r):
    return 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-12


def robust_affine_fit(nu, rho, cfg: CGSSFConfig, init=None, rng=None):
    """IRLS with Huber weights in the inverse-depth domain. Returns a dict with the
    estimate, its covariance (Eq. 3), the robust residual scale and the sufficient statistics."""
    nu = np.asarray(nu, float); rho = np.asarray(rho, float)
    n = nu.size
    if init is None:  # minimal 2-point RANSAC for a robust start (no prior available)
        rng = rng or np.random.default_rng(0)
        best, best_cnt = None, -1
        thr = 0.05 * np.median(np.abs(rho)) + 1e-9
        for _ in range(cfg.ransac_iters):
            i, j = rng.choice(n, 2, replace=False)
            if abs(nu[i] - nu[j]) < 1e-9:
                continue
            s = (rho[i] - rho[j]) / (nu[i] - nu[j]); b = rho[i] - s * nu[i]
            cnt = np.sum(np.abs(rho - s * nu - b) < thr)
            if cnt > best_cnt:
                best, best_cnt = (s, b), cnt
        init = best if best is not None else (np.median(rho) / max(np.median(nu), 1e-9), 0.0)
    s, b = init
    w = np.ones(n)
    nb = rb = Snn = W = 0.0
    for _ in range(cfg.irls_iters):
        r = rho - s * nu - b
        sig = _mad_sigma(r)
        a = np.abs(r) / sig
        w = np.where(a <= cfg.huber_c, 1.0, cfg.huber_c / np.maximum(a, 1e-12))
        s, b, W, nb, rb, Snn = _wls(nu, rho, w)
    r = rho - s * nu - b
    sig2 = float((w * r * r).sum() / max(w.sum() - 2.0, 1.0))
    if Snn > 1e-15:
        cov = (sig2 / Snn) * np.array([[1.0, -nb], [-nb, nb * nb + Snn / W]])
    else:
        cov = np.full((2, 2), np.inf)
    kappa = Snn / (W * nb * nb) if nb != 0 else 0.0
    return dict(theta=np.array([s, b]), cov=cov, sigma2=sig2, W=W, nubar=nb, rhobar=rb,
                Snn=Snn, kappa=kappa, weights=w)


class CGSSF:
    """Recursive estimator of theta = [s, b] with chi-square gating and rank-one updates."""

    def __init__(self, cfg: CGSSFConfig | None = None, seed: int = 0):
        self.cfg = cfg or CGSSFConfig()
        self.theta = None
        self.P = None
        self.t = None
        self.sigma2 = None          # last residual variance (Eq. 10, second term)
        self._rejected = []         # buffer of rejected full-rank measurements
        self._rejected_rank1 = []   # buffer of rejected rank-one measurements (implied shift)
        self._u = deque(maxlen=max(self.cfg.gate_window, 1))  # recent d^2 / median(chi2_m), accepted or rejected
        self.last_lambda = 1.0      # gate scale of the last measurement
        self.rng = np.random.default_rng(seed)
        self.last_status = "none"
        self.last_kappa = float("nan")
        self.last_fit = None        # raw per-frame robust fit (for calibration / diagnostics)
        self.last_d2 = float("nan")  # normalized innovation squared of the last measurement (diagnostics)

    # ----- Eq. (5): random-walk prediction with gauge-invariant process noise
    def predict(self, t, view_change=None):
        if self.theta is None:
            self.t = t
            return
        dt = max(t - self.t, 0.0)
        s = max(abs(self.theta[0]), 1e-9)
        index = dt
        if self.cfg.q_mode == "view":
            index = max(float(view_change), 0.0) if (view_change is not None and np.isfinite(view_change)) else 0.0
        Q = (s * s) * np.diag([self.cfg.eta_s ** 2, self.cfg.eta_b ** 2]) * index
        self.P = self.P + Q
        self.t = t

    def update(self, nu_anchor, rho_anchor, t, view_change=None):
        """One CG-SSF step. Returns a status string:
        'init', 'full', 'partial', 'rejected', 'reset', or 'skipped'.
        view_change: view change since the previous step (used when q_mode == 'view')."""
        cfg = self.cfg
        self.predict(t, view_change)
        self.last_fit = None
        self.last_d2 = float("nan")
        n = len(nu_anchor)
        if n < cfg.n_min:
            self.last_status = "skipped"
            return self.last_status
        fit = robust_affine_fit(nu_anchor, rho_anchor, cfg,
                                init=None if self.theta is None else tuple(self.theta), rng=self.rng)
        self.last_kappa = float(fit["kappa"])
        self.last_fit = dict(theta=fit["theta"].copy(), kappa=float(fit["kappa"]), n=int(n), W=float(fit["W"]))
        full = fit["kappa"] >= cfg.kappa_min and np.all(np.isfinite(fit["cov"]))
        if self.theta is None:                       # initialization needs a full-rank fit
            if not full or fit["theta"][0] <= 0:
                self.last_status = "skipped"
                return self.last_status
            self.theta, self.P, self.sigma2 = fit["theta"].copy(), fit["cov"].copy(), fit["sigma2"]
            self.last_status = "init"
            return self.last_status
        s_ref = max(abs(self.theta[0]), 1e-9)
        R0 = (s_ref * s_ref) * np.diag([cfg.sigma_s ** 2, cfg.sigma_b ** 2])
        if full:
            H = np.eye(2); y = fit["theta"]; R = fit["cov"] + R0
        elif cfg.use_partial:
            H = np.array([[fit["nubar"], 1.0]]); y = np.array([fit["rhobar"]])
            R = np.array([[fit["sigma2"] / fit["W"]]]) + H @ R0 @ H.T
        else:
            self.last_status = "skipped"
            return self.last_status
        e = y - H @ self.theta
        S = H @ self.P @ H.T + R
        d2 = float(e @ np.linalg.solve(S, e))
        self.last_d2 = d2
        m = H.shape[0]
        if full:
            self._rejected_rank1.clear()
        # Eq. (6): the threshold is scaled by lambda (beta_k in the paper) = max(1, median of the last |W| normalized innovations),
        # computed before the current one is added; lambda = 1 while fewer than |W|/2 are available
        lam = 1.0
        if cfg.gate_window > 0 and len(self._u) >= max(1, cfg.gate_window // 2):
            lam = max(1.0, float(np.median(self._u)))
        self.last_lambda = lam
        if cfg.gate_window > 0:
            self._u.append(d2 / _CHI2_MED[m])
        if cfg.use_gate and d2 > lam * _CHI2[m][cfg.gate_conf]:
            if full:
                self._rejected.append((fit["theta"].copy(), (fit["cov"] + R0).copy(), fit["sigma2"]))
                if len(self._rejected) >= cfg.reset_after and self._consistent():
                    th = np.median(np.array([r[0] for r in self._rejected]), axis=0)
                    self.theta = th
                    self.P = np.median(np.array([r[1] for r in self._rejected]), axis=0)
                    self.sigma2 = float(np.median([r[2] for r in self._rejected]))
                    self._rejected.clear()
                    self.last_status = "reset"
                    return self.last_status
            else:
                # rank-one: the shift implied by the observed mean inverse depth at the current scale
                nb = fit["nubar"]
                self._rejected_rank1.append((float(fit["rhobar"] - self.theta[0] * nb), float(R[0, 0]), nb,
                                             fit["sigma2"]))
                if len(self._rejected_rank1) >= cfg.reset_after and self._consistent_rank1():
                    rr = np.array(self._rejected_rank1)
                    b_med, r_med, nb_med = np.median(rr[:, 0]), np.median(rr[:, 1]), np.median(rr[:, 2])
                    pss = self.P[0, 0]
                    self.theta = np.array([self.theta[0], b_med])
                    self.P = np.array([[pss, -nb_med * pss], [-nb_med * pss, r_med + nb_med * nb_med * pss]])
                    self.sigma2 = float(np.median(rr[:, 3]))
                    self._rejected_rank1.clear()
                    self.last_status = "reset"
                    return self.last_status
            self.last_status = "rejected"
            return self.last_status
        G = self.P @ H.T @ np.linalg.inv(S)          # Eq. (7)
        self.theta = self.theta + G @ e
        self.P = (np.eye(2) - G @ H) @ self.P
        self.P = 0.5 * (self.P + self.P.T)
        self.sigma2 = fit["sigma2"]
        self._rejected.clear()
        self._rejected_rank1.clear()
        self.last_status = "full" if full else "partial"
        return self.last_status

    def _consistent_rank1(self):
        """Rejected rank-one measurements agree on the implied shift (majority within the 1-dof gate)."""
        rr = np.array(self._rejected_rank1)
        med = np.median(rr[:, 0])
        ok = int(np.sum((rr[:, 0] - med) ** 2 / np.maximum(rr[:, 1], 1e-18) <= _CHI2[1][self.cfg.gate_conf]))
        return ok >= len(rr) // 2 + 1

    def _consistent(self):
        """Rejected measurements agree with their median (majority within the 2-dof gate)."""
        ths = np.array([r[0] for r in self._rejected])
        med = np.median(ths, axis=0)
        ok = 0
        for th, cov, _ in self._rejected:
            d = th - med
            try:
                ok += float(d @ np.linalg.solve(cov, d)) <= _CHI2[2][self.cfg.gate_conf]
            except np.linalg.LinAlgError:
                pass
        return ok >= len(self._rejected) // 2 + 1

    # ----- Eqs. (10)-(12): per-pixel depth, standard deviation and TSDF weight
    def median_depth(self, default=2.0):
        """Depth of the median-disparity pixel (nu~ = 1): 1 / (s + b), map units."""
        if self.theta is None or not (self.theta[0] + self.theta[1] > 0):
            return default
        return float(np.clip(1.0 / (self.theta[0] + self.theta[1]), self.cfg.z_min, self.cfg.z_max))

    def depth(self, nu_tilde, voxel=None, trunc=None):
        cfg = self.cfg
        s, b = self.theta
        rho = s * nu_tilde + b
        h0, h1 = nu_tilde, np.ones_like(nu_tilde)
        P = self.P
        var_rho = h0 * h0 * P[0, 0] + 2 * h0 * h1 * P[0, 1] + h1 * h1 * P[1, 1] + (self.sigma2 or 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(rho > 0, 1.0 / rho, np.inf)
        sig_z = z * z * np.sqrt(np.maximum(var_rho, 0.0))
        valid = (z >= cfg.z_min) & (z <= cfg.z_max)
        w = None
        if voxel is not None and trunc is not None:
            valid &= sig_z <= trunc
            w = np.where(valid, np.minimum(1.0, (voxel / np.maximum(sig_z, 1e-12)) ** 2), 0.0)
        return z, sig_z, valid, w

    def depth_torch(self, nu_tilde, voxel=None, trunc=None):
        """GPU version of `depth` for a torch tensor nu_tilde (H, W).
        Returns (z, sig_z, valid, w) as tensors; z is +inf where rho <= 0."""
        import torch
        cfg = self.cfg
        s, b = (float(x) for x in self.theta)
        P = self.P
        rho = nu_tilde * s + b
        var_rho = (nu_tilde * nu_tilde * float(P[0, 0]) + nu_tilde * (2.0 * float(P[0, 1]))
                   + float(P[1, 1]) + float(self.sigma2 or 0.0))
        z = torch.where(rho > 0, 1.0 / rho.clamp(min=1e-12), torch.full_like(rho, float("inf")))
        sig_z = z * z * torch.sqrt(var_rho.clamp(min=0.0))
        valid = (z >= cfg.z_min) & (z <= cfg.z_max)
        w = None
        if voxel is not None and trunc is not None:
            valid = valid & (sig_z <= trunc)
            w = torch.where(valid, torch.clamp((voxel / sig_z.clamp(min=1e-12)) ** 2, max=1.0),
                            torch.zeros_like(z))
        return z, sig_z, valid, w


def view_change(T_prev, T_cur, depth):
    """Rotation angle (rad) + camera translation / depth between two camera-to-world poses."""
    if T_prev is None or T_cur is None:
        return None
    R = np.asarray(T_prev, np.float64)[:3, :3].T @ np.asarray(T_cur, np.float64)[:3, :3]
    ang = float(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))
    tr = float(np.linalg.norm(np.asarray(T_cur, np.float64)[:3, 3] - np.asarray(T_prev, np.float64)[:3, 3]))
    return ang + tr / max(depth, 1e-6)


def calibrate_noise_view(fits, max_lag=10, min_n=50, min_kappa=0.05):
    """As calibrate_noise, with the elapsed time replaced by the accumulated view change.
    fits: list of (frame_index, s, b, n_anchors, kappa, cumulative_view_change). For every lag
    L = 1..max_lag, the robust variance D(L) of the relative changes of s and b/s over L frames is
    paired with the mean view change m_L accumulated over those L frames and fitted by
    D(L) = 2 sigma^2 + eta^2 m_L (eta per sqrt(unit of view change))."""
    fits = [f for f in fits if f[3] >= min_n and f[4] >= min_kappa and f[1] > 0]
    idx = np.array([f[0] for f in fits], int)
    s = np.array([f[1] for f in fits], float)
    b = np.array([f[2] for f in fits], float)
    M = np.array([f[5] for f in fits], float)
    pos = {k: n for n, k in enumerate(idx.tolist())}
    rv = lambda v: (1.4826 * np.median(np.abs(v - np.median(v)))) ** 2
    lags, mL, Ds, Db = [], [], [], []
    for L in range(1, max_lag + 1):
        a = np.array([n for n, k in enumerate(idx.tolist()) if k + L in pos], int)
        if a.size < 20:
            continue
        c = np.array([pos[idx[n] + L] for n in a], int)
        lags.append(L)
        mL.append(float(np.mean(M[c] - M[a])))
        Ds.append(rv((s[c] - s[a]) / s[a]))
        Db.append(rv((b[c] - b[a]) / s[a]))
    mL = np.array(mL)
    A = np.stack([2.0 * np.ones_like(mL), mL], 1)

    def fit(D):
        (sig2, eta2), *_ = np.linalg.lstsq(A, np.asarray(D), rcond=None)
        return float(np.sqrt(max(eta2, 0.0))), float(np.sqrt(max(sig2, 0.0)))
    eta_s, sigma_s = fit(Ds)
    eta_b, sigma_b = fit(Db)
    return dict(eta_s=eta_s, eta_b=eta_b, sigma_s=sigma_s, sigma_b=sigma_b, lags=lags, view=mL.tolist(),
                D_s=[float(x) for x in Ds], D_b=[float(x) for x in Db], n_frames=int(idx.size))


def calibrate_noise(fits, dt, max_lag=10, min_n=50, min_kappa=0.05):
    """Calibrate the process noise (eta_s, eta_b) and the measurement-noise floor
    (sigma_s, sigma_b) from per-frame robust fits of a held-out sequence.

    fits: list of (frame_index, s, b, n_anchors, kappa) in frame order; dt: frame period (s).
    Well-observed frames (n >= min_n, kappa >= min_kappa) give relative changes
    ds = (s_{k+L} - s_k) / s_k and db = (b_{k+L} - b_k) / s_k; their robust
    variance (1.4826 MAD)^2 is fitted, over lags L = 1..max_lag, by the variogram of a
    random walk observed in white noise,  D(L) = 2 sigma^2 + eta^2 * L * dt.
    Returns dict(eta_s, eta_b, sigma_s, sigma_b, lags, D_s, D_b, n_frames)."""
    fits = [f for f in fits if f[3] >= min_n and f[4] >= min_kappa and f[1] > 0]
    idx = np.array([f[0] for f in fits], int)
    s = np.array([f[1] for f in fits], float)
    b = np.array([f[2] for f in fits], float)
    pos = {k: n for n, k in enumerate(idx.tolist())}
    rv = lambda v: (1.4826 * np.median(np.abs(v - np.median(v)))) ** 2
    lags, Ds, Db = [], [], []
    for L in range(1, max_lag + 1):
        a = np.array([n for n, k in enumerate(idx.tolist()) if k + L in pos], int)
        if a.size < 20:
            continue
        c = np.array([pos[idx[n] + L] for n in a], int)
        lags.append(L)
        Ds.append(rv((s[c] - s[a]) / s[a]))
        Db.append(rv((b[c] - b[a]) / s[a]))
    lags = np.array(lags, float)
    A = np.stack([2.0 * np.ones_like(lags), lags * dt], 1)

    def fit(D):
        (sig2, eta2), *_ = np.linalg.lstsq(A, np.asarray(D), rcond=None)
        return float(np.sqrt(max(eta2, 0.0))), float(np.sqrt(max(sig2, 0.0)))
    eta_s, sigma_s = fit(Ds)
    eta_b, sigma_b = fit(Db)
    return dict(eta_s=eta_s, eta_b=eta_b, sigma_s=sigma_s, sigma_b=sigma_b,
                lags=lags.tolist(), D_s=[float(x) for x in Ds], D_b=[float(x) for x in Db],
                n_frames=int(idx.size))
