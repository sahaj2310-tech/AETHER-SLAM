"""
Controlled simulation of monocular-depth-prior alignment (Section VI-C of the manuscript).

Ground truth is known, so every number produced here is exactly reproducible:
    python simulate_alignment.py --seeds 20 --out sim_results.json

Scenario (fixed a priori, documented in the paper):
  * 1200 depth frames at 10 Hz per run, 20 runs (seeds 0..19).
  * Segments of 40-150 frames: textured (p=0.5), low-texture (p=0.2), planar view (p=0.15),
    clustered anchors (p=0.15).
      textured     : rho* ~ U(0.2, 1.4); anchors ~ Poisson(150); 5 % outliers
      low-texture  : rho* ~ U(0.2, 1.4); anchors ~ Poisson(10);  25 % outliers
      planar view  : rho* = rho_p (1 + U(-0.03, 0.03)), rho_p ~ U(0.3, 1.0);
                     anchors ~ Poisson(60); 10 % outliers (whole view is one surface)
      clustered    : rho* ~ U(0.2, 1.4), but anchors only on pixels with |rho* - rho_c| < 0.03,
                     rho_c ~ U(0.4, 1.0); anchors ~ Poisson(60); 10 % outliers
  * Structured-outlier events overlaid on textured segments (rate 1/80 frames, length U{2..12}):
    a fraction U(0.3, 0.6) of anchors lies on an unmasked moving object whose inverse depths are
    biased by a common factor U(1.3, 1.8) (mutually consistent, hence "precise but wrong").
  * True alignment: log s* random walk (0.005 / frame), shift b* random walk (0.002 / frame),
    abrupt jumps of log s* by +/-U(0.15, 0.30) at Poisson rate 1/400 frames,
    plus i.i.d. per-frame jitter of s* (2 %) -- network output is not temporally consistent.
  * Network output: nu~ = (rho* - b*)/s* * (1 + N(0, 0.05^2)), 2000 pixels per frame.
  * Anchor inverse depth: rho* + N(0, 0.02^2); outliers ~ U(0.05, 2.0).
Metrics: AbsRel over all pixels (z clipped to [0.1, 10]), catastrophic frames (AbsRel > 0.20),
         scale consistency SC = std_k log median(z*/z_hat).
"""
from __future__ import annotations
import argparse, json, time
import numpy as np
from cg_ssf import CGSSF, CGSSFConfig, robust_affine_fit
from backend.semantic.cg_ssf import calibrate_noise

FPS = 10.0
K = 1200
M = 2000


def make_run(seed):
    rng = np.random.default_rng(1000 + seed)
    # segment schedule
    types = []
    while len(types) < K:
        L = int(rng.integers(40, 151))
        t = rng.choice(["tex", "low", "planar", "cluster"], p=[0.5, 0.2, 0.15, 0.15])
        types += [t] * L
    types = types[:K]
    # transient structured-outlier events (only inside textured segments)
    ev = [None] * K
    k = 0
    while k < K:
        if types[k] == "tex" and rng.random() < 1.0 / 80:
            L = int(rng.integers(2, 13)); fr = rng.uniform(0.3, 0.6); c = rng.uniform(1.3, 1.8)
            for j in range(k, min(K, k + L)):
                if types[j] == "tex":
                    ev[j] = (fr, c); types[j] = "event"
            k += L
        else:
            k += 1
    # true alignment
    logs = np.log(0.6) + np.cumsum(rng.normal(0, 0.005, K))
    jumps = rng.random(K) < 1.0 / 400
    jump_vals = rng.choice([-1, 1], K) * rng.uniform(0.15, 0.30, K)
    logs = logs + np.cumsum(np.where(jumps, jump_vals, 0.0))
    b = rng.uniform(-0.1, 0.1) + np.cumsum(rng.normal(0, 0.002, K))
    s = np.exp(logs) * (1 + rng.normal(0, 0.02, K))
    frames = []
    for k in range(K):
        t = types[k]
        pool = None
        if t == "planar":
            rp = rng.uniform(0.3, 1.0)
            rho = rp * (1 + rng.uniform(-0.03, 0.03, M))
            lam, pout = 60, 0.10
        elif t == "cluster":
            rho = rng.uniform(0.2, 1.4, M)
            rc = rng.uniform(0.4, 1.0)
            pool = np.flatnonzero(np.abs(rho - rc) < 0.03)
            lam, pout = 60, 0.10
        else:  # textured, low-texture, or textured frame during an event
            rho = rng.uniform(0.2, 1.4, M)
            lam, pout = (10, 0.25) if t == "low" else (150, 0.05)
        nu = (rho - b[k]) / s[k] * (1 + rng.normal(0, 0.05, M))
        nu = np.maximum(nu, 1e-3)
        n = int(rng.poisson(lam))
        cand = np.arange(M) if pool is None else pool
        idx = rng.choice(cand, size=min(n, cand.size), replace=False)
        ra = rho[idx] + rng.normal(0, 0.02, idx.size)
        out = rng.random(idx.size) < pout
        ra[out] = rng.uniform(0.05, 2.0, out.sum())
        if ev[k] is not None:
            fr, c = ev[k]
            dyn = rng.random(idx.size) < fr
            ra[dyn] = rho[idx][dyn] * c
        frames.append(dict(type=t, rho=rho, nu=nu, a_nu=nu[idx], a_rho=ra))
    return dict(types=types, s=s, b=b, frames=frames)


def ols(nu, rho):
    A = np.stack([nu, np.ones_like(nu)], 1)
    th, *_ = np.linalg.lstsq(A, rho, rcond=None)
    return th


# ---------------------------------------------------------------- estimators
class PerFrame:
    def __init__(self, robust):
        self.robust, self.cfg, self.theta = robust, CGSSFConfig(), None
    def step(self, f, t):
        if len(f["a_nu"]) >= 2:
            th = robust_affine_fit(f["a_nu"], f["a_rho"], self.cfg)["theta"] if self.robust else ols(f["a_nu"], f["a_rho"])
            if np.all(np.isfinite(th)):
                self.theta = th
    def depth(self, nu):
        s, b = self.theta
        return 1.0 / np.maximum(s * nu + b, 1e-6)


class SmoothReuse:
    """Robust inverse-domain fit + EMA + reuse-on-failure (anchor count rule)."""
    def __init__(self, alpha=0.7, n_min=8):
        self.alpha, self.n_min, self.cfg, self.theta = alpha, n_min, CGSSFConfig(), None
    def step(self, f, t):
        if len(f["a_nu"]) < self.n_min:
            return
        th = robust_affine_fit(f["a_nu"], f["a_rho"], self.cfg,
                               init=None if self.theta is None else tuple(self.theta))["theta"]
        if not np.all(np.isfinite(th)):
            return
        self.theta = th if self.theta is None else self.alpha * self.theta + (1 - self.alpha) * th
    def depth(self, nu):
        s, b = self.theta
        return 1.0 / np.maximum(s * nu + b, 1e-6)


class Prototype:
    """Earlier pipeline: depth-domain least squares on d_raw = 1/(nu + 0.05),
    optional EMA (alpha = 0.7, the prototype's value) and reuse when fewer than
    10 anchors are available, depth clipped to [0.1, 10]."""
    def __init__(self, alpha=0.7, n_min=10):
        self.alpha, self.n_min, self.theta = alpha, n_min, None
    def step(self, f, t):
        if len(f["a_nu"]) < max(self.n_min, 2):
            return
        draw = 1.0 / (f["a_nu"] + 0.05)
        za = 1.0 / np.maximum(f["a_rho"], 1e-3)
        th = ols(draw, za)
        if not np.all(np.isfinite(th)):
            return
        self.theta = th if (self.theta is None or self.alpha == 0) else self.alpha * self.theta + (1 - self.alpha) * th
    def depth(self, nu):
        a, c = self.theta
        return a / (nu + 0.05) + c


class Filter:
    def __init__(self, **kw):
        self.f = CGSSF(CGSSFConfig(**kw))
    @property
    def theta(self):
        return self.f.theta
    def step(self, fr, t):
        self.f.update(fr["a_nu"], fr["a_rho"], t)
    def depth(self, nu):
        s, b = self.f.theta
        return 1.0 / np.maximum(s * nu + b, 1e-6)


class Oracle:
    def __init__(self, run):
        self.run, self.theta, self.k = run, None, 0
    def step(self, f, t):
        self.theta = np.array([self.run["s"][self.k], self.run["b"][self.k]]); self.k += 1
    def depth(self, nu):
        s, b = self.theta
        return 1.0 / np.maximum(s * nu + b, 1e-6)


def calibrate_eta(seed=10_000):
    """Noise calibration on a held-out run (never used for testing), with the same
    procedure as on real data (backend/semantic/cg_ssf.calibrate_noise): per-frame
    robust fits on well-observed frames (N >= 50 anchors, kappa >= 0.05) and a
    variogram fit D(L) = 2 sigma^2 + eta^2 L dt over lags 1..10."""
    run = make_run(seed)
    cfg = CGSSFConfig()
    fits = []
    for k, f in enumerate(run["frames"]):
        if len(f["a_nu"]) >= 8:
            r = robust_affine_fit(f["a_nu"], f["a_rho"], cfg)
            fits.append((k, float(r["theta"][0]), float(r["theta"][1]), len(f["a_nu"]), float(r["kappa"])))
    c = calibrate_noise(fits, 1.0 / FPS)
    return c["eta_s"], c["eta_b"], c["sigma_s"], c["sigma_b"]


ETA_S, ETA_B, SIG_S, SIG_B = calibrate_eta()
NOISE = dict(eta_s=ETA_S, eta_b=ETA_B, sigma_s=SIG_S, sigma_b=SIG_B)

ESTIMATORS = {
    "proto_perframe": lambda run: Prototype(alpha=0.0),
    "proto_ema_reuse": lambda run: Prototype(alpha=0.7),
    "inv_ols": lambda run: PerFrame(robust=False),
    "inv_irls": lambda run: PerFrame(robust=True),
    "irls_ema_reuse": lambda run: SmoothReuse(alpha=0.7),
    "cgssf_nogate": lambda run: Filter(use_gate=False, **NOISE),
    "cgssf_nopartial": lambda run: Filter(use_partial=False, **NOISE),
    "cgssf_fixedgate": lambda run: Filter(gate_window=0, **NOISE),
    "cgssf": lambda run: Filter(**NOISE),
    "oracle": lambda run: Oracle(run),
}


def evaluate(est, run):
    absrel, logm, scale_ratio, statuses = [], [], [], []
    for k, f in enumerate(run["frames"]):
        est.step(f, k / FPS)
        if getattr(est, "f", None) is not None:
            statuses.append(est.f.last_status)
        if est.theta is None:
            absrel.append(np.nan); logm.append(np.nan); scale_ratio.append(np.nan); continue
        zt = 1.0 / f["rho"]
        zh = np.clip(est.depth(f["nu"]), 0.1, 10.0)
        absrel.append(float(np.mean(np.abs(zh - zt) / zt)))
        logm.append(float(np.log(np.median(zt / zh))))
        scale_ratio.append(float(est.theta[0] / run["s"][k]) if not isinstance(est, Prototype) else np.nan)
    a = np.array(absrel); l = np.array(logm)
    ok = np.isfinite(a)
    types = np.array(run["types"])
    by_type = {t: float(np.nanmean(a[types == t])) for t in ("tex", "low", "planar", "cluster", "event")}
    return dict(absrel=float(np.nanmean(a)), fail=float(np.mean(a[ok] > 0.20)),
                sc=float(np.nanstd(l)), series=a.tolist(), ratio=scale_ratio, status=statuses,
                by_type=by_type)


def _one_seed(seed):
    run = make_run(seed)
    out = {}
    for name, mk in ESTIMATORS.items():
        est = mk(run)
        t0 = time.perf_counter()
        r = evaluate(est, run)
        r["sec_per_frame"] = (time.perf_counter() - t0) / K
        if seed != 0:
            r.pop("series"); r.pop("ratio"); r.pop("status")
        out[name] = r
    sens = {"ema_alpha": {}, "cgssf_eta": {}, "cgssf_sigma": {}, "cgssf_gate": {}, "cgssf_kappa": {}}
    sens["ema_alpha"][0.7] = out["irls_ema_reuse"]["absrel"]
    sens["cgssf_eta"][1.0] = out["cgssf"]["absrel"]
    sens["cgssf_sigma"][1.0] = out["cgssf"]["absrel"]
    sens["cgssf_gate"][0.99] = out["cgssf"]["absrel"]
    sens["cgssf_kappa"][0.01] = out["cgssf"]["absrel"]
    for alpha in [0.0, 0.5, 0.9, 0.98]:
        sens["ema_alpha"][alpha] = evaluate(SmoothReuse(alpha=alpha), run)["absrel"]
    for mult in [0.25, 4.0]:
        sens["cgssf_eta"][mult] = evaluate(Filter(**{**NOISE, "eta_s": mult * ETA_S, "eta_b": mult * ETA_B}), run)["absrel"]
    for mult in [0.0, 4.0]:
        sens["cgssf_sigma"][mult] = evaluate(Filter(**{**NOISE, "sigma_s": mult * SIG_S, "sigma_b": mult * SIG_B}), run)["absrel"]
    for g in [0.95, 0.999]:
        sens["cgssf_gate"][g] = evaluate(Filter(**NOISE, gate_conf=g), run)["absrel"]
    for km in [0.003, 0.03]:
        sens["cgssf_kappa"][km] = evaluate(Filter(**NOISE, kappa_min=km), run)["absrel"]
    return seed, out, sens, (run["types"], run["s"].tolist()) if seed == 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10, help="number of runs (seeds 0..N-1)")
    ap.add_argument("--chunk", type=str, default=None, help="A:B -> run seeds A..B-1 and save a partial file")
    ap.add_argument("--aggregate", nargs="*", default=None, help="partial files to aggregate")
    ap.add_argument("--out", default="sim_results.json")
    args = ap.parse_args()
    print(f"calibrated eta_s={ETA_S:.5f}  eta_b={ETA_B:.5f}  sigma_s={SIG_S:.5f}  sigma_b={SIG_B:.5f}")
    if args.aggregate is None:
        a, b = (0, args.seeds) if args.chunk is None else map(int, args.chunk.split(":"))
        results = [_one_seed(s) for s in range(a, b)]
        if args.chunk is not None:
            json.dump(results, open(args.out, "w")); print("saved", args.out); return
    else:
        results = []
        for p in args.aggregate:
            results += json.load(open(p))
    res = {k: dict(absrel=[], fail=[], sc=[]) for k in ESTIMATORS}
    bytype = {k: [] for k in ESTIMATORS}
    sens_all, keep, timing = [], {}, []
    for seed, out, sens, meta in results:
        for name, r in out.items():
            for m in ("absrel", "fail", "sc"):
                res[name][m].append(r[m])
            bytype[name].append(r["by_type"])
        timing.append(out["cgssf"]["sec_per_frame"])
        sens_all.append(sens)
        if seed == 0:
            keep = {n: dict(series=r["series"], ratio=r["ratio"], status=r["status"]) for n, r in out.items()}
            keep["_types"], keep["_s"] = meta
    summary = {k: {m: (float(np.mean(v[m])), float(np.std(v[m]))) for m in v} for k, v in res.items()}
    sens = {g: {str(p): float(np.mean([s[g][p] if p in s[g] else s[g][str(p)] for s in sens_all])) for p in sens_all[0][g]} for g in sens_all[0]}
    cats = ("tex", "low", "planar", "cluster", "event")
    bt = {k: {t: float(np.nanmean([d[t] for d in v])) for t in cats} for k, v in bytype.items()}
    out = dict(n_runs=len(results), summary=summary, by_type=bt, eta=(ETA_S, ETA_B), sigma=(SIG_S, SIG_B),
               sensitivity=sens, seed0=keep,
               cgssf_seconds_per_frame_incl_eval=float(np.mean(timing)))
    json.dump(out, open(args.out, "w"))
    for k, v in summary.items():
        print(f"{k:18s} AbsRel {100*v['absrel'][0]:6.2f}±{100*v['absrel'][1]:.2f}  "
              f"Fail {100*v['fail'][0]:6.2f}±{100*v['fail'][1]:.2f}  SC {100*v['sc'][0]:6.2f}±{100*v['sc'][1]:.2f}")
    for k, v in bt.items():
        print(f"{k:18s} " + "  ".join(f"{t}:{100*x:6.2f}" for t, x in v.items()))
    print("sensitivity", json.dumps(sens))
    print("sec/frame (cgssf incl. eval)", np.mean(timing))


if __name__ == "__main__":
    main()
