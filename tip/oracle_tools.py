"""Ground-truth providers and per-frame diagnostics for the evaluation runner.

* `gt_pose_provider(seq)`   camera-to-world ground-truth pose of every frame index: TUM motion
  capture interpolated to the frame timestamp (linear position, spherical linear rotation;
  frames outside the capture or across gaps > 50 ms get none), Replica `traj.txt`.
* `GTDepth(seq, mesh)`      metric ground-truth depth of a frame index: TUM registered Kinect
  depth (associated within 20 ms, undistorted like the RGB image), Replica depth rendered from
  the ground-truth mesh with the ground-truth pose (Open3D ray casting), cached as float16.
* `frame_depth_errors`      AbsRel, delta_1 and log-median ratio of one depth map.
* `gate_diagnostics`        per-frame AbsRel of the raw per-frame fit, of the estimator's output
  and of the per-frame best affine fit to the ground truth, by alignment status.
* `unc_frame_samples`, `sparsification`   quality of the per-pixel depth uncertainty: sparsification
  curves (error of the pixels kept after removing the most uncertain ones), AUSE / AURG, coverage.
"""
import numpy as np

from eval_common import associate, read_tum_traj


def gt_pose_provider(seq):
    if seq.kind == "replica":
        return {i: T.astype(np.float64) for i, T in enumerate(seq.gt_poses())}
    from scipy.spatial.transform import Rotation, Slerp
    t, p, q = read_tum_traj(seq.gt_trajectory_path())
    order = np.argsort(t, kind="stable")
    t, p, q = t[order], p[order], q[order]
    first = np.r_[True, np.diff(t) > 0]        # repeated timestamps (fr2/xyz: 8 of 36820): keep the first
    t, p, q = t[first], p[first], q[first]
    slerp = Slerp(t, Rotation.from_quat(q))
    out = {}
    for i, ti in enumerate(seq.timestamps):
        j = int(np.searchsorted(t, ti))
        if j == 0 or j >= len(t) or t[j] - t[j - 1] > 0.05:
            continue
        a = (ti - t[j - 1]) / (t[j] - t[j - 1])
        T = np.eye(4)
        T[:3, :3] = slerp([ti]).as_matrix()[0]
        T[:3, 3] = (1 - a) * p[j - 1] + a * p[j]
        out[i] = T
    return out


class GTDepth:
    """Metric ground-truth depth per frame index (None if unavailable)."""

    def __init__(self, seq, mesh_path=None, poses=None):
        self.seq = seq
        self.cache = {}
        if seq.kind == "tum":
            dt, df, _ = seq.gt_depth_index()
            ia, ib = associate(np.asarray(seq.timestamps), dt, 0.02)
            self.files = {int(i): df[j] for i, j in zip(ia, ib)}
        else:
            import open3d as o3d
            mesh = o3d.io.read_triangle_mesh(mesh_path)
            self.scene = o3d.t.geometry.RaycastingScene()
            self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
            self.poses = poses if poses is not None else gt_pose_provider(seq)

    def __call__(self, i):
        if i in self.cache:
            return self.cache[i].astype(np.float32)
        if self.seq.kind == "tum":
            f = self.files.get(i)
            return None if f is None else self.seq.gt_depth(f)
        import open3d as o3d
        T = self.poses.get(i)
        if T is None:
            return None
        W, H = self.seq.size
        rays = self.scene.create_rays_pinhole(intrinsic_matrix=o3d.core.Tensor(self.seq.K),
                                              extrinsic_matrix=o3d.core.Tensor(np.linalg.inv(T)),
                                              width_px=W, height_px=H)
        th = self.scene.cast_rays(rays)["t_hit"].numpy()
        z = th * (rays.numpy()[..., 3:6] @ T[:3, 2])
        z = np.where(np.isfinite(z) & (z > 0), z, 0.0).astype(np.float32)
        self.cache[i] = z.astype(np.float16)
        return self.cache[i].astype(np.float32)     # same values on every call, whatever the variant order


def frame_depth_errors(zh, zg, zmin=0.1, zmax=10.0):
    """(absrel, delta1, log median(zg / zh)) over pixels with valid ground truth; None if < 100."""
    m = (zg > zmin) & (zg < zmax) & (zh > 0) & np.isfinite(zh)
    if m.sum() < 100:
        return None
    r = zh[m] / zg[m]
    return (float(np.mean(np.abs(zh[m] - zg[m]) / zg[m])), float(np.mean(np.maximum(r, 1 / r) < 1.25)),
            float(np.log(np.median(zg[m] / zh[m]))))


def gate_diagnostics(rows, nu_of, gt_of, scale_of, cfg, kappa_min, zmin=0.1, zmax=10.0, max_px=20000, seed=0):
    """rows: per-frame dicts (i, gen, status, kappa, d2, theta_fit, theta_out, med).
    nu_of(i): raw network output (numpy H x W) or None; gt_of(i): metric depth or None;
    scale_of(gen): map-gauge -> metres scale or None. Returns (per-frame records, summary)."""
    from backend.semantic.cg_ssf import robust_affine_fit
    rng = np.random.default_rng(seed)
    recs = []
    for r in rows:
        if r.get("theta_out") is None or r.get("med") is None:
            continue
        s_g = scale_of(r["gen"])
        g = gt_of(r["i"])
        nu = nu_of(r["i"])
        if s_g is None or g is None or nu is None:
            continue
        m = np.flatnonzero(((g > zmin) & (g < zmax) & np.isfinite(g)).reshape(-1))
        if m.size < 200:
            continue
        if m.size > max_px:
            m = rng.choice(m, max_px, replace=False)
        x = nu.reshape(-1)[m].astype(np.float64) / r["med"]
        zg = g.reshape(-1)[m].astype(np.float64)

        def absrel(th):
            rho = th[0] * x + th[1]
            z = np.where(rho > 0, 1.0 / np.maximum(rho, 1e-6), zmax)
            z = np.clip(z, zmin, zmax) * s_g
            return float(np.mean(np.abs(z - zg) / zg))
        fit = robust_affine_fit(x, s_g / zg, cfg, init=None, rng=rng)       # best affine fit in the gauge
        rec = dict(i=r["i"], status=r["status"], kappa=r.get("kappa"), d2=r.get("d2"),
                   absrel_out=absrel(r["theta_out"]),
                   absrel_fit=(absrel(r["theta_fit"]) if r.get("theta_fit") is not None else float("nan")),
                   absrel_best=absrel(fit["theta"]))
        recs.append(rec)

    def summ(sel):
        if not sel:
            return dict(n=0)
        a_out = np.array([q["absrel_out"] for q in sel]); a_fit = np.array([q["absrel_fit"] for q in sel])
        ok = np.isfinite(a_fit)
        return dict(n=len(sel), absrel_out=float(a_out.mean()), absrel_fit=float(np.nanmean(a_fit)) if ok.any() else None,
                    absrel_best=float(np.mean([q["absrel_best"] for q in sel])),
                    frac_fit_worse=float(np.mean(a_fit[ok] > a_out[ok])) if ok.any() else None)
    full_k = lambda q: q.get("kappa") is not None and np.isfinite(q["kappa"]) and q["kappa"] >= kappa_min
    d2s = np.array([q["d2"] for q in recs if q.get("d2") is not None and np.isfinite(q["d2"]) and full_k(q)])
    summary = dict(all=summ(recs),
                   accepted_full=summ([q for q in recs if q["status"] == "full"]),
                   rejected_full=summ([q for q in recs if q["status"] == "rejected" and full_k(q)]),
                   rejected_rank1=summ([q for q in recs if q["status"] == "rejected" and not full_k(q)]),
                   rank1=summ([q for q in recs if q["status"] == "partial"]),
                   nis_full_median=float(np.median(d2s)) if d2s.size else None,
                   nis_full_n=int(d2s.size))
    return recs, summary


def unc_frame_samples(zh, z, sig, sigma2, zg, rng, max_px=2000, zmin=0.1, zmax=10.0):
    """Pixel samples of one frame for the uncertainty evaluation, over the pixels of
    `frame_depth_errors`: evaluated depth zh, ground truth zg and three uncertainties of the same
    depth: sigma_z of the estimator (alignment covariance and residual variance), the residual term
    alone (z^2 sigma_hat) and the depth z alone. Non-finite uncertainties (rho <= 0) become +inf.
    Returns a float32 array (n, 5) or None."""
    m = np.flatnonzero(((zg > zmin) & (zg < zmax) & (zh > 0) & np.isfinite(zh)).reshape(-1))
    if m.size < 100:
        return None
    if m.size > max_px:
        m = rng.choice(m, max_px, replace=False)
    zr = z.reshape(-1)[m].astype(np.float64)
    with np.errstate(invalid="ignore", over="ignore"):
        out = np.stack([zh.reshape(-1)[m], zg.reshape(-1)[m], sig.reshape(-1)[m],
                        zr * zr * np.sqrt(max(float(sigma2), 0.0)), zr], 1)
    out[~np.isfinite(out)] = np.inf
    return out.astype(np.float32)


def _trapz(y, x):
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def sparsification(frames, n_bins=50, seed=0):
    """Sparsification analysis of the per-pixel uncertainties in `frames` (list of
    `unc_frame_samples` arrays). For AbsRel and RMSE: the error of the pixels kept after removing
    the fraction f = 0, 1/n_bins, ... of the most uncertain ones, for each uncertainty and for the
    oracle ranking by the true error; AUSE is the area between a curve and the oracle curve, AURG the
    area between the error of random removal (the full-set error) and the curve. 'pooled' ranks the
    pixels of all frames together (as fusion combines them); 'per_frame' averages the frame-wise
    areas. Also the coverage of |zh - zg| <= k sigma and the Spearman correlation of a per-frame
    uncertainty (median relative uncertainty) with the frame's AbsRel."""
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    fr = np.arange(n_bins) / n_bins
    names = ("sigma", "residual", "depth")

    def curve(u, e, tb):
        order = np.lexsort((tb, -u))                 # most uncertain first, random tie-break
        tail = np.cumsum(e[order][::-1])[::-1]
        n = len(e)
        k = np.floor(fr * n).astype(int)
        return tail[k] / (n - k)

    def analyse(S, tb):
        zh, zg = S[:, 0].astype(np.float64), S[:, 1].astype(np.float64)
        errs = {"absrel": (np.abs(zh - zg) / zg, lambda c: c), "rmse": ((zh - zg) ** 2, np.sqrt)}
        out = {}
        for metric, (e, post) in errs.items():
            cur = {k: post(curve(S[:, 2 + j].astype(np.float64), e, tb)) for j, k in enumerate(names)}
            cur["oracle"] = post(curve(e, e, tb))
            full = float(post(e.mean()))
            out[metric] = dict(full=full, curve=cur,
                               ause={k: _trapz(cur[k] - cur["oracle"], fr) for k in names},
                               aurg={k: _trapz(full - cur[k], fr) for k in names})
        return out

    S = np.concatenate(frames)
    pooled = analyse(S, rng.random(len(S)))
    per = [analyse(f, rng.random(len(f))) for f in frames]
    per_frame = {m: {a: {k: float(np.mean([p[m][a][k] for p in per])) for k in names} for a in ("ause", "aurg")}
                 for m in ("absrel", "rmse")}
    zh, zg = S[:, 0].astype(np.float64), S[:, 1].astype(np.float64)
    ae = np.abs(zh - zg)
    coverage = {}
    for j, k in ((2, "sigma"), (3, "residual")):
        u = S[:, j].astype(np.float64)
        ok = np.isfinite(u) & (u > 0)
        coverage[k] = {"n": int(ok.sum()), "k1": float(np.mean(ae[ok] <= u[ok])),
                       "k2": float(np.mean(ae[ok] <= 2 * u[ok])), "median_ratio": float(np.median(ae[ok] / u[ok]))}
    f_err = np.array([np.mean(np.abs(f[:, 0] - f[:, 1]) / f[:, 1]) for f in frames])
    f_unc = {k: np.array([np.median(f[:, 2 + j].astype(np.float64) / (f[:, 0] if k != "depth" else 1.0))
                          for f in frames]) for j, k in enumerate(names)}
    within = [spearmanr(f[:, 2], f[:, 4])[0] for f in frames if np.isfinite(f[:, 2]).sum() > 10]
    return dict(n_px=int(len(S)), n_frames=len(frames), fractions=fr.tolist(),
                pooled={m: {**{a: v[a] for a in ("full", "ause", "aurg")},
                            "curve": {k: c.tolist() for k, c in v["curve"].items()}} for m, v in pooled.items()},
                per_frame=per_frame, coverage=coverage,
                frame_spearman={k: float(spearmanr(f_unc[k], f_err)[0]) for k in names},
                within_frame_spearman_sigma_depth=float(np.nanmedian(within)) if within else None)
