"""Shared evaluation utilities: TUM-format trajectories, timestamp association, Sim(3) Umeyama alignment."""
import numpy as np


def read_tum_traj(path):
    """TUM format: timestamp tx ty tz qx qy qz qw  ->  (t[N], p[N,3], q[N,4])"""
    a = np.loadtxt(path, comments="#")
    return a[:, 0], a[:, 1:4], a[:, 4:8]


def associate(t_a, t_b, max_dt=0.02):
    """Greedy nearest-timestamp association; returns index arrays (ia, ib)."""
    j = np.clip(np.searchsorted(t_b, t_a), 1, len(t_b) - 1)
    j = np.where(np.abs(t_b[j - 1] - t_a) < np.abs(t_b[j] - t_a), j - 1, j)
    ok = np.abs(t_b[j] - t_a) <= max_dt
    ia, ib = np.flatnonzero(ok), j[ok]
    _, first = np.unique(ib, return_index=True)          # one-to-one
    return ia[first], ib[first]


def umeyama_sim3(src, dst):
    """Least-squares similarity (s, R, t) with dst ~= s R src + t  [Umeyama 1991]."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (xs ** 2).sum() / len(src)
    s = np.trace(np.diag(D) @ S) / var_s
    t = mu_d - s * R @ mu_s
    return s, R, t


def ate_rmse(p_est, p_gt):
    """ATE RMSE after Sim(3) alignment; returns (rmse, s, R, t)."""
    s, R, t = umeyama_sim3(p_est, p_gt)
    aligned = (s * (R @ p_est.T)).T + t
    return float(np.sqrt(np.mean(np.sum((aligned - p_gt) ** 2, 1)))), s, R, t
