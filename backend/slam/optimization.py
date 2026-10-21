import torch
import numpy as np
import logging
from backend.config import Config

logger = logging.getLogger("SLAMOptimizer")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"SLAM Optimizer running on: {device}")


def quaternion_to_rotation_matrix(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    norm = torch.sqrt(w * w + x * x + y * y + z * z + 1e-9)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - w * z)
    r02 = 2 * (x * z + w * y)
    r10 = 2 * (x * y + w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - w * x)
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x * x + y * y)

    return torch.stack([
        torch.stack([r00, r01, r02], dim=1),
        torch.stack([r10, r11, r12], dim=1),
        torch.stack([r20, r21, r22], dim=1)
    ], dim=1)


def rotation_matrix_to_quaternion(R):
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S

    q = np.array([qw, qx, qy, qz], dtype=np.float32)
    return q / np.linalg.norm(q)


def orthonormalize_rotation(R: np.ndarray) -> np.ndarray:
    """SVD projection back onto SO(3); guards against quaternion-induced drift."""
    U, _, Vt = np.linalg.svd(R)
    R_o = U @ Vt
    if np.linalg.det(R_o) < 0:
        U[:, -1] *= -1
        R_o = U @ Vt
    return R_o.astype(np.float32)


class LocalBundleAdjustment(torch.nn.Module):
    """Anchors the first keyframe by keeping its parameters out of the optimizer."""
    def __init__(self, anchor_t, anchor_q, init_poses_t, init_poses_q, init_pts_3d):
        super().__init__()
        # Fixed anchor (not a Parameter)
        self.register_buffer("anchor_t", torch.from_numpy(anchor_t).float())
        self.register_buffer("anchor_q", torch.from_numpy(anchor_q).float())

        # Free poses (everything after the anchor)
        self.poses_t = torch.nn.Parameter(torch.from_numpy(init_poses_t).float())
        self.poses_q = torch.nn.Parameter(torch.from_numpy(init_poses_q).float())
        self.pts_3d = torch.nn.Parameter(torch.from_numpy(init_pts_3d).float())

    def all_poses(self):
        t = torch.cat([self.anchor_t.unsqueeze(0), self.poses_t], dim=0)
        q = torch.cat([self.anchor_q.unsqueeze(0), self.poses_q], dim=0)
        return t, q

    def forward(self, kf_indices, pt_indices, K):
        t_all, q_all = self.all_poses()
        t = t_all[kf_indices]
        q = q_all[kf_indices]
        pts = self.pts_3d[pt_indices]

        R = quaternion_to_rotation_matrix(q)
        R_CW = R.transpose(1, 2)
        t_CW = -torch.bmm(R_CW, t.unsqueeze(2)).squeeze(2)

        P_cam = torch.bmm(R_CW, pts.unsqueeze(2)).squeeze(2) + t_CW
        z = P_cam[:, 2].clamp(min=1e-3)  # avoid div by ~0

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        u_proj = fx * (P_cam[:, 0] / z) + cx
        v_proj = fy * (P_cam[:, 1] / z) + cy
        return torch.stack([u_proj, v_proj], dim=1)


def run_local_bundle_adjustment(mapper, tracker, window_size=10, iterations=15):
    if len(mapper.keyframes) < 3:
        return

    keyframes = mapper.keyframes[-window_size:]
    num_kf = len(keyframes)

    # Build observations using stable point IDs
    pid_to_local = {}
    unique_pts_3d = []
    kf_indices = []
    pt_indices = []
    observed_2d = []
    pid_order = []

    for kf_idx, kf in enumerate(keyframes):
        for i in range(len(kf.pts_3d)):
            pid = int(kf.pt_ids[i])
            if pid not in pid_to_local:
                pid_to_local[pid] = len(unique_pts_3d)
                unique_pts_3d.append(kf.pts_3d[i])
                pid_order.append(pid)
            kf_indices.append(kf_idx)
            pt_indices.append(pid_to_local[pid])
            observed_2d.append(kf.pts_2d[i])

    if len(unique_pts_3d) < 10 or len(kf_indices) < 20:
        return

    # Anchor the first keyframe; optimize the rest
    anchor_kf = keyframes[0]
    anchor_t = anchor_kf.T_WC[:3, 3].astype(np.float32)
    anchor_q = rotation_matrix_to_quaternion(anchor_kf.T_WC[:3, :3]).astype(np.float32)

    free_kfs = keyframes[1:]
    init_poses_t = np.array([kf.T_WC[:3, 3] for kf in free_kfs], dtype=np.float32) \
        if free_kfs else np.zeros((0, 3), dtype=np.float32)
    init_poses_q = np.array([rotation_matrix_to_quaternion(kf.T_WC[:3, :3]) for kf in free_kfs],
                            dtype=np.float32) if free_kfs else np.zeros((0, 4), dtype=np.float32)
    init_pts_3d = np.array(unique_pts_3d, dtype=np.float32)

    kf_indices_t = np.array(kf_indices, dtype=np.int64)
    pt_indices_t = np.array(pt_indices, dtype=np.int64)
    observed_2d_t = np.array(observed_2d, dtype=np.float32)

    model = LocalBundleAdjustment(anchor_t, anchor_q, init_poses_t, init_poses_q, init_pts_3d).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    huber_loss = torch.nn.HuberLoss(delta=5.0)

    K_tensor = torch.from_numpy(tracker.K).float().to(device)
    observed_2d_tensor = torch.from_numpy(observed_2d_t).float().to(device)
    kf_indices_tensor = torch.from_numpy(kf_indices_t).to(device)
    pt_indices_tensor = torch.from_numpy(pt_indices_t).to(device)

    last_loss = None
    with torch.enable_grad():            # callers may run under torch.no_grad()
        for _ in range(iterations):
            optimizer.zero_grad()
            projected = model(kf_indices_tensor, pt_indices_tensor, K_tensor)
            loss = huber_loss(projected, observed_2d_tensor)
            if not torch.isfinite(loss):
                logger.warning("BA loss diverged (non-finite). Aborting this pass.")
                return
            loss.backward()
            optimizer.step()
            last_loss = loss.item()

    # Read back optimized vars
    t_all, q_all = model.all_poses()
    opt_t = t_all.detach().cpu().numpy()
    opt_q = q_all.detach().cpu().numpy()
    opt_pts_3d = model.pts_3d.detach().cpu().numpy()

    if not np.all(np.isfinite(opt_t)) or not np.all(np.isfinite(opt_q)) \
            or not np.all(np.isfinite(opt_pts_3d)):
        logger.warning("BA produced non-finite values. Skipping update.")
        return

    # Update keyframe poses
    for kf_idx, kf in enumerate(keyframes):
        q = opt_q[kf_idx]
        t = opt_t[kf_idx]
        qw, qx, qy, qz = q
        norm = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz + 1e-9)
        qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
        R = np.array([
            [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)]
        ], dtype=np.float32)
        R = orthonormalize_rotation(R)
        kf.T_WC[:3, :3] = R
        kf.T_WC[:3, 3] = t

    # Update map points by ID (O(M) instead of O(N*M))
    pid_to_xyz = {}
    for local_idx, pid in enumerate(pid_order):
        mapper.update_point(pid, opt_pts_3d[local_idx])
        pid_to_xyz[pid] = opt_pts_3d[local_idx]
    # ... and in the keyframes and the tracker's working set, so that the next
    # PnP and the depth-alignment anchors use the refined coordinates.
    for kf in keyframes:
        for i, pid in enumerate(kf.pt_ids.tolist()):
            xyz = pid_to_xyz.get(pid)
            if xyz is not None:
                kf.pts_3d[i] = xyz
    tracker.update_points(pid_to_xyz)

    # Sync tracker's pose to latest keyframe's optimized pose
    tracker.T_WC = keyframes[-1].T_WC.copy()

    if last_loss is not None:
        logger.debug(f"BA complete. Loss: {last_loss:.4f}")


# =============================================================================
# Windowed bundle adjustment with a sparse trust-region (Levenberg-Marquardt
# type) solver. Replaces the first-order (Adam) pass above by default.
# =============================================================================

def _rotvec_to_R(rv):
    th = np.linalg.norm(rv, axis=-1, keepdims=True)
    k = rv / np.maximum(th, 1e-12)
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    z = np.zeros_like(kx)
    Kx = np.stack([np.stack([z, -kz, ky], -1), np.stack([kz, z, -kx], -1), np.stack([-ky, kx, z], -1)], -2)
    th = th[..., None]
    I = np.eye(3)
    return I + np.sin(th) * Kx + (1 - np.cos(th)) * (Kx @ Kx)


def _R_to_rotvec(R):
    import cv2
    return cv2.Rodrigues(np.asarray(R, np.float64))[0].ravel()


def run_local_bundle_adjustment_lm(mapper, tracker, window_size=10, max_nfev=15, huber_px=2.0, min_obs=2):
    """Jointly refine the poses of the last `window_size` keyframes (the oldest one
    fixed as gauge anchor) and every map point observed in >= `min_obs` of them,
    minimising Huber-robust reprojection error with scipy's sparse trust-region
    reflective solver (2-point finite-difference Jacobian with the BA sparsity
    pattern). Writes the result back into the keyframes, the mapper and the
    tracker's working set."""
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix

    kfs = mapper.keyframes[-window_size:]
    if len(kfs) < 3:
        return None
    K = np.asarray(tracker.K, np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    # observations
    count = {}
    for kf in kfs:
        for pid in kf.pt_ids.tolist():
            count[pid] = count.get(pid, 0) + 1
    multi = {pid for pid, c in count.items() if c >= min_obs}
    if len(multi) < 10:
        return None
    cur = {int(p): tracker.tracked_pts_3d[i].astype(np.float64)
           for i, p in enumerate(tracker.tracked_pt_ids.tolist()) if p in multi}
    pid_index, X0 = {}, []
    kf_idx, pt_idx, obs = [], [], []
    for j, kf in enumerate(kfs):
        for i, pid in enumerate(kf.pt_ids.tolist()):
            if pid not in multi:
                continue
            if pid not in pid_index:
                pid_index[pid] = len(X0)
                X0.append(cur.get(pid, kf.pts_3d[i].astype(np.float64)))
            kf_idx.append(j); pt_idx.append(pid_index[pid]); obs.append(kf.pts_2d[i])
    kf_idx = np.asarray(kf_idx); pt_idx = np.asarray(pt_idx); obs = np.asarray(obs, np.float64)
    X0 = np.asarray(X0, np.float64)
    nk, npt = len(kfs), X0.shape[0]
    if obs.shape[0] < 20:
        return None
    # pose parameters: camera-from-world (rotation vector, translation)
    T_CW = [np.linalg.inv(kf.T_WC.astype(np.float64)) for kf in kfs]
    p0 = np.concatenate([np.concatenate([_R_to_rotvec(T[:3, :3]), T[:3, 3]]) for T in T_CW[1:]])
    x0 = np.concatenate([p0, X0.ravel()])
    fixed_R, fixed_t = T_CW[0][:3, :3], T_CW[0][:3, 3]

    def unpack(x):
        P = x[:6 * (nk - 1)].reshape(nk - 1, 6)
        R = np.concatenate([fixed_R[None], _rotvec_to_R(P[:, :3])], 0)
        t = np.concatenate([fixed_t[None], P[:, 3:]], 0)
        return R, t, x[6 * (nk - 1):].reshape(npt, 3)

    def residuals(x):
        R, t, X = unpack(x)
        pc = np.einsum("nij,nj->ni", R[kf_idx], X[pt_idx]) + t[kf_idx]
        z = np.where(np.abs(pc[:, 2]) < 1e-6, 1e-6, pc[:, 2])
        return np.concatenate([fx * pc[:, 0] / z + cx - obs[:, 0], fy * pc[:, 1] / z + cy - obs[:, 1]])

    m = obs.shape[0]
    A = lil_matrix((2 * m, x0.size), dtype=np.int8)
    rows = np.arange(m)
    for k in range(6):
        free = kf_idx > 0
        cols = 6 * (kf_idx[free] - 1) + k
        A[rows[free], cols] = 1
        A[m + rows[free], cols] = 1
    for k in range(3):
        cols = 6 * (nk - 1) + 3 * pt_idx + k
        A[rows, cols] = 1
        A[m + rows, cols] = 1
    try:
        res = least_squares(residuals, x0, jac_sparsity=A.tocsr(), method="trf", loss="huber",
                            f_scale=huber_px, x_scale="jac", max_nfev=max_nfev, tr_solver="lsmr")
    except Exception as e:  # numerical failure: keep the current estimate
        logger.warning(f"BA (LM) failed: {e}")
        return None
    if not np.all(np.isfinite(res.x)):
        return None
    R, t, X = unpack(res.x)
    for j, kf in enumerate(kfs):
        if j == 0:
            continue
        Tcw = np.eye(4); Tcw[:3, :3] = R[j]; Tcw[:3, 3] = t[j]
        kf.T_WC = np.linalg.inv(Tcw).astype(np.float32)
    pid_to_xyz = {pid: X[i].astype(np.float32) for pid, i in pid_index.items()}
    for pid, xyz in pid_to_xyz.items():
        mapper.update_point(pid, xyz)
    for kf in kfs:
        for i, pid in enumerate(kf.pt_ids.tolist()):
            xyz = pid_to_xyz.get(pid)
            if xyz is not None:
                kf.pts_3d[i] = xyz
    tracker.update_points(pid_to_xyz)
    tracker.T_WC = kfs[-1].T_WC.copy()
    return dict(cost0=float(0.5 * np.sum(residuals(x0) ** 2)), cost=float(res.cost), nfev=int(res.nfev),
                n_obs=int(m), n_pts=int(npt))
