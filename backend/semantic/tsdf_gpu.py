"""Sparse voxel-block TSDF on the GPU with per-pixel integration weights.

Each fused frame contributes, for every voxel within the truncation band of a
valid pixel, the truncated signed distance sdf = z(u) - Z_voxel (projective
distance along the optical axis, as in KinectFusion / Open3D) with the pixel's
weight omega(u) (Eq. 12; omega = 1 for uniform fusion):

    D <- (W D + omega * clamp(sdf / mu, -1, 1)) / (W + omega),   W <- W + omega.

Voxels are stored in blocks of R^3 (R = 8) allocated on demand and addressed
through a sorted key table (torch.searchsorted), so memory grows with the
observed surface only.  The in-block voxel order matches Open3D's
VoxelBlockGrid ((z * R + y) * R + x at position (block * R + local) * delta),
so marching cubes is delegated to Open3D on the CPU without re-ordering.
"""
from __future__ import annotations

import logging
import threading

import numpy as np
import torch

logger = logging.getLogger("WeightedTSDF")

_OFF = 1 << 20          # block-coordinate offset for key packing (21 bits per axis)


def _pack(b: torch.Tensor) -> torch.Tensor:
    b = b + _OFF
    return (b[:, 0] << 42) | (b[:, 1] << 21) | b[:, 2]


def _unpack(k: torch.Tensor) -> torch.Tensor:
    m = (1 << 21) - 1
    return torch.stack([(k >> 42) & m, (k >> 21) & m, k & m], 1) - _OFF


class WeightedTSDF:
    def __init__(self, voxel: float, trunc: float, block_res: int = 8, device: str = "cuda",
                 capacity: int = 4096, max_depth: float | None = None):
        self.voxel = float(voxel)
        self.trunc = float(trunc)
        self.R = int(block_res)
        self.R3 = self.R ** 3
        self.max_depth = max_depth
        self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        R = self.R
        zz, yy, xx = torch.meshgrid(torch.arange(R), torch.arange(R), torch.arange(R), indexing="ij")
        self._local = torch.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], 1).to(
            self.device, torch.float32)                                   # (R^3, 3), x fastest
        self._lock = threading.RLock()
        self._init_capacity = int(capacity)
        self.reset()

    # ------------------------------------------------------------------ storage
    def reset(self):
        with self._lock:
            dev = self.device
            self.n = 0
            self.cap = 0
            self.keys = torch.empty(0, dtype=torch.int64, device=dev)       # sorted packed keys
            self.key_slot = torch.empty(0, dtype=torch.int64, device=dev)   # slot of each sorted key
            self.blocks = torch.empty((0, 3), dtype=torch.int64, device=dev)
            self.tsdf = torch.empty((0, self.R3), dtype=torch.float32, device=dev)
            self.weight = torch.empty((0, self.R3), dtype=torch.float32, device=dev)
            self.color = torch.empty((0, self.R3, 3), dtype=torch.float32, device=dev)
            self.n_integrated = 0
            self._grow(self._init_capacity)

    def _grow(self, need: int):
        if need <= self.cap:
            return
        new = max(need, 2 * self.cap, 1024)
        dev = self.device

        def ext(t, shape, fill=0.0):
            out = torch.full(shape, fill, dtype=t.dtype, device=dev)
            if self.n:
                out[:self.n] = t[:self.n]
            return out
        self.blocks = ext(self.blocks, (new, 3))
        self.tsdf = ext(self.tsdf, (new, self.R3), 1.0)
        self.weight = ext(self.weight, (new, self.R3))
        self.color = ext(self.color, (new, self.R3, 3))
        self.cap = new

    def _activate(self, bc: torch.Tensor) -> torch.Tensor:
        """Return storage slots for unique block coordinates bc (M, 3), allocating new ones."""
        keys = _pack(bc)
        if self.keys.numel():
            pos = torch.searchsorted(self.keys, keys).clamp(max=self.keys.numel() - 1)
            found = self.keys[pos] == keys
        else:
            pos = torch.zeros_like(keys)
            found = torch.zeros_like(keys, dtype=torch.bool)
        slots = torch.empty_like(keys)
        slots[found] = self.key_slot[pos[found]]
        new_mask = ~found
        m = int(new_mask.sum())
        if m:
            self._grow(self.n + m)
            new_slots = torch.arange(self.n, self.n + m, device=self.device)
            slots[new_mask] = new_slots
            self.blocks[self.n:self.n + m] = bc[new_mask]
            self.n += m
            all_keys = torch.cat([self.keys, keys[new_mask]])
            all_slots = torch.cat([self.key_slot, new_slots])
            order = torch.argsort(all_keys)
            self.keys, self.key_slot = all_keys[order], all_slots[order]
        return slots

    # ------------------------------------------------------------------ integration
    @torch.no_grad()
    def integrate(self, depth: torch.Tensor, weight: torch.Tensor, color_rgb: torch.Tensor | None,
                  K: np.ndarray, T_WC: np.ndarray) -> bool:
        """depth (H, W) map units (0 = invalid); weight (H, W) >= 0; color_rgb (H, W, 3) uint8.
        K: 3x3 intrinsics; T_WC: 4x4 camera-to-world pose."""
        dev = self.device
        depth = depth.to(dev, torch.float32)
        weight = weight.to(dev, torch.float32)
        H, W = depth.shape
        valid = (weight > 0) & (depth > 0) & torch.isfinite(depth)
        if self.max_depth is not None:
            valid &= depth <= self.max_depth
        if int(valid.sum()) < 64:
            return False
        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
        T = torch.as_tensor(np.asarray(T_WC, np.float64), device=dev, dtype=torch.float32)
        Rwc, twc = T[:3, :3], T[:3, 3]
        with self._lock:
            # 1) blocks touched by the truncation band of the valid pixels
            v, u = torch.nonzero(valid, as_tuple=True)
            z = depth[v, u]
            xr = (u.float() - cx) / fx
            yr = (v.float() - cy) / fy
            bsize = self.R * self.voxel
            keys = []
            for dz in (-self.trunc, 0.0, self.trunc):
                zz = (z + dz).clamp(min=1e-4)
                pc = torch.stack([xr * zz, yr * zz, zz], 1)
                pw = pc @ Rwc.T + twc
                keys.append(_pack(torch.floor(pw / bsize).to(torch.int64)))
            bc = _unpack(torch.unique(torch.cat(keys)))
            slots = self._activate(bc)
            # 2) project every voxel of those blocks into the image
            pos = (self.blocks[slots].to(torch.float32)[:, None, :] * self.R + self._local[None]) * self.voxel
            pos = pos.reshape(-1, 3)
            Rcw = Rwc.T
            pc = (pos - twc) @ Rcw.T
            Z = pc[:, 2]
            inf = Z > 1e-4
            Zs = torch.where(inf, Z, torch.ones_like(Z))
            uu = torch.round(pc[:, 0] / Zs * fx + cx).to(torch.int64)
            vv = torch.round(pc[:, 1] / Zs * fy + cy).to(torch.int64)
            ok = inf & (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            idx = torch.nonzero(ok, as_tuple=True)[0]
            uu, vv, Z = uu[idx], vv[idx], Z[idx]
            d = depth[vv, uu]
            w = weight[vv, uu]
            sdf = d - Z
            keep = valid[vv, uu] & (sdf >= -self.trunc)
            idx, uu, vv, sdf, w = idx[keep], uu[keep], vv[keep], sdf[keep], w[keep]
            if idx.numel() == 0:
                return False
            obs = torch.clamp(sdf / self.trunc, max=1.0)
            slot = slots[idx // self.R3]
            loc = idx % self.R3
            W_old = self.weight[slot, loc]
            W_new = W_old + w
            self.tsdf[slot, loc] = (self.tsdf[slot, loc] * W_old + obs * w) / W_new
            if color_rgb is not None:
                col = color_rgb.to(dev)[vv, uu].to(torch.float32)   # 0..255 (Open3D divides by 255)
                self.color[slot, loc] = (self.color[slot, loc] * W_old[:, None] + col * w[:, None]) / W_new[:, None]
            self.weight[slot, loc] = W_new
            self.n_integrated += 1
        return True

    # ------------------------------------------------------------------ extraction
    def extract_mesh(self, weight_threshold: float = 1e-4):
        """Marching cubes (Open3D VoxelBlockGrid on the CPU). Returns a legacy TriangleMesh or None."""
        import open3d as o3d
        import open3d.core as o3c
        with self._lock:
            n = self.n
            if n == 0:
                return None
            blocks = self.blocks[:n].to(torch.int32).cpu().numpy()
            tsdf = self.tsdf[:n].cpu().numpy()
            weight = self.weight[:n].cpu().numpy()
            color = self.color[:n].cpu().numpy()
        cpu = o3c.Device("CPU:0")
        vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=("tsdf", "weight", "color"),
            attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
            attr_channels=((1), (1), (3)),
            voxel_size=self.voxel, block_resolution=self.R,
            block_count=int(n * 1.25) + 64, device=cpu)
        keys = o3c.Tensor(blocks, device=cpu)
        vbg.hashmap().activate(keys)
        buf, found = vbg.hashmap().find(keys)
        bi = o3c.Tensor(buf.numpy().astype(np.int64), device=cpu)
        R = self.R
        vbg.attribute("tsdf")[bi] = o3c.Tensor(tsdf.reshape(n, R, R, R, 1), device=cpu)
        vbg.attribute("weight")[bi] = o3c.Tensor(weight.reshape(n, R, R, R, 1), device=cpu)
        vbg.attribute("color")[bi] = o3c.Tensor(color.reshape(n, R, R, R, 3), device=cpu)
        mesh = vbg.extract_triangle_mesh(weight_threshold=weight_threshold).to_legacy()
        if len(mesh.vertices) == 0:
            return mesh
        if mesh.has_vertex_colors():
            mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(np.asarray(mesh.vertex_colors), 0.0, 1.0))
        return mesh

    def memory_mb(self) -> float:
        per_block = self.R3 * (4 + 4 + 12) + 8 * 3
        return self.cap * per_block / 2 ** 20

    def stats(self) -> dict:
        return {"blocks": int(self.n), "capacity_blocks": int(self.cap),
                "memory_mb": round(self.memory_mb(), 1), "integrated": int(self.n_integrated)}
