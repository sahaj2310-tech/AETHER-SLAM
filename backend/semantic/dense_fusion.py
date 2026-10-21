"""Dense fusion -> triangle mesh (live wrapper around the GPU TSDF).

Per-frame aligned depth + tracked pose are integrated into a sparse voxel-block
TSDF on the GPU (`tsdf_gpu.WeightedTSDF`) with the per-pixel weights of the
alignment (Eq. 12 of the paper: inverse-variance weights from CG-SSF; unit
weights for the ablation variants). Marching cubes runs on the CPU (Open3D) on
a cooldown, off the perception hot path, and the mesh is decimated and packed
into a JSON-friendly snapshot for the WebSocket broadcaster.

Threading: the fusion stage calls `integrate` / `extract_mesh`; the mesh
broadcaster reads `latest_mesh()`. Shared state is guarded by an RLock.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger("DenseFusion")

try:
    import open3d as o3d  # noqa: F401
    import torch
    from backend.semantic.tsdf_gpu import WeightedTSDF
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    logger.warning("open3d/torch not available; dense fusion disabled.")


@dataclass
class MeshSnapshot:
    """JSON-friendly mesh payload for the WebSocket broadcaster."""
    vertices: list = field(default_factory=list)   # flat [x,y,z,x,y,z,...]
    colors: list = field(default_factory=list)     # flat [r,g,b,r,g,b,...] 0-1
    triangles: list = field(default_factory=list)  # flat [i,j,k,i,j,k,...]
    n_vertices: int = 0
    n_triangles: int = 0
    extracted_at: float = 0.0
    # Auto-estimated room dimensions (map units) + floor area of the scanned
    # footprint. {} until enough surface is captured.
    dimensions: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "vertices": self.vertices,
            "colors": self.colors,
            "triangles": self.triangles,
            "n_vertices": int(self.n_vertices),
            "n_triangles": int(self.n_triangles),
            "extracted_at": float(self.extracted_at),
            "dimensions": self.dimensions,
        }


def _estimate_room_dimensions(vertices: np.ndarray) -> dict:
    """Width/height/depth (2nd-98th percentile extents per axis) and floor area
    (convex hull of the x-z projection). Monocular: map units, not metres."""
    if vertices is None or vertices.shape[0] < 50:
        return {}
    xs, ys, zs = vertices[:, 0], vertices[:, 1], vertices[:, 2]

    def extent(a):
        lo, hi = np.percentile(a, 2), np.percentile(a, 98)
        return float(max(0.0, hi - lo))

    width, height, depth = extent(xs), extent(ys), extent(zs)
    floor_area = round(width * depth, 2)  # fallback: bounding footprint
    try:
        from scipy.spatial import ConvexHull
        pts = np.stack([xs, zs], axis=1).astype(np.float64)
        if pts.shape[0] > 4000:
            idx = np.random.default_rng(0).choice(pts.shape[0], 4000, replace=False)
            pts = pts[idx]
        floor_area = round(float(ConvexHull(pts).volume), 2)  # 2D hull .volume = area
    except Exception:
        pass
    return {"width_m": round(width, 2), "height_m": round(height, 2), "depth_m": round(depth, 2),
            "floor_area_m2": floor_area}


class DenseFusion:
    """Thread-safe live wrapper of the GPU TSDF."""

    def __init__(self, config, *, voxel_length: Optional[float] = None, sdf_trunc: Optional[float] = None,
                 mesh_max_vertices: Optional[int] = None):
        self.config = config
        self.enabled = OPEN3D_AVAILABLE
        self.voxel_length = float(voxel_length or getattr(config, "TSDF_VOXEL_M", 0.03))
        self.sdf_trunc = float(sdf_trunc or getattr(config, "TSDF_TRUNC_M", 0.09))
        self.max_depth = float(getattr(config, "TSDF_MAX_DEPTH_M", 4.0))
        self.mesh_max_vertices = int(mesh_max_vertices if mesh_max_vertices is not None
                                     else getattr(config, "TSDF_MESH_MAX_VERTICES", 28000))
        self._lock = threading.RLock()
        self._latest_mesh = MeshSnapshot()
        self._last_extract_ts = 0.0
        self._integrations_since_extract = 0
        self.last_extract_ms = 0.0
        self._tsdf = (WeightedTSDF(self.voxel_length, self.sdf_trunc, max_depth=self.max_depth)
                      if self.enabled else None)
        if self.enabled:
            logger.info(f"Dense fusion ready (GPU TSDF, voxel={self.voxel_length}, trunc={self.sdf_trunc}, "
                        f"max_depth={self.max_depth}, device={self._tsdf.device}).")

    def reset(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._tsdf.reset()
            self._latest_mesh = MeshSnapshot()
            self._integrations_since_extract = 0
            logger.info("Dense fusion volume reset.")

    def integrate(self, color_bgr: np.ndarray, depth, weight, T_WC: np.ndarray) -> bool:
        """Fold a frame into the TSDF. depth / weight: torch tensors (or numpy) of the
        frame size; depth 0 or weight 0 = pixel not fused."""
        if not self.enabled or color_bgr is None or depth is None or T_WC is None:
            return False
        if not np.all(np.isfinite(T_WC)):
            return False
        dev = self._tsdf.device
        d = depth if isinstance(depth, torch.Tensor) else torch.from_numpy(np.asarray(depth, np.float32))
        w = (weight if isinstance(weight, torch.Tensor) else
             torch.from_numpy(np.asarray(weight, np.float32)) if weight is not None else (d > 0).float())
        rgb = torch.from_numpy(np.ascontiguousarray(color_bgr[:, :, ::-1])).to(dev)
        with self._lock:
            ok = self._tsdf.integrate(d.to(dev), w.to(dev), rgb, self.config.INTRINSICS_MATRIX, T_WC)
            if ok:
                self._integrations_since_extract += 1
            return ok

    def extract_mesh(self, force: bool = False, cooldown_s: float = 3.0) -> Optional[MeshSnapshot]:
        if not self.enabled:
            return None
        now = time.time()
        with self._lock:
            if (not force and (now - self._last_extract_ts) < cooldown_s
                    and self._integrations_since_extract < 5):
                return None
            self._last_extract_ts = now
            self._integrations_since_extract = 0
        t0 = time.perf_counter()
        try:
            mesh = self._tsdf.extract_mesh()
        except Exception as e:
            logger.error(f"Mesh extraction failed: {e}")
            return None
        if mesh is None or len(mesh.vertices) == 0:
            with self._lock:
                self._latest_mesh = MeshSnapshot(extracted_at=now)
            return self._latest_mesh
        mesh.compute_vertex_normals()
        if len(mesh.vertices) > self.mesh_max_vertices:
            try:
                mesh = mesh.simplify_quadric_decimation(int(self.mesh_max_vertices * 2))
            except Exception as e:
                logger.debug(f"Decimation failed: {e}")
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        triangles = np.asarray(mesh.triangles, dtype=np.int32)
        colors = (np.asarray(mesh.vertex_colors, dtype=np.float32) if len(mesh.vertex_colors) > 0
                  else np.tile([0.6, 0.6, 0.7], (len(vertices), 1)).astype(np.float32))
        dims = _estimate_room_dimensions(vertices)
        snap = MeshSnapshot(vertices=np.round(vertices, 4).flatten().tolist(),
                            colors=np.round(colors, 3).flatten().tolist(),
                            triangles=triangles.flatten().tolist(), n_vertices=int(vertices.shape[0]),
                            n_triangles=int(triangles.shape[0]), extracted_at=now, dimensions=dims)
        self.last_extract_ms = 1e3 * (time.perf_counter() - t0)
        with self._lock:
            self._latest_mesh = snap
        logger.info(f"Mesh: {snap.n_vertices} v / {snap.n_triangles} f ({self.last_extract_ms:.0f} ms)")
        return snap

    def full_mesh(self):
        """Undecimated mesh (legacy Open3D TriangleMesh) for export."""
        if not self.enabled:
            return None
        mesh = self._tsdf.extract_mesh()
        if mesh is not None and len(mesh.vertices):
            mesh.compute_vertex_normals()
        return mesh

    def latest_mesh(self) -> MeshSnapshot:
        with self._lock:
            return self._latest_mesh

    def stats(self) -> dict:
        s = {"enabled": self.enabled, "voxel_m": self.voxel_length, "trunc_m": self.sdf_trunc,
             "max_depth_m": self.max_depth, "last_extract_ts": self._last_extract_ts,
             "last_extract_ms": round(self.last_extract_ms, 1),
             "n_vertices": self._latest_mesh.n_vertices, "n_triangles": self._latest_mesh.n_triangles}
        if self.enabled:
            s.update(self._tsdf.stats())
        return s
