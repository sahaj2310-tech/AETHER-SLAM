"""Whole-system map serialisation.

Bundles the scene graph (already JSON), the sparse map (keyframes + points),
the camera trajectory, and the dense mesh (vertices + triangles only; we
don't try to round-trip the TSDF volume since Open3D's PLY export is the
canonical persistent form anyway).

Layout under `backend/data/session/`:

    session.json          # metadata + scene_graph + sparse map + trajectory
    mesh.ply              # Open3D triangle mesh (binary PLY), optional

Loading is best-effort: any missing/corrupt piece is skipped with a warning.
Reloads are intentionally read-only: they hydrate the dashboard with a
prior session's state, but the SLAM tracker still bootstraps from scratch
on the next webcam frame.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger("Persistence")


def session_dir(config) -> str:
    d = os.path.join(config.DATA_DIR, "session")
    os.makedirs(d, exist_ok=True)
    return d


def _ply_path(config) -> str:
    return os.path.join(session_dir(config), "mesh.ply")


def _json_path(config) -> str:
    return os.path.join(session_dir(config), "session.json")


# ---------------------------------------------------------------------------

def save_session(config, semantic_map=None, mapper=None,
                 dense_fusion=None) -> Optional[str]:
    """Snapshot the whole map state to disk. Returns the JSON path or None."""
    try:
        sd = session_dir(config)
    except Exception as e:
        logger.warning(f"session dir failed: {e}")
        return None

    payload: Dict = {
        "saved_at": time.time(),
        "schema_version": 1,
        "scene_graph": {"nodes": [], "edges": []},
        "objects": [],
        "trajectory": [],
        "keyframes": [],
        "loop_events": [],
        "mesh_stats": {},
    }

    if semantic_map is not None:
        try:
            payload["scene_graph"] = semantic_map.get_scene_graph()
            payload["objects"] = semantic_map.get_semantic_objects()
        except Exception as e:
            logger.warning(f"semantic snapshot failed: {e}")

    if mapper is not None:
        try:
            payload["trajectory"] = [kf.T_WC[:3, 3].tolist() for kf in mapper.keyframes]
            payload["keyframes"] = [
                {
                    "id": int(kf.id),
                    "T_WC": kf.T_WC.tolist(),
                    "n_points": int(kf.pts_3d.shape[0]) if kf.pts_3d is not None else 0,
                }
                for kf in mapper.keyframes
            ]
            payload["loop_events"] = list(mapper.loop_events)
            payload["map_points"] = {
                "n": len(mapper.point_by_id),
                "voxel_size_m": float(mapper.config.VOXEL_SIZE),
            }
        except Exception as e:
            logger.warning(f"mapper snapshot failed: {e}")

    if dense_fusion is not None and dense_fusion.enabled:
        try:
            snap = dense_fusion.latest_mesh()
            payload["mesh_stats"] = {
                "n_vertices": int(snap.n_vertices),
                "n_triangles": int(snap.n_triangles),
                "extracted_at": float(snap.extracted_at),
                "voxel_m": dense_fusion.voxel_length,
                "trunc_m": dense_fusion.sdf_trunc,
            }
            # Dump the actual mesh to PLY via Open3D
            if snap.n_vertices > 0:
                import open3d as o3d
                mesh = o3d.geometry.TriangleMesh()
                verts = np.array(snap.vertices, dtype=np.float32).reshape(-1, 3)
                tris = np.array(snap.triangles, dtype=np.int32).reshape(-1, 3)
                mesh.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
                mesh.triangles = o3d.utility.Vector3iVector(tris)
                if snap.colors and len(snap.colors) == len(snap.vertices):
                    cols = np.array(snap.colors, dtype=np.float32).reshape(-1, 3)
                    mesh.vertex_colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
                mesh.compute_vertex_normals()
                ply = _ply_path(config)
                o3d.io.write_triangle_mesh(ply, mesh)
                logger.info(f"Mesh saved → {ply}")
        except Exception as e:
            logger.warning(f"mesh save failed: {e}")

    jpath = _json_path(config)
    try:
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Session saved → {jpath} "
                    f"({len(payload['objects'])} objects, "
                    f"{len(payload['keyframes'])} keyframes)")
        return jpath
    except Exception as e:
        logger.error(f"session JSON write failed: {e}")
        return None


def load_session(config) -> Optional[Dict]:
    """Read back the last session.json (best-effort)."""
    jpath = _json_path(config)
    if not os.path.exists(jpath):
        return None
    try:
        with open(jpath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"session load failed: {e}")
        return None
