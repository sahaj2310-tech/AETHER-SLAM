"""Persistent semantic map + scene graph reasoning.

Each visual track (DeepSORT id) becomes a node in a `networkx.Graph` that
persists across sessions via a JSON file on disk.

Node attributes
    id, label, centroid_3d, bbox_3d, confidence, hits,
    first_seen, last_seen, sound_source

Edges encode spatial proximity ("near"), containment ("on"/"inside"
heuristics), and audio association ("emits_sound"), with attributes that
can be queried by the frontend to answer questions like "what's near the
laptop?".

Designed to be cheap: the graph is updated once per frame in O(M) where M
is the number of *current* detections, not O(M^2) over all known objects.

Persistence is opt-in: on shutdown we write the graph to
`backend/data/scene_graph.json`; on startup we load it back so we don't lose
"the user has a coffee mug on the desk" between runs.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Dict, List, Optional

import numpy as np
import networkx as nx

from backend.config import Config

logger = logging.getLogger("SemanticMap")

NEAR_THRESHOLD_M = 1.2          # proximity edge cutoff
SUPPORT_HEIGHT_TOL = 0.3        # for naive "on top of" inference


class SemanticObject:
    """One persistent object tracked by visual track id."""
    def __init__(self, track_id: int, label: str,
                 centroid_3d, bbox_3d, confidence: float):
        self.id = int(track_id)
        self.label = label
        self.centroid_3d = np.array(centroid_3d, dtype=np.float32)
        self.bbox_3d = np.array(bbox_3d, dtype=np.float32)
        self.confidence = float(confidence)
        self.hits = 1
        self.first_seen = time.time()
        self.last_seen = self.first_seen

    def update(self, centroid_3d, bbox_3d, confidence, alpha=0.25):
        nc = np.array(centroid_3d, dtype=np.float32)
        nb = np.array(bbox_3d, dtype=np.float32)
        # Confidence-aware fusion: a high-confidence observation pulls
        # the centroid harder; a low-confidence one barely moves it.
        # Effective weight = alpha * confidence, clamped to keep the
        # filter responsive while still rejecting noise.
        eff_alpha = float(np.clip(alpha * (0.4 + 0.6 * float(confidence)),
                                  0.05, 0.6))
        self.centroid_3d = (1 - eff_alpha) * self.centroid_3d + eff_alpha * nc
        self.bbox_3d = (1 - eff_alpha) * self.bbox_3d + eff_alpha * nb
        # Track *exponential* mean confidence so the registry shows how
        # certain we currently are, not just the historical maximum.
        self.confidence = float(0.85 * self.confidence + 0.15 * float(confidence))
        self.hits += 1
        self.last_seen = time.time()

    def to_dict(self):
        return {
            "id": self.id,
            "label": self.label,
            "centroid_3d": self.centroid_3d.tolist(),
            "bbox_3d": self.bbox_3d.tolist(),
            "confidence": float(self.confidence),
            "hits": int(self.hits),
            "first_seen": float(self.first_seen),
            "last_seen": float(self.last_seen),
        }


class SemanticMap:
    def __init__(self, config: Config):
        self.config = config
        self.objects: Dict[int, SemanticObject] = {}
        self.graph = nx.Graph()
        self.frame_counter = 0
        self.association_threshold = 0.7
        # Guards `objects` and `graph` against the semantic stage's
        # update_map / prune racing with reads from the broadcaster.
        self._lock = threading.RLock()
        self._persist_path = os.path.join(self.config.DATA_DIR, "scene_graph.json")
        self.load_persistent()

    # ------------------------------------------------------------------
    # CRUD + per-frame update
    # ------------------------------------------------------------------
    def update_map(self, detections: List[Dict]):
        """Detections must include a `track_id` field (from DeepSORT)."""
        with self._lock:
            self.frame_counter += 1
            if not detections:
                return

            for det in detections:
                tid = int(det.get("track_id", -1))
                label = det["label"]
                centroid = det["centroid_3d"]
                bbox = det["bbox_3d"]
                conf = det["confidence"]

                if tid in self.objects:
                    self.objects[tid].update(centroid, bbox, conf)
                else:
                    obj = SemanticObject(tid, label, centroid, bbox, conf)
                    self.objects[tid] = obj
                    self.graph.add_node(tid, label=label)
                    logger.info(f"Registered new object: {label} (track={tid})")

                # Always refresh node attributes; they are broadcast verbatim, so
                # keep them as native Python primitives (no numpy types).
                self.graph.nodes[tid].update({
                    "label": label,
                    "centroid_3d": [float(x) for x in self.objects[tid].centroid_3d],
                    "bbox_3d": [float(x) for x in self.objects[tid].bbox_3d],
                    "hits": int(self.objects[tid].hits),
                    "last_seen": float(self.objects[tid].last_seen),
                })

            self._rebuild_edges()

    def _rebuild_edges(self):
        """Recompute proximity / support edges from current node positions."""
        self.graph.clear_edges()

        ids = list(self.objects.keys())
        for i in range(len(ids)):
            o1 = self.objects[ids[i]]
            for j in range(i + 1, len(ids)):
                o2 = self.objects[ids[j]]
                d = float(np.linalg.norm(o1.centroid_3d - o2.centroid_3d))
                if d >= NEAR_THRESHOLD_M:
                    continue

                # Build a kinds list so both 'near' and 'on' can coexist on
                # the same node pair; NetworkX's add_edge is upsert-only
                # which used to clobber 'near' when 'on' was also added.
                kinds = ["near"]
                dy = float(o1.centroid_3d[1] - o2.centroid_3d[1])
                dxz = float(np.linalg.norm(
                    np.array([o1.centroid_3d[0] - o2.centroid_3d[0],
                              o1.centroid_3d[2] - o2.centroid_3d[2]])))
                if dxz < 0.4 and abs(dy) < SUPPORT_HEIGHT_TOL and abs(dy) > 0.05:
                    kinds.append("on")
                    # Canonicalise the (upper, lower) direction
                    if dy > 0:
                        u, v = o1.id, o2.id
                    else:
                        u, v = o2.id, o1.id
                    self.graph.add_edge(u, v, kind="on",
                                        kinds=kinds, distance=d, dy=abs(dy))
                else:
                    self.graph.add_edge(o1.id, o2.id, kind="near",
                                        kinds=kinds, distance=d)

    def reset(self):
        """Drop all in-memory objects + edges. Called on a SLAM reset so the
        registry never mixes detections from two different coordinate frames
        (the pre/post-reset poses are unrelated). The on-disk persistent
        scene graph is intentionally left untouched."""
        with self._lock:
            self.objects.clear()
            self.graph.clear()
            self.frame_counter = 0
            logger.info("Semantic map reset.")

    # ------------------------------------------------------------------
    # Pruning + query API
    # ------------------------------------------------------------------
    def prune(self, max_unseen_seconds: float = 30.0):
        """Drop objects we haven't seen for a while *and* that were rarely seen.
        Long-lived objects survive even if temporarily occluded."""
        with self._lock:
            now = time.time()
            drop = []
            for tid, obj in self.objects.items():
                unseen = now - obj.last_seen
                if unseen > max_unseen_seconds and obj.hits < 5:
                    drop.append(tid)
            for tid in drop:
                del self.objects[tid]
                if self.graph.has_node(tid):
                    self.graph.remove_node(tid)

    def get_semantic_objects(self) -> List[Dict]:
        with self._lock:
            return [obj.to_dict() for obj in self.objects.values() if obj.hits >= 2]

    def get_scene_graph(self) -> Dict:
        """Lightweight JSON payload of the graph for the frontend."""
        with self._lock:
            nodes = []
            for n, attrs in self.graph.nodes(data=True):
                entry = {"id": int(n), "label": str(attrs.get("label", "?"))}
                if "centroid_3d" in attrs:
                    entry["centroid_3d"] = [float(x) for x in attrs["centroid_3d"]]
                if "hits" in attrs:
                    entry["hits"] = int(attrs["hits"])
                nodes.append(entry)

            edges = []
            for u, v, d in self.graph.edges(data=True):
                kinds = d.get("kinds")
                if not kinds:
                    kinds = [d.get("kind", "?")]
                edges.append({
                    "source": int(u),
                    "target": int(v),
                    "kind": str(d.get("kind", kinds[0])),
                    "kinds": [str(k) for k in kinds],
                    "distance": float(d.get("distance", 0.0)),
                })
            return {"nodes": nodes, "edges": edges}

    def get_scene_description(self) -> List[str]:
        """Plain-language summary derived from current edges."""
        with self._lock:
            out = []
            for u, v, d in self.graph.edges(data=True):
                kinds = d.get("kinds", [d.get("kind", "?")])
                lu = self.graph.nodes[u].get("label", "?")
                lv = self.graph.nodes[v].get("label", "?")
                if "on" in kinds:
                    out.append(f"{lu} #{u} on {lv} #{v}")
                elif "near" in kinds:
                    out.append(f"{lu} #{u} near {lv} #{v} ({d.get('distance', 0):.1f}m)")
            return out

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_persistent(self):
        try:
            os.makedirs(os.path.dirname(self._persist_path), exist_ok=True)
            data = {
                "objects": [obj.to_dict() for obj in self.objects.values()],
                "edges": [
                    {"source": int(u), "target": int(v), **d}
                    for u, v, d in self.graph.edges(data=True)
                ],
                "saved_at": time.time(),
            }
            with open(self._persist_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Scene graph saved -> {self._persist_path}")
        except Exception as e:
            logger.warning(f"Scene graph save failed: {e}")

    def load_persistent(self):
        if not os.path.exists(self._persist_path):
            return
        try:
            with open(self._persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for entry in data.get("objects", []):
                obj = SemanticObject(
                    entry["id"], entry["label"],
                    entry["centroid_3d"], entry["bbox_3d"], entry["confidence"],
                )
                obj.hits = int(entry.get("hits", 1))
                obj.first_seen = float(entry.get("first_seen", time.time()))
                obj.last_seen = float(entry.get("last_seen", time.time()))
                self.objects[obj.id] = obj
                self.graph.add_node(obj.id, label=obj.label,
                                    centroid_3d=[float(x) for x in obj.centroid_3d],
                                    bbox_3d=[float(x) for x in obj.bbox_3d],
                                    hits=int(obj.hits), last_seen=float(obj.last_seen))
            for ed in data.get("edges", []):
                if ed["source"] in self.graph and ed["target"] in self.graph:
                    self.graph.add_edge(ed["source"], ed["target"],
                                        **{k: v for k, v in ed.items()
                                           if k not in ("source", "target")})
            logger.info(f"Loaded {len(self.objects)} persistent objects.")
        except Exception as e:
            logger.warning(f"Scene graph load failed: {e}")
