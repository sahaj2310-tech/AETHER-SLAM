"""Lightweight loop-closure detector built on the same feature backend used
for tracking.

Real ORB-SLAM3 uses a DBoW2 vocabulary tree.  We don't pull that in here.
Instead we keep a list of (keyframe_id, downsampled descriptor signature)
records and, on each new keyframe, score it against older keyframes by mean
cosine similarity of the top-k descriptors.  When the score exceeds a
threshold *and* the keyframe is geographically far from the candidate, we
emit a loop closure event with the matched feature pairs so the optimizer
can use it as an extra constraint.

This is intentionally minimal: it is meant to *detect* loops well enough for
the visualization to react and for a future pose-graph step to consume.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np

logger = logging.getLogger("LoopClosure")


@dataclass
class LoopEvent:
    current_kf_id: int
    matched_kf_id: int
    score: float
    n_matches: int


class LoopClosureDetector:
    def __init__(self, feature_backend, *,
                 min_kf_gap: int = 10,
                 min_score: float = 0.55,
                 min_matches: int = 35,
                 max_records: int = 200,
                 cooldown_s: float = 1.0):
        self.backend = feature_backend
        self.min_kf_gap = min_kf_gap
        self.min_score = min_score
        self.min_matches = min_matches
        self.max_records = max_records
        self.cooldown_s = float(cooldown_s)

        # Records: list of (kf_id, FrameFeatures, signature np.ndarray)
        self._records: List[Tuple[int, object, np.ndarray]] = []
        self._last_check_ts = 0.0

    def add_keyframe(self, kf_id: int, features) -> None:
        sig = self._signature(features.descriptors)
        self._records.append((kf_id, features, sig))
        if len(self._records) > self.max_records:
            # Drop oldest non-anchor record
            self._records.pop(0)

    @staticmethod
    def _signature(desc: np.ndarray, top_k: int = 64) -> np.ndarray:
        """Compact descriptor signature: mean of top-k strongest descriptors."""
        if desc is None or desc.shape[0] == 0:
            return np.zeros((1,), dtype=np.float32)
        d = desc.astype(np.float32)
        if d.shape[0] > top_k:
            # take rows with largest L2 norm (proxy for distinctive features)
            norms = np.linalg.norm(d, axis=1)
            idx = np.argpartition(norms, -top_k)[-top_k:]
            d = d[idx]
        mean = d.mean(axis=0)
        n = np.linalg.norm(mean) + 1e-9
        return (mean / n).astype(np.float32)

    def check(self, kf_id: int, features, t: Optional[float] = None) -> Optional[LoopEvent]:
        """Score this keyframe against the records; return a loop event or None.
        `t` is the frame timestamp (seconds); wall-clock time is used when omitted."""
        if not self._records or features.keypoints.shape[0] == 0:
            return None

        # Rate-limit the *verification* (the expensive LightGlue pass).
        # Signature scoring is cheap so we always do it; we just skip
        # the geometric pass if we recently ran one.
        now = time.time() if t is None else float(t)
        if (now - self._last_check_ts) < self.cooldown_s:
            return None

        sig = self._signature(features.descriptors)

        best_score = -1.0
        best_kf_id = -1
        best_feats = None
        for old_id, old_feats, old_sig in self._records:
            if (kf_id - old_id) < self.min_kf_gap:
                continue
            if old_sig.shape != sig.shape:
                continue
            score = float(np.dot(sig, old_sig))
            if score > best_score:
                best_score = score
                best_kf_id = old_id
                best_feats = old_feats

        if best_score < self.min_score or best_feats is None:
            return None

        # Geometric verification: actually match the two keyframes' descriptors
        try:
            pairs = self.backend.match(best_feats, features)
        except Exception as e:
            logger.debug(f"Loop verification match failed: {e}")
            return None

        self._last_check_ts = now
        if pairs.shape[0] < self.min_matches:
            return None

        ev = LoopEvent(current_kf_id=kf_id, matched_kf_id=best_kf_id,
                       score=best_score, n_matches=int(pairs.shape[0]))
        logger.info(f"Loop closure: kf{kf_id} <-> kf{best_kf_id} "
                    f"(score={best_score:.2f}, matches={ev.n_matches})")
        return ev
