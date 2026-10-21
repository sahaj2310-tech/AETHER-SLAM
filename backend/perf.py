"""Thread-safe runtime statistics of the live pipeline (served at /api/perf).

* service time of every stage iteration (ms),
* throughput of every stage (frames per second over a sliding window),
* age of a frame when the fusion stage finishes with it (capture -> fusion),
* the staleness bound of Proposition 2,  a_n <= 2 * sum_i L_i,  evaluated with
  the largest observed service time L_i of each stage on the path.
"""
from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np

PATH_STAGES = ("slam", "semantic", "fusion")


class PerfMonitor:
    def __init__(self, maxlen: int = 20000, window_s: float = 10.0):
        self._lock = threading.Lock()
        self._svc = {}
        self._ts = {}
        self._ages = deque(maxlen=maxlen)
        self.window_s = window_s
        self.maxlen = maxlen
        self.t_start = time.perf_counter()
        self.counts = {}

    def reset(self):
        with self._lock:
            self._svc.clear(); self._ts.clear(); self._ages.clear(); self.counts.clear()
            self.t_start = time.perf_counter()

    def record(self, stage: str, ms: float):
        now = time.perf_counter()
        with self._lock:
            self._svc.setdefault(stage, deque(maxlen=self.maxlen)).append(float(ms))
            self._ts.setdefault(stage, deque(maxlen=self.maxlen)).append(now)
            self.counts[stage] = self.counts.get(stage, 0) + 1

    def record_age(self, ms: float):
        with self._lock:
            self._ages.append(float(ms))

    def fps(self, stage: str) -> float:
        now = time.perf_counter()
        with self._lock:
            ts = self._ts.get(stage)
            if not ts:
                return 0.0
            recent = [t for t in ts if now - t <= self.window_s]
        if len(recent) < 2:
            return 0.0
        return (len(recent) - 1) / max(recent[-1] - recent[0], 1e-6)

    def snapshot(self) -> dict:
        with self._lock:
            svc = {k: np.asarray(v) for k, v in self._svc.items()}
            ts = {k: np.asarray(v) for k, v in self._ts.items()}
            ages = np.asarray(self._ages)
            counts = dict(self.counts)
            up = time.perf_counter() - self.t_start
        out = {"uptime_s": round(up, 1), "counts": counts, "stages": {}}
        for k, v in svc.items():
            if v.size == 0:
                continue
            t = ts[k]
            span = t[-1] - t[0] if t.size > 1 else 0.0
            out["stages"][k] = {
                "n": int(v.size), "median_ms": round(float(np.median(v)), 2),
                "p95_ms": round(float(np.percentile(v, 95)), 2), "max_ms": round(float(v.max()), 2),
                "mean_ms": round(float(v.mean()), 2),
                "fps_mean": round((t.size - 1) / span, 2) if span > 0 else 0.0,
                "fps_window": round(self.fps(k), 2),
            }
        if ages.size:
            out["age_ms"] = {"n": int(ages.size), "median": round(float(np.median(ages)), 1),
                             "p95": round(float(np.percentile(ages, 95)), 1), "max": round(float(ages.max()), 1)}
        Ls = [out["stages"][k]["max_ms"] for k in PATH_STAGES if k in out["stages"]]
        if len(Ls) == len(PATH_STAGES):
            out["bound_ms"] = round(2.0 * float(np.sum(Ls)), 1)
            out["bound_p95_ms"] = round(2.0 * float(np.sum([out["stages"][k]["p95_ms"] for k in PATH_STAGES])), 1)
        try:
            import torch
            if torch.cuda.is_available():
                out["gpu"] = {"name": torch.cuda.get_device_name(0),
                              "allocated_mb": round(torch.cuda.memory_allocated() / 2 ** 20, 1),
                              "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
                              "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 2 ** 20, 1)}
        except Exception:
            pass
        return out
