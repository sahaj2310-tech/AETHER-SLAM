"""Multi-threaded perception pipeline.

Stages (each on its own daemon thread):

    [CaptureStage]   ─ camera/simulator ─┐
                                          ▼
                                      frame_q (maxsize=1, drop-oldest)
                                          │
                              ┌───────────┼───────────┐
                              ▼                       ▼
                         [SlamStage]            [SemanticStage]
                              │                       │
                          pose_q (1)             sem_q (1)
                              │                       │
                              └───────────┬───────────┘
                                          ▼
                                     [FusionStage]
                                          │
                                       state ─► broadcaster (async)

A separate `[AudioStage]` thread polls the acoustic locator and pushes
audio events into `audio_q`, which the fusion stage drains opportunistically.

Design notes:
  * Every queue is `maxsize=1` with a *drop-oldest* policy.  This keeps each
    stage processing the freshest frame and never builds backpressure,
    which matters because SLAM and YOLO have different per-frame costs.
  * No microservices, no message broker: plain `queue.Queue` and threads.
  * The orchestrator owns the lifecycle: `start()` / `stop()`.  Modules
    themselves don't know they're running in a thread.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger("Pipeline")


# ---------------------------------------------------------------------------
# Drop-oldest queue
# ---------------------------------------------------------------------------

class LatestQueue(queue.Queue):
    """Bounded queue where put() drops the old item if full."""
    def __init__(self, maxsize: int = 1):
        super().__init__(maxsize=maxsize)

    def put_latest(self, item: Any) -> None:
        try:
            self.put_nowait(item)
        except queue.Full:
            try:
                self.get_nowait()
            except queue.Empty:
                pass
            try:
                self.put_nowait(item)
            except queue.Full:
                pass

    def get_latest(self, timeout: float = 0.05) -> Optional[Any]:
        try:
            return self.get(timeout=timeout)
        except queue.Empty:
            return None


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------

@dataclass
class Stage:
    name: str
    target: Callable
    daemon: bool = True

    def __post_init__(self):
        self._thread = threading.Thread(target=self._wrap, name=self.name, daemon=self.daemon)
        self._stop = threading.Event()
        self._latency_ms = 0.0

    def _wrap(self):
        logger.info(f"Stage start: {self.name}")
        try:
            self.target(self._stop, self)
        except Exception:
            logger.exception(f"Stage crashed: {self.name}")
        logger.info(f"Stage end: {self.name}")

    def start(self):
        self._thread.start()

    def stop(self, join_timeout: float = 2.0):
        self._stop.set()
        self._thread.join(timeout=join_timeout)

    @property
    def latency_ms(self) -> float:
        return self._latency_ms

    def record_latency(self, dt_seconds: float):
        # EMA so a transient spike doesn't dominate
        ms = dt_seconds * 1000.0
        self._latency_ms = 0.8 * self._latency_ms + 0.2 * ms

    def is_alive(self) -> bool:
        return self._thread.is_alive()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sleep_until(deadline_ts: float):
    """Sleep until the wall-clock deadline (drop-on-late)."""
    delay = deadline_ts - time.time()
    if delay > 0:
        time.sleep(delay)
