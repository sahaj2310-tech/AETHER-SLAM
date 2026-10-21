"""Microphone sound / voice-activity detection.

Detection ONLY, no direction finding. (Bearing/azimuth needs a real spaced
microphone array, which a laptop doesn't have; that whole subsystem was
removed.) This reports:

    energy      – RMS level of the latest audio chunk
    level       – energy normalised to 0..1 for a UI meter
    vad_active  – is sound present right now (self-calibrating threshold)
    type        – "idle" / "voice" / "loud"
    available   – whether a working microphone is currently open

Robust to flaky Windows audio:
  * Tries every host API's default input device (and all input devices) with
    native sample rates, keeping the FIRST whose callback actually delivers
    samples. A dead/disabled endpoint never fires its callback and is skipped.
  * Opens the stream ONCE and keeps it (repeatedly opening/closing devices
    wedges the Windows audio stack).
  * Self-healing: if no mic is available at startup (e.g. Windows mic access
    is disabled), it keeps retrying on a back-off, so enabling the mic (or
    rebooting) makes detection start working without restarting the server.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

logger = logging.getLogger("AudioDetector")

try:
    import sounddevice as sd
    SD_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    SD_AVAILABLE = False
    logger.warning("sounddevice not installed; audio detection disabled.")


class AudioDetector:
    def __init__(self, config):
        self.config = config
        self.stream = None
        self.device_index = None
        self.device_name = "?"
        self.channels = 1
        self.sample_rate = int(getattr(config, "AUDIO_SAMPLE_RATE", 44100))
        self.blocksize = int(getattr(config, "AUDIO_BLOCKSIZE", 1024))

        self._lock = threading.Lock()
        self._rms = 0.0

        # Self-calibrating VAD: a running estimate of the quiet noise floor;
        # we flag activity when RMS rises well above it (so it works regardless
        # of how hot/quiet the particular microphone's gain is).
        self._noise_floor = 0.005
        self._abs_floor = float(getattr(config, "VAD_ABS_FLOOR", 0.006))
        self._margin = float(getattr(config, "VAD_MARGIN", 3.5))
        self._loud = float(getattr(config, "VAD_LOUD_RMS", 0.12))
        self._level_full = float(getattr(config, "VAD_LEVEL_FULL_RMS", 0.12))

        self._last_open_attempt = 0.0
        self._open_backoff = 3.0  # seconds between open attempts while failing

    # ------------------------------------------------------------------
    def _candidate_devices(self):
        devs = sd.query_devices()
        order = []
        try:
            order.append(sd.default.device[0])
        except Exception:
            pass
        for h in sd.query_hostapis():
            di = h.get("default_input_device", -1)
            if di is not None and di >= 0:
                order.append(di)
        # then any remaining input device
        for idx, d in enumerate(devs):
            if d["max_input_channels"] >= 1:
                order.append(idx)
        out, seen = [], set()
        skip = ("output", "speaker", "wave out", "loopback", "what u hear")
        for idx in order:
            if idx in seen or idx is None or idx < 0 or idx >= len(devs):
                continue
            seen.add(idx)
            d = devs[idx]
            if d["max_input_channels"] < 1:
                continue
            if any(s in d["name"].lower() for s in skip):
                continue
            ch = 1  # mono mix-down is all we need for detection
            sr = int(d["default_samplerate"]) or self.sample_rate
            out.append((idx, ch, sr, d["name"]))
        return out

    def _make_callback(self):
        def _cb(indata, frames, time_info, status):
            x = indata[:, 0] if indata.ndim > 1 else indata.ravel()
            rms = float(np.sqrt(np.mean(x * x)) + 1e-12)
            with self._lock:
                self._rms = rms
        return _cb

    def _try_open(self, idx, ch, sr, name):
        """Open a persistent stream and confirm the callback actually fires."""
        fired = {"ok": False}
        cb = self._make_callback()

        def _probe_cb(indata, frames, time_info, status):
            fired["ok"] = True
            cb(indata, frames, time_info, status)

        try:
            s = sd.InputStream(device=idx, channels=ch, samplerate=sr,
                               blocksize=self.blocksize, callback=_probe_cb)
            s.start()
        except Exception:
            return False
        time.sleep(0.4)
        if not fired["ok"]:
            try:
                s.stop(); s.close()
            except Exception:
                pass
            return False
        self.stream = s
        self.device_index = idx
        self.device_name = name
        self.channels = ch
        self.sample_rate = sr
        return True

    def ensure_open(self):
        """Open a microphone if we don't have one yet (throttled, self-healing)."""
        if self.stream is not None or not SD_AVAILABLE:
            return
        now = time.time()
        if now - self._last_open_attempt < self._open_backoff:
            return
        self._last_open_attempt = now
        for idx, ch, sr, name in self._candidate_devices():
            if self._try_open(idx, ch, sr, name):
                logger.info(f"Microphone open: '{name[:34]}' @ {sr} Hz (detection only).")
                self._open_backoff = 3.0
                return
        # No device worked; back off so we don't churn the audio stack.
        self._open_backoff = min(15.0, self._open_backoff + 2.0)
        logger.debug("No microphone available yet; will retry.")

    # ------------------------------------------------------------------
    def event(self) -> dict:
        """Latest detection snapshot for the frontend."""
        if self.stream is None:
            return {"available": False, "energy": 0.0, "level": 0.0,
                    "vad_active": False, "type": "no_mic"}
        with self._lock:
            rms = self._rms
        threshold = max(self._abs_floor, self._noise_floor * self._margin)
        active = rms > threshold
        if not active:
            # Adapt the noise floor toward the current quiet level.
            self._noise_floor = 0.97 * self._noise_floor + 0.03 * rms
        level = float(min(1.0, rms / max(1e-6, self._level_full)))
        if not active:
            sound_type = "idle"
        elif rms > self._loud:
            sound_type = "loud"
        else:
            sound_type = "voice"
        return {"available": True, "energy": round(rms, 5), "level": round(level, 3),
                "vad_active": bool(active), "type": sound_type}

    def close(self):
        if self.stream is not None:
            try:
                self.stream.stop(); self.stream.close()
            except Exception:
                pass
            self.stream = None
