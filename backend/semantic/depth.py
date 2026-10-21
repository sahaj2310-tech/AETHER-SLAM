"""Monocular depth network wrapper.

`DepthEstimator.predict` returns the network's *relative inverse depth* nu (an
affine-invariant disparity-like map, MiDaS/DPT convention) at the frame
resolution, as a float32 tensor on the model device.  Aligning nu to the SLAM
map gauge is done by `backend.semantic.alignment` (CG-SSF or an ablation
variant); this module no longer contains any alignment logic.

Supported models (config.DEPTH_MODEL_TYPE):
  DPT_Hybrid (default), DPT_Large, MiDaS_small   -- torch.hub intel-isl/MiDaS
  DepthAnythingV2_Small                          -- Hugging Face transformers
  DepthAnythingV2_MetricIndoor_Small             -- the metric (indoor) version of the same
                                                    network; its depth (m) is returned inverted,
                                                    so nu is metric inverse depth (comparison of
                                                    metric and relative priors)
"""
import logging
import warnings

import cv2
import numpy as np
import torch

from backend.config import Config

# The DPT/MiDaS backbones pull in older timm import paths and a renamed model
# alias -- both emit harmless deprecation warnings on load.
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
warnings.filterwarnings("ignore", message=".*Importing from timm.*")
warnings.filterwarnings("ignore", message=".*Mapping deprecated model name.*")

logger = logging.getLogger("DepthEstimator")

MIDAS_MODELS = ("DPT_Hybrid", "DPT_Large", "MiDaS_small")
HF_MODELS = {"DepthAnythingV2_Small": "depth-anything/Depth-Anything-V2-Small-hf",
             "DepthAnythingV2_Base": "depth-anything/Depth-Anything-V2-Base-hf",
             "DepthAnythingV2_MetricIndoor_Small": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"}
METRIC_MODELS = ("DepthAnythingV2_MetricIndoor_Small",)


class DepthEstimator:
    def __init__(self, config: Config, model_type: str | None = None):
        self.config = config
        self.model_type = model_type or config.DEPTH_MODEL_TYPE
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_half = bool(config.DEPTH_HALF_PRECISION) and self.device.type == "cuda"
        self.model = None
        self.transform = None
        self._hf_processor = None
        try:
            logger.info(f"Initializing depth network ({self.model_type}) on {self.device}...")
            if self.model_type in MIDAS_MODELS:
                self.model = torch.hub.load("intel-isl/MiDaS", self.model_type, trust_repo=True)
                tr = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
                self.transform = tr.dpt_transform if self.model_type.startswith("DPT") else tr.small_transform
            elif self.model_type in HF_MODELS:
                from transformers import AutoModelForDepthEstimation
                self.model = AutoModelForDepthEstimation.from_pretrained(HF_MODELS[self.model_type])
            else:
                raise ValueError(f"unknown depth model {self.model_type}")
            self.model.to(self.device)
            if self.use_half:
                self.model = self.model.half()
            self.model.eval()
            logger.info(f"Depth network loaded (half={self.use_half}).")
        except Exception as e:
            logger.error(f"Failed to load depth network: {e}")
            self.model = None

    @property
    def available(self) -> bool:
        return self.model is not None

    # ------------------------------------------------------------------
    def _hf_input(self, rgb: np.ndarray) -> torch.Tensor:
        """Depth Anything V2 preprocessing: resize (keep aspect) so both sides are
        multiples of 14 with the short side 518, ImageNet normalisation."""
        h, w = rgb.shape[:2]
        s = 518.0 / min(h, w)
        nh, nw = int(round(h * s / 14)) * 14, int(round(w * s / 14)) * 14
        x = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
        x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        return torch.from_numpy(x.transpose(2, 0, 1))[None]

    @torch.no_grad()
    def predict(self, frame_bgr: np.ndarray) -> torch.Tensor | None:
        """Relative inverse depth nu (H, W) float32 on self.device, or None on failure."""
        if self.model is None:
            return None
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        try:
            if self.model_type in MIDAS_MODELS:
                x = self.transform(rgb).to(self.device)
                if self.use_half:
                    x = x.half()
                pred = self.model(x)
            else:
                x = self._hf_input(rgb).to(self.device)
                if self.use_half:
                    x = x.half()
                pred = self.model(pixel_values=x).predicted_depth
                if self.model_type in METRIC_MODELS:      # metric depth (m) -> inverse depth (1/m)
                    pred = 1.0 / pred.float().clamp(min=1e-3)
            nu = torch.nn.functional.interpolate(pred.unsqueeze(1).float(), size=(h, w),
                                                 mode="bicubic", align_corners=False)[0, 0]
            return torch.nan_to_num(nu, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as e:
            logger.error(f"Depth estimation error: {e}")
            return None

    def gauge_depth(self, nu: torch.Tensor, valid: torch.Tensor | None = None) -> np.ndarray:
        """Depth used to (re)initialise the map: z = D0 / nu~ with nu~ = nu / median(nu),
        i.e. alignment parameters s = 1/D0, b = 0. NaN outside [z_min, z_max]."""
        d0 = float(getattr(self.config, "GAUGE_MEDIAN_DEPTH", 2.0))
        sel = nu[valid] if valid is not None else nu.reshape(-1)
        sel = sel[sel > 0]
        med = float(sel.median()) if sel.numel() else 1.0
        nu_t = nu / max(med, 1e-12)
        z = torch.where(nu_t > 0, d0 / nu_t.clamp(min=1e-12), torch.full_like(nu_t, float("nan")))
        z = torch.where((z >= self.config.DEPTH_Z_MIN) & (z <= self.config.DEPTH_Z_MAX), z,
                        torch.full_like(z, float("nan")))
        if valid is not None:
            z = torch.where(valid, z, torch.full_like(z, float("nan")))
        return z.cpu().numpy().astype(np.float32)

    def reset_state(self):
        """Kept for API compatibility; the network itself is stateless."""
        return None
