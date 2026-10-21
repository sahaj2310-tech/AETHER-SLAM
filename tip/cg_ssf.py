"""CG-SSF reference implementation used by the simulation and the evaluation scripts.

This module re-exports the implementation that runs inside AETHER-SLAM
(backend/semantic/cg_ssf.py), so that the simulation, the benchmarks and the
live system all execute exactly the same estimator."""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from backend.semantic.cg_ssf import (  # noqa: E402,F401
    CGSSF, CGSSFConfig, _CHI2, normalize_disparity, robust_affine_fit,
)
