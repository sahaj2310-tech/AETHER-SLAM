"""Provenance helpers: a content hash of the source tree (the repository is not a
git checkout), the GPU / CPU description and library versions."""
import hashlib
import os
import platform

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def code_hash(dirs=("backend", "tip")) -> str:
    h = hashlib.sha256()
    for d in dirs:
        for base, subdirs, files in sorted(os.walk(os.path.join(ROOT, d))):
            subdirs[:] = sorted(x for x in subdirs if x not in ("__pycache__", "static", "data"))
            for f in sorted(files):
                if f.endswith(".py"):
                    p = os.path.join(base, f)
                    h.update(os.path.relpath(p, ROOT).replace(os.sep, "/").encode())
                    h.update(open(p, "rb").read())
    return h.hexdigest()[:16]


def environment() -> dict:
    info = {"platform": platform.platform(), "python": platform.python_version(),
            "cpu": platform.processor()}
    try:
        import subprocess
        out = subprocess.check_output(["powershell", "-NoProfile", "-Command",
                                       "(Get-CimInstance Win32_Processor).Name"], text=True, timeout=20)
        info["cpu"] = out.strip()
    except Exception:
        pass
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(p.total_memory / 2 ** 30, 2)
    except Exception:
        pass
    for mod in ("kornia", "open3d", "ultralytics", "numpy", "cv2"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", None)
        except Exception:
            pass
    return info
