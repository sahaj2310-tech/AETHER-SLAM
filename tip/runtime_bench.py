"""
Runtime and memory benchmark (Section VI-H, Proposition 2).

Part A -- isolated components on real frames (warm-up, then N timed runs, each
          followed by torch.cuda.synchronize()); peak memory allocated by the
          PyTorch allocator while the component is loaded and running.
Part B -- the live asynchronous configuration: the real server (backend.main)
          replays a recorded sequence at its native 30 Hz as a virtual camera with
          a dashboard WebSocket client attached; /api/perf gives throughput, stage
          service times, the capture->fusion age of every fused frame and peak GPU
          memory. Optionally repeated with the process capped at 4 GB of device memory.

  python tip/runtime_bench.py --seq <data>/rgbd_dataset_freiburg1_room --out results/runtime
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)


def _timeit(fn, n, warmup, sync):
    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); sync(); ts.append(1e3 * (time.perf_counter() - t0))
    ts = np.array(ts)
    return dict(median_ms=round(float(np.median(ts)), 2), p95_ms=round(float(np.percentile(ts, 95)), 2),
                mean_ms=round(float(ts.mean()), 2), n=int(n))


def components(seq_root, n=200, warmup=20, depth_models=("DPT_Hybrid", "DPT_Large", "MiDaS_small", "DepthAnythingV2_Small")):
    import torch
    _ol = torch.load
    torch.load = lambda *a, **k: _ol(*a, **{**k, "weights_only": k.get("weights_only", False)})
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from backend.config import Config
    from backend.sources import RecordedSequence
    cfg = Config()
    seq = RecordedSequence.open(seq_root)
    cfg.set_camera(seq.K, *seq.size)
    frames = [seq.frame(i) for i in range(0, 400, 2)]
    sync = torch.cuda.synchronize
    out = {}
    mb = lambda: round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)

    def fresh():
        import gc
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        return torch.cuda.memory_allocated()

    # ---- DISK + LightGlue
    base = fresh()
    from backend.slam.feature_backend import DiskLightGlueBackend
    fb = DiskLightGlueBackend(max_features=cfg.MAX_FEATURES, half_precision=cfg.FEATURE_HALF_PRECISION)
    feats = [fb.detect(f) for f in frames[:50]]
    k = {"i": 0}

    def det():
        fb.detect(frames[k["i"] % len(frames)]); k["i"] += 1

    def mat():
        fb.match(feats[k["i"] % 49], feats[k["i"] % 49 + 1]); k["i"] += 1

    def both():
        a = fb.detect(frames[k["i"] % len(frames)]); fb.match(feats[k["i"] % 50], a); k["i"] += 1
    out["DISK"] = _timeit(det, n, warmup, sync)
    out["LightGlue"] = _timeit(mat, n, warmup, sync)
    out["DISK+LightGlue"] = _timeit(both, n, warmup, sync)
    out["DISK+LightGlue"]["peak_mb"] = round((torch.cuda.max_memory_allocated() - base) / 2 ** 20, 1)
    del fb, feats

    # ---- YOLOv8n-seg (person boxes)
    base = fresh()
    from backend.semantic.detector import YOLODetector
    yd = YOLODetector(cfg, person_only=True)
    out["YOLOv8n-seg"] = _timeit(lambda: yd.detect_person_boxes(frames[k["i"] % len(frames)]), n, warmup, sync)
    out["YOLOv8n-seg"]["peak_mb"] = round((torch.cuda.max_memory_allocated() - base) / 2 ** 20, 1)
    del yd

    # ---- depth networks
    from backend.semantic.depth import DepthEstimator
    nus = None
    for dm in depth_models:
        base = fresh()
        try:
            de = DepthEstimator(cfg, model_type=dm)
            if de.model is None:
                out[dm] = {"error": "not available"}; continue
            r = _timeit(lambda: de.predict(frames[k["i"] % len(frames)]), n, warmup, sync)
            r["peak_mb"] = round((torch.cuda.max_memory_allocated() - base) / 2 ** 20, 1)
            out[dm] = r
            if dm == cfg.DEPTH_MODEL_TYPE:
                nus = [de.predict(f) for f in frames[:60]]
            del de
        except Exception as e:
            out[dm] = {"error": str(e)[:200]}

    # ---- CG-SSF update (CPU) with real-size anchor sets, and per-pixel depth/weights (GPU)
    from backend.semantic.cg_ssf import CGSSF, CGSSFConfig
    rng = np.random.default_rng(0)
    f = CGSSF(CGSSFConfig(eta_s=cfg.CGSSF_ETA_S, eta_b=cfg.CGSSF_ETA_B, sigma_s=cfg.CGSSF_SIGMA_S, sigma_b=cfg.CGSSF_SIGMA_B))
    nu_a = rng.uniform(0.5, 2.0, 120); rho_a = 0.4 * nu_a + 0.25 + rng.normal(0, 0.01, 120)
    tt = {"t": 0.0}

    def upd():
        tt["t"] += 1 / 30; f.update(nu_a + rng.normal(0, 0.01, 120), rho_a, tt["t"])
    out["CG-SSF update (CPU)"] = _timeit(upd, 2000, 50, lambda: None)
    from backend.semantic.alignment import _median_normalise
    nu0 = nus[0] if nus else torch.rand(seq.size[1], seq.size[0], device="cuda")

    def dmap():
        nt = _median_normalise(nu0, None)
        f.depth_torch(nt, voxel=cfg.TSDF_VOXEL_M, trunc=cfg.TSDF_TRUNC_M)
    out["Depth, sigma_z, weights (GPU)"] = _timeit(dmap, n, warmup, sync)

    # ---- weighted TSDF integration (GPU) + marching cubes (CPU), on aligned real depth
    from backend.semantic.tsdf_gpu import WeightedTSDF
    base = fresh()
    ts = WeightedTSDF(cfg.TSDF_VOXEL_M, cfg.TSDF_TRUNC_M, max_depth=cfg.TSDF_MAX_DEPTH_M)
    Ks = cfg.INTRINSICS_MATRIX
    gts = np.loadtxt(os.path.join(seq_root, "groundtruth.txt"), comments="#")
    from scipy.spatial.transform import Rotation
    poses = []
    for row in gts[::10][:60]:
        T = np.eye(4); T[:3, :3] = Rotation.from_quat(row[4:8]).as_matrix(); T[:3, 3] = row[1:4]; poses.append(T)
    depths = []
    for nu in (nus or [nu0]):
        z, _, valid, w = f.depth_torch(_median_normalise(nu, None), voxel=cfg.TSDF_VOXEL_M, trunc=cfg.TSDF_TRUNC_M)
        depths.append((torch.where(valid, z, torch.zeros_like(z)), w))
    rgb = torch.from_numpy(np.ascontiguousarray(frames[0][:, :, ::-1])).cuda()
    j = {"i": 0}

    def integ():
        d, w = depths[j["i"] % len(depths)]
        ts.integrate(d, w, rgb, Ks, poses[j["i"] % len(poses)]); j["i"] += 1
    out["Weighted TSDF integration (GPU)"] = _timeit(integ, n, warmup, sync)
    out["Weighted TSDF integration (GPU)"]["peak_mb"] = round((torch.cuda.max_memory_allocated() - base) / 2 ** 20, 1)
    out["Weighted TSDF integration (GPU)"]["blocks"] = ts.stats()
    out["Marching cubes (CPU)"] = _timeit(lambda: ts.extract_mesh(), 10, 2, lambda: None)
    return out


# ------------------------------------------------------------------------- live
def live(seq_root, seconds, warmup_s, cap_gb=None, port=8010, ws=True, variant="cgssf"):
    import urllib.request
    env = dict(os.environ, AETHER_SOURCE=seq_root, PYTHONIOENCODING="utf-8")
    if cap_gb:
        env["AETHER_GPU_CAP_GB"] = str(cap_gb)
    log = open(os.path.join(ROOT, "backend", "data", f"runtime_server_{port}.log"), "w")
    code = ("import backend.main as m, uvicorn; m.config.PORT=%d; m.config.CAPTURE_LOOP=False; "
            "m.config.AUDIO_DETECTION_ENABLED=False; m.config.ALIGNMENT_VARIANT=%r; "
            "uvicorn.Server(uvicorn.Config(m.app, host='127.0.0.1', port=%d, log_level='warning')).run()"
            % (port, variant, port))
    proc = subprocess.Popen([sys.executable, "-c", code], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"

    def get(path, method="GET"):
        req = urllib.request.Request(base + path, method=method)
        return json.loads(urllib.request.urlopen(req, timeout=10).read())
    t0 = time.time()
    while True:
        try:
            get("/api/status"); break
        except Exception:
            if time.time() - t0 > 300 or proc.poll() is not None:
                proc.kill(); raise RuntimeError("server did not start")
            time.sleep(1)
    stop = threading.Event()
    received = {"n": 0, "bytes": 0}

    def ws_client():
        import asyncio
        import websockets

        async def run():
            async with websockets.connect(f"ws://127.0.0.1:{port}/ws", max_size=None) as w:
                while not stop.is_set():
                    try:
                        msg = await asyncio.wait_for(w.recv(), timeout=1.0)
                        received["n"] += 1; received["bytes"] += len(msg)
                    except asyncio.TimeoutError:
                        pass
        try:
            asyncio.run(run())
        except Exception:
            pass
    th = threading.Thread(target=ws_client, daemon=True)
    if ws:
        th.start()
    # wait until frames flow, then discard the warm-up window
    while get("/api/perf")["counts"].get("fusion", 0) < 5:
        time.sleep(0.5)
        if time.time() - t0 > 300:
            break
    time.sleep(warmup_s)
    get("/api/perf/reset", "POST")
    time.sleep(seconds)
    snap = get("/api/perf")
    status = get("/api/status")
    stop.set()
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except Exception:
        proc.kill()
    snap["status"] = status
    snap["ws_messages"] = received["n"]; snap["ws_mb"] = round(received["bytes"] / 2 ** 20, 1)
    snap["cap_gb"] = cap_gb
    return snap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--live-seq", default=None, help="sequence replayed in the live test (default: --seq)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--warmup", type=float, default=8.0)
    ap.add_argument("--skip-components", action="store_true")
    ap.add_argument("--skip-live", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(os.path.join(ROOT, "backend", "data"), exist_ok=True)
    from provenance import code_hash, environment
    res = {"env": environment(), "code_hash": code_hash(), "finished": None}
    path = os.path.join(a.out, "runtime.json")
    if os.path.exists(path):
        res.update({k: v for k, v in json.load(open(path)).items() if k not in ("env", "code_hash")})
    if not a.skip_components:
        res["components"] = components(a.seq)
        json.dump(res, open(path, "w"), indent=1)
        print(json.dumps(res["components"], indent=1))
    if not a.skip_live:
        ls = a.live_seq or a.seq
        res["live"] = live(ls, a.seconds, a.warmup)
        json.dump(res, open(path, "w"), indent=1)
        res["live_cap4gb"] = live(ls, a.seconds, a.warmup, cap_gb=4.0, port=8011)
        print(json.dumps({k: res[k] for k in ("live", "live_cap4gb")}, indent=1)[:4000])
    res["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(res, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
