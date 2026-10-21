"""
Calibrate the CG-SSF noise parameters (process noise eta_s, eta_b and the
measurement-noise floor sigma_s, sigma_b) on a held-out sequence that is
excluded from every table (TUM fr2/xyz):

  python tip/calibrate_eta.py --root <data>/rgbd_dataset_freiburg2_xyz --out results/calibration

The system runs with the full CG-SSF; at every frame the raw per-frame robust
fit (Eqs. 2-4) is recorded, and cg_ssf.calibrate_noise fits the variogram of a
random walk observed in white noise to the fits of well-observed frames
(>= 50 anchors, kappa >= 0.05). Because the anchors depend on the alignment
(new map points are created from the aligned depth), the procedure is repeated
with the calibrated values until they change by less than 10 % (fixed point).

--q-mode view indexes the random walk by the camera's view change (rotation angle in rad +
translation / median scene depth) instead of time: the variogram is fitted against the
cumulative view change between the two fits (cg_ssf.calibrate_noise_view).
"""
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import torch  # noqa: E402
_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": k.get("weights_only", False)})
from backend.config import Config  # noqa: E402
from backend.pipeline.synchronous import NetworkBundle, SynchronousPipeline  # noqa: E402
from backend.semantic.cg_ssf import calibrate_noise, calibrate_noise_view, view_change  # noqa: E402
from backend.sources import RecordedSequence  # noqa: E402
from provenance import code_hash, environment  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--passes", type=int, default=4); ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--q-mode", default="time", choices=("time", "view"))
    ap.add_argument("--init", default="", help="eta_s,eta_b,sigma_s,sigma_b of the first pass")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = Config()
    seq = RecordedSequence.open(a.root)
    cfg.set_camera(seq.K, *seq.size)
    n = len(seq) if a.max_frames <= 0 else min(a.max_frames, len(seq))
    dt = float(np.median(np.diff(seq.timestamps[:n])))
    nets = NetworkBundle(cfg, cache=True)
    params = dict(eta_s=cfg.CGSSF_ETA_S, eta_b=cfg.CGSSF_ETA_B, sigma_s=cfg.CGSSF_SIGMA_S, sigma_b=cfg.CGSSF_SIGMA_B)
    if a.init:
        params = dict(zip(("eta_s", "eta_b", "sigma_s", "sigma_b"), (float(x) for x in a.init.split(","))))
    Config.CGSSF_Q_MODE = a.q_mode
    history = []
    for it in range(a.passes):
        Config.CGSSF_ETA_S, Config.CGSSF_ETA_B = params["eta_s"], params["eta_b"]
        Config.CGSSF_SIGMA_S, Config.CGSSF_SIGMA_B = params["sigma_s"], params["sigma_b"]
        pipe = SynchronousPipeline(cfg, nets, variant="cgssf", seed=0)
        fits = []
        t0 = time.time()
        M, prev_T = 0.0, None
        for i in range(n):
            r = pipe.step(seq.frame(i), seq.timestamps[i], i)   # decoded on the fly (long sequence)
            if r.pose is not None:
                dm = view_change(prev_T, r.pose, pipe.aligner.f.median_depth())
                M += dm if dm is not None else 0.0
                prev_T = r.pose
            lf = pipe.aligner.f.last_fit if r.align_status not in ("none", "skipped") else None
            if lf is not None and r.pose is not None:
                fits.append((i, float(lf["theta"][0]), float(lf["theta"][1]), lf["n"], lf["kappa"], M))
            pipe.aligner.f.last_fit = None
        json.dump(fits, open(os.path.join(a.out, f"fits_pass{it}.json"), "w"))   # per-frame fits (inspection)
        if a.q_mode == "view":
            cal = calibrate_noise_view(fits)
            curve = dict(lags=cal["lags"], view=cal["view"], D_s=cal["D_s"], D_b=cal["D_b"])
        else:
            cal = calibrate_noise([f_[:5] for f_ in fits], dt)
            curve = dict(lags=cal["lags"], D_s=cal["D_s"], D_b=cal["D_b"])
        rec = dict(iteration=it, used=dict(params), calibrated={k: cal[k] for k in ("eta_s", "eta_b", "sigma_s", "sigma_b")},
                   variogram=curve, n_fit_frames=cal["n_frames"], view_change_total=M,
                   status_counts=pipe.status_counts, reinit=pipe.n_reinit, tracked=None, seconds=time.time() - t0)
        history.append(rec)
        print(f"pass {it}: used {params} -> calibrated {rec['calibrated']} "
              f"({cal['n_frames']} frames, status {pipe.status_counts})", flush=True)
        new = rec["calibrated"]
        change = max(abs(new[k] - params[k]) / max(abs(new[k]), 1e-9) for k in new)
        params = dict(new)
        if it > 0 and change < 0.10:
            break
    out = dict(sequence=os.path.basename(os.path.normpath(a.root)), frames=n, dt=dt, q_mode=a.q_mode, final=params,
               passes=history, code_hash=code_hash(), env=environment(),
               finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(out, open(os.path.join(a.out, "calibration.json"), "w"), indent=1)
    print("FINAL", json.dumps(params))


if __name__ == "__main__":
    main()
