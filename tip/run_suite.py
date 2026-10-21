"""
Replay a TUM RGB-D or Replica (NICE-SLAM render) sequence through AETHER-SLAM in
synchronous mode (every frame processed, nothing dropped) for several alignment
variants and seeds, and evaluate every run in-process.

  python tip/run_suite.py --root <data>/rgbd_dataset_freiburg1_desk --name fr1_desk \
         --variants cgssf,proto_ema --seeds 0,1,2 --runs <runs>

The networks run once per frame: their outputs (and LightGlue matches) are cached
and shared by all variants / seeds of the sequence (NetworkBundle, cache=True).

Per run, <runs>/<tag>/<name>/seed<k>/ receives
  traj_tum.txt   online camera-to-world pose of every tracked frame (TUM format)
  frames.csv     per-frame status: map generation, tracking / alignment status, anchors, s, b
  timing.csv     per-frame stage timings (ms; network stages are ~0 when cached)
  metrics.json   ATE (Sim(3)), depth AbsRel / delta1 / SC (TUM), Acc / Comp / CR / F (Replica)
  meta.json      variant, seed, dataset, intrinsics, code hash, GPU / CPU, versions, peak memory
  gt_tum.txt     (Replica) ground-truth trajectory with t = index / 30
  mesh.ply       final mesh (only with --save-mesh)
With --oracle-poses the camera pose of every frame is the ground-truth pose (the map frame
is the metric ground-truth frame): depth (TUM and Replica) and reconstruction are evaluated
at scale 1 without trajectory alignment, isolating alignment and fusion from tracking.
--extra-fusion adds fusion volumes for the accuracy-completeness trade-off (CG-SSF variants);
--diag records per-frame errors of the raw fit / estimator output / best affine fit.
--unc-eval (with --oracle-poses) evaluates the per-pixel depth uncertainty of the CG-SSF variants
against the ground-truth depth (sparsification curves, AUSE / AURG, coverage; oracle_tools.sparsification).
--band-variants runs the listed variants also in the anchor-restriction test (tag suffix _band);
--plan runs several depth networks and settings in one process, sharing the cached keypoints, matches,
and person boxes (the depth outputs are recomputed for every network).
Depth metrics use the aligned depth of every processed frame (clipped to
[z_min, z_max]) over all pixels with valid ground truth and the single Sim(3)
scale of the trajectory alignment (eval_depth.depth_metrics).
"""
import argparse
import logging
import csv
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": k.get("weights_only", False)})

from backend.config import Config  # noqa: E402
from backend.pipeline.synchronous import NetworkBundle, SynchronousPipeline  # noqa: E402
from backend.semantic.alignment import VARIANTS  # noqa: E402
from backend.sources import RecordedSequence  # noqa: E402
from eval_common import associate, read_tum_traj, umeyama_sim3  # noqa: E402
from eval_depth import depth_metrics  # noqa: E402
from oracle_tools import (GTDepth, frame_depth_errors, gate_diagnostics, gt_pose_provider,  # noqa: E402
                          sparsification, unc_frame_samples)
from provenance import code_hash, environment  # noqa: E402


class _ErrorLog(logging.Handler):
    """Collects every ERROR record of the process (network load failure, out-of-memory in depth
    estimation or detection, ...). A run is not saved once one occurred: the cached network outputs
    shared by all variants could be affected."""
    def __init__(self):
        super().__init__(logging.ERROR)
        self.msgs = []

    def emit(self, record):
        self.msgs.append(record.getMessage()[:200])


ERRLOG = _ErrorLog()
_console = logging.StreamHandler()  # same output as Python's default last-resort handler
_console.setLevel(logging.WARNING)
logging.getLogger().addHandler(_console)
logging.getLogger().addHandler(ERRLOG)


def write_traj(path, rows):
    from scipy.spatial.transform import Rotation
    with open(path, "w") as fh:
        for t, T in rows:
            q = Rotation.from_matrix(T[:3, :3]).as_quat()
            fh.write(f"{t:.6f} {T[0,3]:.6f} {T[1,3]:.6f} {T[2,3]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")


def sim3_segments(times, gens, poses, gt_t, gt_p, max_dt=0.02):
    """ATE for the whole run and per map generation (segment)."""
    times = np.asarray(times); P = np.asarray([T[:3, 3] for T in poses]); gens = np.asarray(gens)
    out = {}
    ia, ib = associate(times, gt_t, max_dt)
    if len(ia) >= 10 and np.ptp(P[ia], axis=0).max() > 1e-9:
        s, R, t = umeyama_sim3(P[ia], gt_p[ib])
        al = (s * (R @ P[ia].T)).T + t
        out["all"] = dict(ate=float(np.sqrt(np.mean(np.sum((al - gt_p[ib]) ** 2, 1)))), s=float(s), R=R, t=t,
                          n=int(len(ia)))
    for g in np.unique(gens):
        m = gens == g
        ia, ib = associate(times[m], gt_t, max_dt)
        if len(ia) >= 10 and np.ptp(P[m][ia], axis=0).max() > 1e-9:
            s, R, t = umeyama_sim3(P[m][ia], gt_p[ib])
            al = (s * (R @ P[m][ia].T)).T + t
            out[int(g)] = dict(ate=float(np.sqrt(np.mean(np.sum((al - gt_p[ib]) ** 2, 1)))), s=float(s), R=R, t=t,
                               n=int(len(ia)))
    return out


def evaluate_run(seq, rec, gt, sim, mesh_fn, recon_gt=None):
    """rec: per-frame records; returns metrics dict."""
    m = {}
    gens = [r["gen"] for r in rec["poses"]]
    times = [r["t"] for r in rec["poses"]]
    poses = [r["T"] for r in rec["poses"]]
    n_frames = rec["n_frames"]
    m["frames"] = n_frames
    m["tracked"] = len(poses)
    m["tracked_frac"] = len(poses) / max(n_frames, 1)
    m["reinit"] = int(rec["reinit"])
    if len(poses) < 10:
        m["failed"] = True
        return m
    seg = sim3_segments(times, gens, poses, gt["t"], gt["p"])
    main = "all" if m["reinit"] == 0 else None
    if main is None:   # longest segment (for diagnostics only; the run counts as a failure)
        cand = [(v["n"], k) for k, v in seg.items() if k != "all"]
        main = max(cand)[1] if cand else None
    if main is None or main not in seg:
        m["failed"] = True
        return m
    A = seg[main]
    m["ate_cm"] = 100 * A["ate"]
    m["sim3_scale"] = A["s"]
    m["sim3"] = {"s": A["s"], "R": np.asarray(A["R"]).tolist(), "t": np.asarray(A["t"]).tolist()}
    m["eval_segment"] = str(main)
    m["failed"] = bool(m["reinit"] > 0 or m["tracked_frac"] < 0.9)
    # depth (TUM): one similarity scale per map (the trajectory alignment's), every processed
    # frame. Without re-initialisation this is a single scale for the whole sequence; after a
    # re-initialisation each map segment uses the scale of its own trajectory alignment.
    if seq.kind == "tum" and rec["depth"]:
        dt, df, _ = seq.gt_depth_index()
        pairs_all, scales = [], []
        for g in sorted(set(rec["depth_gen"].values())):
            key = "all" if m["reinit"] == 0 else int(g)
            if key not in seg:
                continue
            pt = np.array(sorted(x for x in rec["depth"] if rec["depth_gen"][x] == g))
            ip, ig = associate(pt, dt, 0.02)
            for i, j in zip(ip, ig):
                pairs_all.append((rec["depth"][pt[i]].astype(np.float32) * seg[key]["s"], seq.gt_depth(df[j])))
        if pairs_all:
            dm = depth_metrics(pairs_all, 1.0)
            m.update(absrel=dm["absrel"], delta1=dm["delta1"], sc=dm["sc"], depth_frames=dm["frames"])
    # reconstruction (Replica)
    if recon_gt is not None and mesh_fn is not None:
        from eval_recon import recon_metrics
        mesh = mesh_fn()
        if mesh is not None and len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
            last = max(seg.keys(), key=lambda k: -1 if k == "all" else k)
            B = seg["all"] if m["reinit"] == 0 else seg.get(last, A)
            pts = np.asarray(mesh.sample_points_uniformly(number_of_points=200_000).points)
            pred = (B["s"] * (B["R"] @ pts.T)).T + B["t"]
            m.update(recon_metrics(pred, recon_gt))
            m["recon_sim3"] = {"s": float(B["s"]), "R": np.asarray(B["R"]).tolist(), "t": np.asarray(B["t"]).tolist()}
            m["mesh_vertices"] = int(len(mesh.vertices))
    return m


# (name, kind, value): see SynchronousPipeline.extra
EXTRA_FUSION = [("unc_c0.5", "unc", 0.5), ("unc_c2", "unc", 2.0), ("unc_c4", "unc", 4.0),
                ("unc_cinf", "unc", float("inf")), ("uni_c1", "uni", 1.0), ("uni_cinf", "uni", float("inf")),
                ("uni_z1.5", "depth", 1.5), ("uni_z2", "depth", 2.0), ("uni_z3", "depth", 3.0)]


def evaluate_oracle_run(seq, rec, mesh_fn, recon_gt=None, band_block=0):
    """Ground-truth-pose run: depth (TUM, Replica) and reconstruction (Replica) at scale 1.
    band_block > 0 (anchor-restriction test): also the depth metrics of the restricted blocks
    (frame index // band_block odd) and of the other blocks."""
    m = {"frames": rec["n_frames"], "tracked": len(rec["poses"]),
         "tracked_frac": len(rec["poses"]) / max(rec["n_frames"], 1), "tracker_reinit": int(rec["reinit"]),
         "reinit": 0, "failed": False, "oracle_poses": True}
    d = rec["oracle_depth"]
    if d:
        a_ = np.array(d)
        m.update(absrel=float(a_[:, 0].mean()), delta1=float(a_[:, 1].mean()), sc=float(a_[:, 2].std()),
                 depth_frames=int(len(d)))
        if band_block > 0:
            r = (a_[:, 3].astype(np.int64) // band_block) % 2 == 1
            for name, sel in (("restricted", r), ("unrestricted", ~r)):
                if sel.any():
                    m[f"absrel_{name}"] = float(a_[sel, 0].mean())
                    m[f"sc_{name}"] = float(a_[sel, 2].std())
                    m[f"frames_{name}"] = int(sel.sum())
    if recon_gt is not None and mesh_fn is not None:
        from eval_recon import recon_metrics
        mesh = mesh_fn()
        if mesh is not None and len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
            pts = np.asarray(mesh.sample_points_uniformly(number_of_points=200_000).points)
            m.update(recon_metrics(pts, recon_gt))
            m["mesh_vertices"] = int(len(mesh.vertices))
    return m


def evaluate_extra(pipe, metrics, recon_gt, oracle):
    """Reconstruction metrics of the additional fusion volumes (same transform as the main mesh)."""
    from eval_recon import recon_metrics
    if oracle:
        tf = lambda P: P
    else:
        S = metrics.get("recon_sim3")
        if S is None:
            return {}
        tf = lambda P: (S["s"] * (np.asarray(S["R"]) @ P.T)).T + np.asarray(S["t"])
    out = {}
    for name, mesh in pipe.extract_extra_meshes().items():
        if mesh is None or len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
            out[name] = None
            continue
        pts = np.asarray(mesh.sample_points_uniformly(number_of_points=200_000).points)
        out[name] = {**recon_metrics(tf(pts), recon_gt), "mesh_vertices": int(len(mesh.vertices))}
    return out


def make_scale_of(metrics):
    """Map-gauge -> metre scale per map generation (only runs without re-initialisation)."""
    s = metrics.get("sim3_scale")
    if s is None or metrics.get("reinit", 0) > 0:
        return lambda g: None
    return lambda g: float(s)


def load_gt(seq, out_dir):
    if seq.kind == "tum":
        t, p, _ = read_tum_traj(seq.gt_trajectory_path())
    else:
        path = os.path.join(out_dir, "gt_tum.txt")
        seq.write_gt_tum(path)
        t, p, _ = read_tum_traj(path)
    return {"t": t, "p": p}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--name", required=True, help="sequence key, e.g. fr1_desk or room0")
    ap.add_argument("--variants", default="cgssf")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--runs", default=os.path.join(ROOT, "runs"))
    ap.add_argument("--tag-suffix", default="", help="appended to the variant folder name (e.g. _dptlarge)")
    ap.add_argument("--depth-model", default=None)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--recon-gt", default=None, help="culled ground-truth mesh (Replica)")
    ap.add_argument("--save-mesh", default="", help="comma-separated variants whose seed-0 mesh is saved")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--config", default="", help="KEY=VALUE overrides of backend.config, comma-separated")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--preload", action="store_true", help="keep all decoded frames in RAM (faster, ~1-1.5 GB)")
    ap.add_argument("--oracle-poses", action="store_true", help="ground-truth camera poses (isolates alignment/fusion)")
    ap.add_argument("--extra-fusion", action="store_true", help="additional fusion volumes (trade-off curves)")
    ap.add_argument("--diag", action="store_true", help="per-frame errors of fit / output / best fit (gate analysis)")
    ap.add_argument("--unc-eval", action="store_true", help="per-pixel uncertainty evaluation (with --oracle-poses)")
    ap.add_argument("--band-variants", default="",
                    help="variants additionally run in the anchor-restriction test (ANCHOR_BAND=--band-value, "
                         "tag suffix + '_band') in the same process, sharing the cached network outputs")
    ap.add_argument("--band-value", type=float, default=0.1)
    ap.add_argument("--plan", default="",
                    help="several depth networks / settings in one process, sharing the cached keypoints, matches, and "
                         "person boxes: entries 'MODEL|SUFFIX|VARIANTS|BAND_VARIANTS|KEY=VAL,...' separated by ';' "
                         "(MODEL empty: --depth-model); replaces --variants, --band-variants, --tag-suffix, --depth-model")
    a = ap.parse_args()

    # jobs: (depth model, tag suffix, variant, anchor-restriction test, configuration overrides of this job)
    if a.plan:
        jobs = []
        for entry in [e for e in a.plan.split(";") if e.strip()]:
            model, suffix, vs, bvs, cfg = (entry.split("|") + [""] * 5)[:5]
            kv = dict(x.split("=") for x in cfg.split(",") if x)
            jobs += [(model or a.depth_model, suffix, v, False, kv) for v in vs.split(",") if v]
            jobs += [(model or a.depth_model, suffix, v, True, kv) for v in bvs.split(",") if v]
    else:
        jobs = [(a.depth_model, a.tag_suffix, v, False, {}) for v in a.variants.split(",") if v]
        jobs += [(a.depth_model, a.tag_suffix, v, True, {}) for v in a.band_variants.split(",") if v]
    for job in jobs:
        if job[2] not in VARIANTS:
            raise SystemExit(f"unknown variant {job[2]}")
    seeds = [int(s) for s in a.seeds.split(",") if s != ""]
    save_mesh = set(x for x in a.save_mesh.split(",") if x)

    config = Config()
    for kv in [x for x in a.config.split(",") if x]:
        k, v = kv.split("=")
        old = getattr(config, k)
        setattr(Config, k, type(old)(v) if not isinstance(old, bool) else v.lower() in ("1", "true", "yes"))
    seq = RecordedSequence.open(a.root, a.width)
    config.set_camera(seq.K, *seq.size)
    n = len(seq) if a.max_frames <= 0 else min(len(seq), a.max_frames)
    env = environment()
    chash = code_hash()
    # Accuracy runs are made deterministic (cuDNN autotuning can pick different
    # convolution algorithms in different processes); runtime is measured
    # separately in the live configuration (tip/runtime_bench.py).
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"[{a.name}] {seq.kind} {n} frames, K={np.round(seq.K, 2).tolist()} size={seq.size}", flush=True)
    t_load = time.time()
    if a.preload:                         # keep decoded frames in RAM (shared by all runs)
        frames = [seq.frame(i) for i in range(n)]
    else:
        frames = None
    print(f"[{a.name}] decoded frames in {time.time() - t_load:.1f}s", flush=True)
    nets = NetworkBundle(config, depth_model=jobs[0][0] if jobs else a.depth_model, cache=not a.no_cache)
    gt_poses = gt_pose_provider(seq) if a.oracle_poses else {}
    gt_depth = None
    if a.oracle_poses or a.diag:
        gt_depth = GTDepth(seq, a.recon_gt, poses=gt_poses or None) if (seq.kind == "tum" or a.recon_gt) else None
    if a.oracle_poses:
        print(f"[{a.name}] ground-truth poses for {len(gt_poses)}/{n} frames", flush=True)
    recon_gt = None
    if a.recon_gt:
        from eval_recon import sample_mesh
        recon_gt = sample_mesh(a.recon_gt, 200_000)

    def set_config(kv):
        """Apply KEY=VALUE overrides of one job to Config; returns the previous values."""
        old = {}
        for k, v in kv.items():
            o = getattr(Config, k)
            old[k] = o
            setattr(Config, k, type(o)(v) if not isinstance(o, bool) else v.lower() in ("1", "true", "yes"))
        return old

    band0 = float(getattr(Config, "ANCHOR_BAND", 0.0))
    for model, suffix, variant, banded, kv in jobs:
        nets.set_depth_model(model)
        old_cfg = set_config(kv)
        Config.ANCHOR_BAND = a.band_value if banded else band0
        tag = variant + suffix + ("_band" if banded else "")
        overrides = ",".join(x for x in (a.config, ",".join(f"{k}={v}" for k, v in kv.items()),
                                         f"ANCHOR_BAND={a.band_value:g}" if banded else "") if x)
        for seed in seeds:
            out = os.path.join(a.runs, tag, a.name, f"seed{seed}")
            if os.path.exists(os.path.join(out, "metrics.json")) and not a.force:
                print(f"[{a.name}] skip {tag} seed{seed} (done)", flush=True)
                continue
            os.makedirs(out, exist_ok=True)
            gt = load_gt(seq, out)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            extra = EXTRA_FUSION if (a.extra_fusion and variant == "cgssf" and a.recon_gt) else None
            pipe = SynchronousPipeline(config, nets, variant=variant, seed=seed, oracle_poses=a.oracle_poses,
                                       extra_fusion=extra)
            rec = {"poses": [], "depth": {}, "depth_gen": {}, "n_frames": n, "reinit": 0, "diag": [],
                   "oracle_depth": [], "unc": []}
            urng = np.random.default_rng(10_000 + seed)       # pixel sampling of --unc-eval only
            rows, frows = [], []
            t0 = time.time()
            for i in range(n):
                t = seq.timestamps[i]
                tw = time.perf_counter()
                gp = gt_poses.get(i) if a.oracle_poses else None
                gd = None
                if a.oracle_poses and gp is not None and gt_depth is not None:
                    gd = gt_depth(i)
                gdt = torch.from_numpy(gd) if (gd is not None and variant == "oracle_align") else None
                r = pipe.step(frames[i] if frames is not None else seq.frame(i), t, i, gt_pose=gp, gt_depth=gdt)
                if a.oracle_poses and r.z_eval is not None and gd is not None:
                    zh = r.z_eval.float().cpu().numpy()
                    e = frame_depth_errors(zh, gd)
                    if e is not None:
                        rec["oracle_depth"].append((*e, i))
                    lm = getattr(pipe.aligner, "last_maps", None)
                    if a.unc_eval and lm is not None:
                        smp = unc_frame_samples(zh, lm[0].float().cpu().numpy(), lm[1].float().cpu().numpy(),
                                                float(pipe.aligner.f.sigma2 or 0.0), gd, urng)
                        if smp is not None:
                            rec["unc"].append(smp)
                if r.diag:
                    rec["diag"].append({"i": i, "gen": r.map_generation, "status": r.align_status, **r.diag})
                wall = 1e3 * (time.perf_counter() - tw)
                if r.pose is not None:
                    rec["poses"].append({"t": t, "T": r.pose, "gen": r.map_generation})
                if r.z_eval is not None and seq.kind == "tum" and not a.oracle_poses:
                    rec["depth"][t] = r.z_eval.half().cpu().numpy()
                    rec["depth_gen"][t] = r.map_generation
                th = pipe.aligner.theta
                fr = {"i": i, "t": f"{t:.6f}", "gen": r.map_generation, "track": r.status,
                      "align": r.align_status, "anchors": r.n_anchors,
                      "s": "" if th is None else f"{th[0]:.6g}", "b": "" if th is None else f"{th[1]:.6g}"}
                if a.oracle_poses or a.diag:
                    tf = r.diag.get("theta_fit") if r.diag else None
                    fr.update(fit_s="" if tf is None else f"{tf[0]:.6g}", fit_b="" if tf is None else f"{tf[1]:.6g}",
                              kappa="" if not r.diag else f"{r.diag.get('kappa', float('nan')):.4g}",
                              nis="" if not r.diag else f"{r.diag.get('d2', float('nan')):.4g}")
                frows.append(fr)
                rows.append({"t": f"{t:.6f}", "wall_ms": round(wall, 3),
                             **{k: round(v, 3) for k, v in r.timing_ms.items()}})
            run_s = time.time() - t0
            rec["reinit"] = pipe.n_reinit
            peak = torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else None
            write_traj(os.path.join(out, "traj_tum.txt"), [(p["t"], p["T"]) for p in rec["poses"]])
            keys = ["t", "wall_ms", "seg", "depth", "track", "align", "map", "fuse"]
            with open(os.path.join(out, "timing.csv"), "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
            with open(os.path.join(out, "frames.csv"), "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(frows[0])); w.writeheader(); w.writerows(frows)
            mesh_cache = {}

            def mesh_fn():
                if "m" not in mesh_cache:
                    mesh_cache["m"] = pipe.extract_mesh()
                return mesh_cache["m"]
            te = time.time()
            if a.oracle_poses:
                band_block = config.ANCHOR_BAND_BLOCK if getattr(config, "ANCHOR_BAND", 0.0) > 0 else 0
                metrics = evaluate_oracle_run(seq, rec, mesh_fn, recon_gt, band_block)
            else:
                metrics = evaluate_run(seq, rec, gt, None, mesh_fn, recon_gt)
            if pipe.extra and recon_gt is not None:
                metrics["extra"] = evaluate_extra(pipe, metrics, recon_gt, a.oracle_poses)
            if (a.oracle_poses or a.diag) and gt_depth is not None and rec["diag"]:
                scale_of = (lambda g: 1.0) if a.oracle_poses else make_scale_of(metrics)
                nu_of = lambda i: (nets._nu[i].float().numpy() if i in nets._nu else None)
                recs, summ = gate_diagnostics(rec["diag"], nu_of, gt_depth, scale_of, pipe.aligner.cfg.cgssf,
                                              config.CGSSF_KAPPA_MIN)
                metrics["gate_diag"] = summ
                with open(os.path.join(out, "diag.csv"), "w", newline="") as fh:
                    if recs:
                        w = csv.DictWriter(fh, fieldnames=list(recs[0])); w.writeheader(); w.writerows(recs)
            if rec["unc"]:
                metrics["unc_eval"] = sparsification(rec["unc"], seed=seed)
            metrics["eval_s"] = time.time() - te
            if variant in save_mesh and seed == 0:
                mesh = mesh_fn()
                if mesh is not None and len(mesh.vertices):
                    import open3d as o3d
                    o3d.io.write_triangle_mesh(os.path.join(out, "mesh.ply"), mesh)
            meta = dict(dataset=seq.kind, sequence=a.name, root=os.path.abspath(a.root), variant=variant,
                        variant_description=VARIANTS[variant], tag=tag, seed=seed,
                        depth_model=nets.depth.model_type, dry_run=False, synchronous=True,
                        network_outputs_cached=not a.no_cache, frames=n, tracked=len(rec["poses"]),
                        reinit=pipe.n_reinit, loop_events=pipe.loop_events,
                        status_counts=pipe.status_counts, intrinsics=seq.K.tolist(), size=list(seq.size),
                        config_overrides=overrides, code_hash=chash, git_commit=None, env=env,
                        gpu=env.get("gpu"), peak_gpu_mb=peak, run_s=run_s, oracle_poses=a.oracle_poses,
                        extra_fusion=[e[:3] for e in (extra or [])], diag=a.diag, unc_eval=a.unc_eval,
                        tsdf=pipe.tsdf.stats(), finished=time.strftime("%Y-%m-%d %H:%M:%S"))
            if ERRLOG.msgs:
                raise RuntimeError(f"[{a.name}] {variant} seed{seed} not saved, errors were logged: {ERRLOG.msgs[:3]}")
            json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
            json.dump(metrics, open(os.path.join(out, "metrics.json"), "w"), indent=1)
            msg = " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()
                           if k in ("ate_cm", "absrel", "delta1", "sc", "acc_cm", "comp_cm", "comp_ratio", "fscore",
                                    "tracked_frac", "reinit", "failed"))
            print(f"[{a.name}] {tag} seed{seed}: {msg}  ({run_s:.0f}s, {n / run_s:.1f} fps, "
                  f"misses={nets.features.misses}, peak={peak:.0f}MB)", flush=True)
            del pipe, rec
            torch.cuda.empty_cache()
        for k, v in old_cfg.items():
            setattr(Config, k, v)


if __name__ == "__main__":
    main()
