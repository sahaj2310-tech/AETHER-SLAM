"""
Aggregate measured runs into the numbers of Tables III-VI (and the per-sequence
supplementary tables).

  python tip/collect_results.py --runs <runs> --out results [--runtime results/runtime/runtime.json]

runs/<tag>/<sequence>/seed<k>/ are written by tip/run_suite.py (metrics.json is
computed in-process from the run's trajectory, depth and mesh; meta.json holds
the provenance). Integrity rules:
  * dry runs are skipped; only runs with metrics.json count;
  * a run is a tracking failure (X) if the map was re-initialised or < 90 % of the
    frames were tracked; failed runs never enter a trajectory-error average (their
    depth / reconstruction metrics, evaluated per map segment, do);
  * per sequence, a value is the mean over the valid seeds; a table cell is
    filled only when every sequence it averages has a valid value (else empty);
  * results/provenance.csv lists the run folder, code hash, GPU and metrics of
    every run behind every number.
"""
import argparse
import csv
import glob
import json
import os

import numpy as np

from collect_extra import (bootstrap_ci, derive_uniform, gate_pooled, gate_scale, holm, rank_biserial, strata, tradeoff,
                           unc_summary)

FR1 = ["fr1_360", "fr1_desk", "fr1_desk2", "fr1_floor", "fr1_plant", "fr1_room", "fr1_rpy", "fr1_teddy", "fr1_xyz"]
DYN = ["fr3_walking_xyz", "fr3_walking_halfsphere", "fr3_sitting_xyz"]
REPLICA = ["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]
# TUM fr3 structure vs. texture (ground-truth poses only); TUM = fr1 and these together
STX = ["fr3_str_tex_far", "fr3_str_tex_near", "fr3_str_notex_far", "fr3_str_notex_near",
       "fr3_nostr_tex_far", "fr3_nostr_tex_near_loop", "fr3_nostr_notex_far", "fr3_nostr_notex_near_loop"]
TUM = FR1 + STX
ABLATION = ["proto_depthls", "proto_ema", "inv_ls", "inv_irls", "irls_ema",
            "cgssf_timeq", "cgssf_norank1", "cgssf_nogate", "cgssf_fixedgate", "cgssf_uniform", "cgssf"]
# ground-truth-pose runs (tag suffix _oracle): variant, row label
ORACLE = [("proto_ema_oracle", "(b)"), ("inv_irls_oracle", "(d)"), ("irls_ema_oracle", "(e)"),
          ("cgssf_timeq_oracle", "(f)"), ("cgssf_norank1_oracle", "(g)"), ("cgssf_nogate_oracle", "(h)"),
          ("cgssf_fixedgate_oracle", "(i)"), ("cgssf_uniform_oracle", "(j)"), ("cgssf_oracle", "(k)"),
          ("oracle_align_oracle", "oracle")]
# anchor-restriction test (tag suffix _oracle_band): in every second block of 50 frames only the anchors
# within +-10% of the median network output reach the alignment; TUM fr1 and fr3 structure vs. texture
BAND = [("proto_ema_oracle_band", "(b)"), ("inv_irls_oracle_band", "(d)"), ("irls_ema_oracle_band", "(e)"),
        ("cgssf_norank1_oracle_band", "(g)"), ("cgssf_uniform_oracle_band", "(j)"), ("cgssf_oracle_band", "(k)")]
# alignment designs of the closest methods on the same inputs (ground-truth poses): normal tag, tag of the
# anchor-restriction test (or None), prior and network, estimator
DESIGNS = [("inv_irls_oracle", "inv_irls_oracle_band", "relative, DPT-Hybrid", "(d) per-frame IRLS"),
           ("irls_ema_oracle", "irls_ema_oracle_band", "relative, DPT-Hybrid", "(e) per-frame IRLS + EMA"),
           ("win_irls_oracle", "win_irls_oracle_band", "relative, DPT-Hybrid", "(l) sliding-window joint fit"),
           ("cgssf_oracle", "cgssf_oracle_band", "relative, DPT-Hybrid", "(k) CG-SSF"),
           ("inv_irls_dav2s_oracle", None, "relative, DAv2-S", "(d) per-frame IRLS"),
           ("cgssf_dav2s_oracle", "cgssf_dav2s_oracle_band", "relative, DAv2-S", "(k) CG-SSF"),
           ("metric_raw_dav2m_oracle", None, "metric, DAv2-S", "none"),
           ("scale_kf_dav2m_oracle", "scale_kf_dav2m_oracle_band", "metric, DAv2-S", "(m) scale-only filter"),
           ("cgssf_dav2m_oracle", "cgssf_dav2m_oracle_band", "metric, DAv2-S", "(k) CG-SSF")]
# held-out tuning of the designs (fr2/xyz, ground-truth poses): tag -> (parameter, value)
TUNE = [(f"win_irls_W{w}_tune", "W", w) for w in (5, 10, 20, 40)] + \
       [(f"scale_kf_dav2m_eta{e}_tune", "eta", float(e)) for e in ("0.071", "0.284", "1.136")]
# process noise scaled by 1/4 and 4 (tip run_suite --config overrides), ground-truth poses, TUM fr1
SENS = [("cgssf_eta025_oracle", "eta/4"), ("cgssf_oracle", "calibrated"), ("cgssf_eta4_oracle", "4 eta")]
BACKBONES = {"cgssf": "DPT_Hybrid", "cgssf_dptlarge": "DPT_Large", "cgssf_midassmall": "MiDaS_small",
             "cgssf_dav2s": "DepthAnythingV2_Small"}
LABEL = {"proto_depthls": "(a) Depth-domain LS (prototype fit)", "proto_ema": "(b) (a) + EMA + reuse (prototype)",
         "inv_ls": "(c) Inverse-domain LS", "inv_irls": "(d) Inverse-domain IRLS", "irls_ema": "(e) (d) + EMA + reuse",
         "cgssf_timeq": "(f) CG-SSF, time-indexed process noise",
         "cgssf_norank1": "(g) CG-SSF w/o rank-one", "cgssf_nogate": "(h) CG-SSF w/o gate",
         "cgssf_fixedgate": "(i) CG-SSF, fixed gate threshold",
         "cgssf_uniform": "(j) CG-SSF, uniform weights", "cgssf": "(k) CG-SSF, weighted (full)",
         "cgssf_nomask": "Full system w/o person masking"}
METRICS = ("ate_cm", "absrel", "delta1", "sc", "acc_cm", "comp_cm", "comp_ratio", "precision", "fscore", "chamfer_cm",
           "absrel_restricted", "absrel_unrestricted")


def load_runs(root, seeds=None):
    per = {}
    for mpath in glob.glob(os.path.join(root, "*", "*", "seed*", "metrics.json")):
        run = os.path.dirname(mpath)
        tag, seq = run.split(os.sep)[-3], run.split(os.sep)[-2]
        meta = json.load(open(os.path.join(run, "meta.json")))
        if meta.get("dry_run"):
            continue
        if seeds is not None and meta.get("seed") not in seeds:
            continue
        m = json.load(open(mpath))
        m.update(run=run, seed=meta.get("seed"), code_hash=meta.get("code_hash"), gpu=meta.get("gpu"),
                 depth_model=meta.get("depth_model"), peak_gpu_mb=meta.get("peak_gpu_mb"),
                 status_counts=meta.get("status_counts", {}), loop_events=meta.get("loop_events"))
        per.setdefault((tag, seq), []).append(m)
    return per


def seq_value(per, tag, seq, key):
    """Mean over seeds. Trajectory error only counts runs without tracking failure; depth and
    reconstruction metrics of a failed run are evaluated per map segment (see run_suite.py)
    and do count."""
    rs = [r for r in per.get((tag, seq), [])
          if r.get(key) is not None and not (key == "ate_cm" and r.get("failed"))]
    return float(np.mean([r[key] for r in rs])) if rs else None


def cell(per, tag, seqs, key, scale=1.0, nd=2):
    vals = [seq_value(per, tag, q, key) for q in seqs]
    if any(v is None for v in vals) or not vals:
        return None
    return round(float(np.mean(vals)) * scale, nd)


def failures(per, tag, seq):
    rs = per.get((tag, seq), [])
    return sum(1 for r in rs if r.get("failed")), len(rs)


def paired_test(per, tag_a, tag_b, seqs, key):
    """Paired Wilcoxon signed-rank test over sequences (seed-averaged values)."""
    from scipy.stats import wilcoxon
    a = [seq_value(per, tag_a, q, key) for q in seqs]
    b = [seq_value(per, tag_b, q, key) for q in seqs]
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 5:
        return None
    x, y = np.array(pairs).T
    try:
        p = float(wilcoxon(x, y).pvalue) if np.any(x != y) else 1.0
    except ValueError:
        p = None
    better = (x > y) if key in ("delta1", "fscore", "comp_ratio", "precision") else (x < y)
    ties = int((x == y).sum())
    sign = 1.0 if key in ("delta1", "fscore", "comp_ratio", "precision") else -1.0
    return dict(n=len(pairs), mean_a=float(x.mean()), mean_b=float(y.mean()), mean_diff=float((x - y).mean()),
                wins_a=int(better.sum()), ties=ties, losses_a=int(len(pairs) - better.sum() - ties), p=p,
                ci=bootstrap_ci(x - y), r_rb=rank_biserial(sign * (x - y)))


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0])
    for r in rows[1:]:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--runtime", default=None, help="runtime benchmark json (Section VI-H)")
    ap.add_argument("--datasets", default=None, help="(unused; kept for compatibility)")
    ap.add_argument("--seeds", default=None, help="comma-separated seeds to include (default: all)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    per = load_runs(a.runs, None if a.seeds is None else {int(x) for x in a.seeds.split(",")})
    # (i) uniform weights = extra fusion volume of the CG-SSF runs (same depth, poses and frames)
    derive_uniform(per, "cgssf", "cgssf_uniform")
    derive_uniform(per, "cgssf_oracle", "cgssf_uniform_oracle")
    derive_uniform(per, "cgssf_oracle_band", "cgssf_uniform_oracle_band")

    # ---- full-system ablation (supplementary material): TUM fr1 (depth) and Replica (reconstruction)
    t3 = []
    for v in ABLATION:
        t3.append(dict(variant=v, label=LABEL[v],
                       TUM_AbsRel_pct=cell(per, v, FR1, "absrel", 100, 1), TUM_delta1_pct=cell(per, v, FR1, "delta1", 100, 1),
                       TUM_SC_x1e2=cell(per, v, FR1, "sc", 100, 1), TUM_ATE_cm=cell(per, v, FR1, "ate_cm", 1, 1),
                       Replica_Acc_cm=cell(per, v, REPLICA, "acc_cm", 1, 2), Replica_Comp_cm=cell(per, v, REPLICA, "comp_cm", 1, 2),
                       Replica_CR_pct=cell(per, v, REPLICA, "comp_ratio", 1, 1), Replica_F_pct=cell(per, v, REPLICA, "fscore", 1, 1),
                       Replica_ATE_cm=cell(per, v, REPLICA, "ate_cm", 1, 2),
                       failures="; ".join(f"{q}:{failures(per, v, q)[0]}/{failures(per, v, q)[1]}"
                                          for q in FR1 + REPLICA if failures(per, v, q)[0])))
    # ---- Table V: full system on Replica
    t4 = [dict(method="AETHER-SLAM (ours)", Acc_cm=cell(per, "cgssf", REPLICA, "acc_cm"),
               Comp_cm=cell(per, "cgssf", REPLICA, "comp_cm"), CR_pct=cell(per, "cgssf", REPLICA, "comp_ratio", 1, 1),
               F_pct=cell(per, "cgssf", REPLICA, "fscore", 1, 1), ATE_cm=cell(per, "cgssf", REPLICA, "ate_cm"))]
    # ---- ATE on TUM, static fr1 average + dynamic fr3 (supplementary material)
    t5 = []
    for v in ("cgssf_nomask", "cgssf"):
        row = dict(variant=v, label=LABEL[v], fr1_avg_cm=cell(per, v, FR1, "ate_cm", 1, 1))
        for q in DYN:
            nf, nt = failures(per, v, q)
            val = cell(per, v, [q], "ate_cm", 1, 1)
            row[q] = "X" if (nt and nf == nt) else val
        t5.append(row)
    # ---- per-sequence tables (supplementary)
    perseq = []
    for (tag, seq), rs in sorted(per.items()):
        row = dict(tag=tag, sequence=seq, runs=len(rs), failed=sum(1 for r in rs if r.get("failed")))
        for k in METRICS:
            vals = [r[k] for r in rs if r.get(k) is not None and not (k == "ate_cm" and r.get("failed"))]
            row[k] = round(float(np.mean(vals)), 4) if vals else None
            if len(vals) > 1:
                row[k + "_std"] = round(float(np.std(vals)), 4)
        perseq.append(row)
    # ---- depth backbones (supplementary material): TUM fr1 AbsRel of the full system per depth model
    backbones = {}
    for (tag, seq), rs in per.items():
        if tag in BACKBONES:
            backbones.setdefault(tag, set()).add(seq)
    # the other networks were run with seed 0 only, so every row of this comparison uses seed 0
    per0 = {k: [r for r in rs if r.get("seed") == 0] for k, rs in per.items()}
    t6acc = [dict(tag=t, depth_model=(per[(t, sorted(s)[0])][0].get("depth_model")), seeds="0",
                  TUM_AbsRel_pct=cell(per0, t, FR1, "absrel", 100, 1), TUM_delta1_pct=cell(per0, t, FR1, "delta1", 100, 1),
                  TUM_ATE_cm=cell(per0, t, FR1, "ate_cm", 1, 1), tracking_failures=sum(failures(per0, t, q)[0] for q in FR1))
             for t, s in sorted(backbones.items())]
    # ---- significance of the key comparisons
    tests = {}
    for b in ("proto_ema", "proto_depthls", "irls_ema", "inv_irls", "inv_ls", "cgssf_uniform", "cgssf_norank1", "cgssf_nogate",
              "cgssf_timeq", "cgssf_fixedgate"):
        for key, seqs in (("absrel", FR1), ("sc", FR1), ("delta1", FR1), ("ate_cm", FR1),
                          ("fscore", REPLICA), ("acc_cm", REPLICA), ("comp_cm", REPLICA)):
            tests[f"cgssf_vs_{b}:{key}"] = paired_test(per, "cgssf", b, seqs, key)
    # ground-truth-pose experiment
    oracle_rows = []
    for tag, lab in ORACLE:
        oracle_rows.append(dict(tag=tag, label=lab,
                                TUM_AbsRel_pct=cell(per, tag, FR1, "absrel", 100, 1), TUM_delta1_pct=cell(per, tag, FR1, "delta1", 100, 1),
                                TUM_SC_x1e2=cell(per, tag, FR1, "sc", 100, 1),
                                STX_AbsRel_pct=cell(per, tag, STX, "absrel", 100, 1),
                                STX_delta1_pct=cell(per, tag, STX, "delta1", 100, 1),
                                STX_SC_x1e2=cell(per, tag, STX, "sc", 100, 1),
                                TUM17_AbsRel_pct=cell(per, tag, TUM, "absrel", 100, 1),
                                TUM17_SC_x1e2=cell(per, tag, TUM, "sc", 100, 1),
                                Replica_AbsRel_pct=cell(per, tag, REPLICA, "absrel", 100, 1),
                                Replica_delta1_pct=cell(per, tag, REPLICA, "delta1", 100, 1),
                                Replica_SC_x1e2=cell(per, tag, REPLICA, "sc", 100, 1),
                                Replica_Acc_cm=cell(per, tag, REPLICA, "acc_cm", 1, 2), Replica_Comp_cm=cell(per, tag, REPLICA, "comp_cm", 1, 2),
                                Replica_CR_pct=cell(per, tag, REPLICA, "comp_ratio", 1, 1),
                                Replica_F_pct=cell(per, tag, REPLICA, "fscore", 1, 1)))
    for b in ("proto_ema_oracle", "inv_irls_oracle", "irls_ema_oracle", "cgssf_timeq_oracle", "cgssf_norank1_oracle",
              "cgssf_nogate_oracle", "cgssf_fixedgate_oracle", "cgssf_uniform_oracle", "oracle_align_oracle"):
        for key, seqs, name in (("absrel", FR1, "tum_absrel"), ("sc", FR1, "tum_sc"), ("delta1", FR1, "tum_delta1"),
                                ("absrel", STX, "stx_absrel"), ("sc", STX, "stx_sc"), ("delta1", STX, "stx_delta1"),
                                ("absrel", TUM, "tum17_absrel"), ("sc", TUM, "tum17_sc"),
                                ("absrel", REPLICA, "rep_absrel"), ("sc", REPLICA, "rep_sc"),
                                ("fscore", REPLICA, "fscore"), ("acc_cm", REPLICA, "acc_cm"), ("comp_cm", REPLICA, "comp_cm")):
            tests[f"cgssf_oracle_vs_{b}:{name}"] = paired_test(per, "cgssf_oracle", b, seqs, key)
    # real-data sensitivity to the calibrated process noise (ground-truth poses, TUM fr1, seed 0; unadjusted tests)
    sens_rows = []
    for tag, lab in SENS:
        st = {}
        for q in FR1:
            for r in per.get((tag, q), []):
                for k, v in (r.get("status_counts") or {}).items():
                    st[k] = st.get(k, 0) + v
        dec = sum(st.get(k, 0) for k in ("full", "partial", "rejected", "reset"))
        sens_rows.append(dict(tag=tag, label=lab, TUM_AbsRel_pct=cell(per, tag, FR1, "absrel", 100, 1),
                              TUM_SC_x1e2=cell(per, tag, FR1, "sc", 100, 1),
                              TUM_rejected_pct=round(100 * (st.get("rejected", 0) + st.get("reset", 0)) / dec, 1) if dec else None))
        if tag != "cgssf_oracle":
            for key, name in (("absrel", "tum_absrel"), ("sc", "tum_sc")):
                tests[f"cgssf_oracle_vs_{tag}:{name}"] = paired_test(per, "cgssf_oracle", tag, FR1, key)
    # anchor-restriction test: depth over all frames and over the restricted blocks (TUM, 17 sequences)
    band_rows = []
    for tag, lab in BAND:
        band_rows.append(dict(tag=tag, label=lab, TUM17_AbsRel_pct=cell(per, tag, TUM, "absrel", 100, 1),
                              TUM17_SC_x1e2=cell(per, tag, TUM, "sc", 100, 1),
                              Restricted_AbsRel_pct=cell(per, tag, TUM, "absrel_restricted", 100, 1),
                              Unrestricted_AbsRel_pct=cell(per, tag, TUM, "absrel_unrestricted", 100, 1),
                              Replica_Restricted_AbsRel_pct=cell(per, tag, REPLICA, "absrel_restricted", 100, 1),
                              Replica_AbsRel_pct=cell(per, tag, REPLICA, "absrel", 100, 1),
                              Replica_Acc_cm=cell(per, tag, REPLICA, "acc_cm", 1, 2), Replica_Comp_cm=cell(per, tag, REPLICA, "comp_cm", 1, 2),
                              Replica_CR_pct=cell(per, tag, REPLICA, "comp_ratio", 1, 1), Replica_F_pct=cell(per, tag, REPLICA, "fscore", 1, 1)))
        if tag not in ("cgssf_oracle_band", "cgssf_uniform_oracle_band"):
            for key, name in (("absrel", "band_absrel"), ("absrel_restricted", "band_absrel_restricted"), ("sc", "band_sc")):
                tests[f"cgssf_oracle_band_vs_{tag}:{name}"] = paired_test(per, "cgssf_oracle_band", tag, TUM, key)
            tests[f"cgssf_oracle_band_vs_{tag}:rep_band_absrel_restricted"] = paired_test(per, "cgssf_oracle_band", tag, REPLICA, "absrel_restricted")
    # alignment designs of the closest methods (same inputs, ground-truth poses)
    design_rows = []
    for tag, btag, prior, est in DESIGNS:
        row = dict(tag=tag, band_tag=btag, prior=prior, estimator=est,
                   TUM17_AbsRel_pct=cell(per, tag, TUM, "absrel", 100, 1), TUM17_delta1_pct=cell(per, tag, TUM, "delta1", 100, 1),
                   TUM17_SC_x1e2=cell(per, tag, TUM, "sc", 100, 1),
                   FR1_AbsRel_pct=cell(per, tag, FR1, "absrel", 100, 1), STX_AbsRel_pct=cell(per, tag, STX, "absrel", 100, 1),
                   Replica_AbsRel_pct=cell(per, tag, REPLICA, "absrel", 100, 1), Replica_F_pct=cell(per, tag, REPLICA, "fscore", 1, 1))
        if btag:
            row.update(TUM17_Restricted_AbsRel_pct=cell(per, btag, TUM, "absrel_restricted", 100, 1),
                       TUM17_Unrestricted_AbsRel_pct=cell(per, btag, TUM, "absrel_unrestricted", 100, 1),
                       Replica_Restricted_AbsRel_pct=cell(per, btag, REPLICA, "absrel_restricted", 100, 1))
        design_rows.append(row)
    for a_, b_ in (("cgssf_oracle", "win_irls_oracle"), ("cgssf_dav2s_oracle", "scale_kf_dav2m_oracle"),
                   ("cgssf_oracle", "scale_kf_dav2m_oracle"), ("scale_kf_dav2m_oracle", "cgssf_dav2m_oracle"),
                   ("cgssf_dav2s_oracle", "inv_irls_dav2s_oracle"), ("scale_kf_dav2m_oracle", "metric_raw_dav2m_oracle"),
                   ("win_irls_oracle", "inv_irls_oracle"), ("win_irls_oracle", "irls_ema_oracle")):
        for key, seqs, name in (("absrel", TUM, "tum17_absrel"), ("absrel", FR1, "tum_absrel"), ("absrel", STX, "stx_absrel"),
                                ("absrel", REPLICA, "rep_absrel"), ("sc", TUM, "tum17_sc")):
            tests[f"design:{a_}_vs_{b_}:{name}"] = paired_test(per, a_, b_, seqs, key)
        ba, bb = a_ + "_band", b_ + "_band"
        if (ba, TUM[0]) in per and (bb, TUM[0]) in per:
            tests[f"design:{ba}_vs_{bb}:tum17_restricted"] = paired_test(per, ba, bb, TUM, "absrel_restricted")
        if (ba, REPLICA[0]) in per and (bb, REPLICA[0]) in per:
            tests[f"design:{ba}_vs_{bb}:rep_restricted"] = paired_test(per, ba, bb, REPLICA, "absrel_restricted")
    # the five primary comparisons fixed before the runs (results/README.md, 2026-10-21): Holm over these five
    primary = ["design:cgssf_oracle_vs_win_irls_oracle:tum17_absrel",
               "design:cgssf_oracle_band_vs_win_irls_oracle_band:tum17_restricted",
               "design:cgssf_dav2s_oracle_vs_scale_kf_dav2m_oracle:tum17_absrel",
               "design:cgssf_dav2s_oracle_band_vs_scale_kf_dav2m_oracle_band:tum17_restricted",
               "design:scale_kf_dav2m_oracle_vs_cgssf_dav2m_oracle:tum17_absrel"]
    fam_design = {k: tests[k]["p"] for k in primary if tests.get(k) and tests[k].get("p") is not None}
    for k, ph in holm(fam_design).items():
        tests[k]["p_holm_design"] = ph
    tune_rows = []
    for tag, par, val in TUNE:
        v = seq_value(per, tag, "fr2_xyz", "absrel")
        if v is not None:
            tune_rows.append(dict(tag=tag, parameter=par, value=val, fr2xyz_AbsRel_pct=round(100 * v, 2),
                                  fr2xyz_SC_x1e2=round(100 * seq_value(per, tag, "fr2_xyz", "sc"), 2)))
    # reconstruction: unit-weight fusion of (k) (= (j)) against (g); only the rank-one update differs
    for key in ("fscore", "acc_cm", "comp_ratio", "comp_cm"):
        tests[f"cgssf_uniform_oracle_band_vs_cgssf_norank1_oracle_band:rep_{key}"] = paired_test(
            per, "cgssf_uniform_oracle_band", "cgssf_norank1_oracle_band", REPLICA, key)
    # share of partially observable frames (kappa < kappa_min among frames with a fit), CG-SSF, ground-truth poses
    kappa_share = {}
    for name, seqs in (("fr1", FR1), ("stx", STX), ("str", STX[:4]), ("nostr", STX[4:]), ("replica", REPLICA)):
        ks = []
        for q in seqs:
            for r in per.get(("cgssf_oracle", q), []):
                with open(os.path.join(r["run"], "frames.csv")) as fh:
                    for row in csv.DictReader(fh):
                        try:
                            k = float(row.get("kappa") or "nan")
                        except ValueError:
                            continue
                        if np.isfinite(k):
                            ks.append(k)
        if ks:
            ks = np.asarray(ks)
            kappa_share[name] = dict(frames=int(ks.size), below_kappa_min=float(np.mean(ks < 0.01)), median=float(np.median(ks)))
    # Holm-Bonferroni within the two families of comparisons reported in the text
    fam_normal = {k: v["p"] for k, v in tests.items() if v and not k.startswith("cgssf_oracle")
                  and k.split(":")[0] in ("cgssf_vs_proto_ema", "cgssf_vs_irls_ema", "cgssf_vs_cgssf_timeq")
                  and k.split(":")[1] in ("absrel", "sc", "ate_cm", "fscore", "acc_cm", "comp_cm")}
    fam_oracle = {k: v["p"] for k, v in tests.items() if v and k.startswith("cgssf_oracle")
                  and k.split(":")[0] in ("cgssf_oracle_vs_irls_ema_oracle", "cgssf_oracle_vs_inv_irls_oracle",
                                           "cgssf_oracle_vs_cgssf_timeq_oracle")
                  and k.split(":")[1] in ("tum_absrel", "tum_sc", "stx_absrel", "stx_sc", "tum17_absrel", "tum17_sc",
                                           "rep_absrel", "rep_sc", "fscore", "acc_cm", "comp_cm")}
    for fam in (fam_normal, fam_oracle):
        for k, ph in holm(fam).items():
            tests[k]["p_holm"] = ph
    trade = dict(normal=tradeoff(per, "cgssf", REPLICA), oracle=tradeoff(per, "cgssf_oracle", REPLICA))
    gate = {}
    for tag in ("cgssf", "cgssf_timeq", "cgssf_fixedgate", "cgssf_oracle", "cgssf_timeq_oracle", "cgssf_fixedgate_oracle"):
        for ds, seqs in (("tum", FR1), ("replica", REPLICA)):
            g = gate_pooled(per, tag, seqs)
            if g:
                g["beta"] = gate_scale(per, tag, seqs)
                gate[f"{tag}:{ds}"] = g
    # quality of the per-pixel depth uncertainty (ground-truth poses, run_suite --unc-eval)
    unc = {}
    for tag in ("cgssf_oracle", "cgssf_timeq_oracle", "cgssf_norank1_oracle", "cgssf_nogate_oracle", "cgssf_fixedgate_oracle"):
        for name, seqs in (("fr1", FR1), ("stx", STX), ("tum17", TUM), ("replica", REPLICA), ("all", TUM + REPLICA)):
            u = unc_summary(per, tag, seqs)
            if u:
                unc[f"{tag}:{name}"] = u
    # how often each CG-SSF / alignment status occurred (summed over runs), per tag and dataset
    status, status_seq = {}, {}
    for (tag, seq), rs in per.items():
        ds = "tum" if seq.startswith("fr") else "replica"
        d = status.setdefault(tag, {}).setdefault(ds, {})
        dq = status_seq.setdefault(tag, {}).setdefault(seq, {})
        for r in rs:
            for k, v in (r.get("status_counts") or {}).items():
                d[k] = d.get(k, 0) + v
                dq[k] = dq.get(k, 0) + v
    # per-frame stratification (ground-truth poses): raw per-frame fit vs. filter output, same anchors
    strat = {name: strata(per, "cgssf_oracle", seqs) for name, seqs in (("tum17", TUM), ("replica", REPLICA))}
    # Holm over the comparisons stated in the abstract (the paper's claims)
    ua = unc.get("cgssf_oracle:all") or {}
    claim_tests = {
        "gt_tum17_k_vs_d_absrel": tests.get("cgssf_oracle_vs_inv_irls_oracle:tum17_absrel"),
        "gt_tum17_k_vs_e_absrel": tests.get("cgssf_oracle_vs_irls_ema_oracle:tum17_absrel"),
        "band_tum17_k_vs_g_restricted": tests.get("cgssf_oracle_band_vs_cgssf_norank1_oracle_band:band_absrel_restricted"),
        "unc_all_sigma_vs_depth_ause": (ua.get("tests") or {}).get("sigma_vs_depth:ause_absrel"),
        "full_replica_k_vs_j_acc": tests.get("cgssf_vs_cgssf_uniform:acc_cm")}
    ch = holm({k: v["p"] for k, v in claim_tests.items() if v and v.get("p") is not None})
    claims = {k: dict(v, p_claims_holm=ch.get(k)) for k, v in claim_tests.items() if v}
    status_sets = {}
    for tag in status_seq:
        for name, seqs in (("fr1", FR1), ("stx", STX), ("replica", REPLICA), ("dyn", DYN)):
            c = {}
            for q in seqs:
                for k, v in status_seq[tag].get(q, {}).items():
                    c[k] = c.get(k, 0) + v
            if c:
                status_sets.setdefault(tag, {})[name] = c
    tables = {"table3_ablation": t3, "table4_ours": t4, "table5_ate": t5, "table6_backbones": t6acc,
              "table_oracle": oracle_rows, "table_sens": sens_rows, "table_band": band_rows,
              "table_designs": design_rows, "table_design_tuning": tune_rows}
    if a.runtime and os.path.exists(a.runtime):
        tables["table6_runtime"] = json.load(open(a.runtime))
    for name, rows in tables.items():
        if isinstance(rows, list):
            write_csv(os.path.join(a.out, name + ".csv"), rows)
    write_csv(os.path.join(a.out, "per_sequence.csv"), perseq)
    flat = []
    for (tag, seq), rs in sorted(per.items()):
        for r in rs:
            flat.append(dict(tag=tag, sequence=seq, seed=r.get("seed"), run=r["run"], code_hash=r.get("code_hash"),
                             gpu=r.get("gpu"), depth_model=r.get("depth_model"), failed=r.get("failed"),
                             reinit=r.get("reinit"), tracked_frac=r.get("tracked_frac"),
                             **{k: r.get(k) for k in METRICS}))
    write_csv(os.path.join(a.out, "provenance.csv"), flat)
    json.dump(dict(tables=tables, per_sequence=perseq, tests=tests, status=status, status_seq=status_seq, runs=flat,
                   tradeoff=trade, gate=gate, unc=unc, status_sets=status_sets, kappa_share=kappa_share,
                   strata=strat, claims=claims),
              open(os.path.join(a.out, "measured_results.json"), "w"), indent=1, default=str)
    cov = {v: sorted(q for (t, q) in per if t == v) for v in sorted({t for t, _ in per})}
    print("valid sequences per tag:", {k: len(v) for k, v in cov.items()})
    print("wrote tables, per_sequence.csv, provenance.csv, measured_results.json to", a.out)


if __name__ == "__main__":
    main()
