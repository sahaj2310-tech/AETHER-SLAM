"""Aggregation for the ground-truth-pose experiment, the fusion trade-off and the gate diagnostics
(used by collect_results.py).

* derive_uniform      (i) "CG-SSF, uniform weights" from the extra fusion volume 'uni_cinf' of the
                      CG-SSF runs: same depth, poses and frames, only the fusion weights differ
* tradeoff            mean accuracy / completion / precision / F-score of every fusion volume over
                      the Replica scenes, and the precision of the depth-cutoff baseline at the
                      completion ratio of uncertainty-weighted fusion (matched completeness)
* gate_pooled         frame-level gate diagnostics pooled over sequences (from diag.csv)
* holm, bootstrap_ci  multiple-comparison correction and confidence intervals
* rank_biserial       matched-pairs rank-biserial correlation (effect size of the Wilcoxon test)
* strata              per-frame comparison of the raw per-frame fit, the filter output and the best
                      affine fit (diag.csv) by number of anchors and observability (frames.csv)
* unc_summary         quality of the per-pixel depth uncertainty (run_suite --unc-eval) over
                      sequences: AUSE / AURG, coverage, paired tests between uncertainties
"""
import csv
import os

import numpy as np

RECON_KEYS = ("acc_cm", "comp_cm", "comp_ratio", "precision", "fscore", "chamfer_cm")
EXTRA_ORDER = [("unc_c0.5", "unc", 0.5), ("unc_c1", "unc", 1.0), ("unc_c2", "unc", 2.0), ("unc_c4", "unc", 4.0),
               ("unc_cinf", "unc", float("inf")), ("uni_c1", "uni", 1.0), ("uni_cinf", "uni", float("inf")),
               ("uni_z1.5", "depth", 1.5), ("uni_z2", "depth", 2.0), ("uni_z3", "depth", 3.0)]


def derive_uniform(per, base="cgssf", derived="cgssf_uniform"):
    """Add per[(derived, seq)] from per[(base, seq)]: reconstruction metrics of the extra volume
    'uni_cinf' (Replica); depth and trajectory metrics are those of the base run."""
    out = {}
    for (tag, seq), rs in per.items():
        if tag != base:
            continue
        new = []
        for r in rs:
            q = dict(r)
            ex = (r.get("extra") or {}).get("uni_cinf")
            if r.get("acc_cm") is not None:          # Replica: needs the extra volume
                if not ex:
                    continue
                for k in RECON_KEYS:
                    q[k] = ex.get(k)
            new.append(q)
        if new:
            out[(derived, seq)] = new
    for k, v in out.items():
        if k not in per:
            per[k] = v
    return per


def _scene_means(per, tag, scenes):
    """{scene: {config: {metric: mean over seeds}}} including the main volume as 'unc_c1'."""
    res = {}
    for q in scenes:
        rs = per.get((tag, q), [])
        if not rs:
            continue
        acc = {}
        for r in rs:
            cfgs = dict((r.get("extra") or {}))
            if r.get("acc_cm") is not None:
                cfgs["unc_c1"] = {k: r.get(k) for k in RECON_KEYS}
            for name, m in cfgs.items():
                if not m:
                    continue
                acc.setdefault(name, []).append(m)
        res[q] = {name: {k: float(np.mean([m[k] for m in ms if m.get(k) is not None])) for k in RECON_KEYS}
                  for name, ms in acc.items()}
    return res


def tradeoff(per, tag, scenes):
    sm = _scene_means(per, tag, scenes)
    if not sm:
        return None
    configs = []
    for name, kind, val in EXTRA_ORDER:
        vals = [sm[q][name] for q in sm if name in sm[q]]
        if len(vals) != len(sm):
            continue
        configs.append(dict(name=name, kind=kind, value=(None if not np.isfinite(val) else val),
                            **{k: float(np.mean([v[k] for v in vals])) for k in RECON_KEYS}, scenes=len(vals)))
    # matched completeness: precision of the unit-weight depth-cutoff family (z <= 1.5, 2, 3 and the
    # 4-unit fusion range = 'uni_cinf') interpolated at the completion ratio of 'unc_c1', per scene
    diffs = []
    for q, d in sm.items():
        fam = [d[n] for n in ("uni_z1.5", "uni_z2", "uni_z3", "uni_cinf") if n in d]
        if "unc_c1" not in d or len(fam) < 2:
            continue
        x = np.array([f["comp_ratio"] for f in fam]); y = np.array([f["precision"] for f in fam])
        o = np.argsort(x)
        cr = d["unc_c1"]["comp_ratio"]
        if cr < x.min() or cr > x.max():
            continue                                 # no extrapolation
        diffs.append((q, d["unc_c1"]["precision"], float(np.interp(cr, x[o], y[o]))))
    match = None
    if diffs:
        a = np.array([u for _, u, _ in diffs]); b = np.array([v for _, _, v in diffs])
        p = None
        if len(diffs) >= 5 and np.any(a != b):
            from scipy.stats import wilcoxon
            p = float(wilcoxon(a, b).pvalue)
        match = dict(n=len(diffs), precision_weighted=float(a.mean()), precision_cutoff=float(b.mean()),
                     wins=int((a > b).sum()), p=p, ci=bootstrap_ci(a - b), scenes=[q for q, _, _ in diffs])
    return dict(configs=configs, matched=match)


def gate_pooled(per, tag, seqs, kappa_min=0.01):
    """Pool the frame-level gate diagnostics (diag.csv) of all runs of `tag` on `seqs`."""
    recs = []
    for q in seqs:
        for r in per.get((tag, q), []):
            path = os.path.join(r["run"], "diag.csv")
            if not os.path.exists(path):
                continue
            with open(path) as fh:
                for row in csv.DictReader(fh):
                    recs.append(row)
    if not recs:
        return None
    f = lambda x: float(x) if x not in ("", None) else float("nan")
    kap = np.array([f(r["kappa"]) for r in recs]); st = np.array([r["status"] for r in recs])
    a_out = np.array([f(r["absrel_out"]) for r in recs]); a_fit = np.array([f(r["absrel_fit"]) for r in recs])
    a_best = np.array([f(r["absrel_best"]) for r in recs]); d2 = np.array([f(r["d2"]) for r in recs])
    full = np.isfinite(kap) & (kap >= kappa_min)

    def cat(m):
        m = m & np.isfinite(a_fit)
        if not m.any():
            return dict(n=0)
        return dict(n=int(m.sum()), absrel_fit=float(a_fit[m].mean()), absrel_out=float(a_out[m].mean()),
                    absrel_best=float(a_best[m].mean()), frac_fit_better=float(np.mean(a_fit[m] < a_out[m])))
    decided = full & np.isin(st, ["full", "rejected", "reset"])
    nis = d2[decided & np.isfinite(d2)]
    return dict(frames=len(recs), all=dict(absrel_out=float(np.nanmean(a_out)), absrel_fit=float(np.nanmean(a_fit)),
                                            absrel_best=float(np.nanmean(a_best))),
                accepted_full=cat(full & (st == "full")), rejected_full=cat(full & (st == "rejected")),
                reject_rate_full=float(np.mean(np.isin(st[decided], ["rejected", "reset"]))) if decided.any() else None,
                nis_median=float(np.median(nis)) if nis.size else None, nis_n=int(nis.size),
                nis_expected_median=1.386294)


CHI2_MED = {1: 0.454936, 2: 1.386294}


def gate_scale(per, tag, seqs, window=30, kappa_min=0.01):
    """Gate scale beta_k of Eq. (6), reconstructed from frames.csv (NIS and kappa of every measurement, in
    order; the window restarts with every new map, as the aligner does): median and 90th percentile over
    the measurements and the share with beta > 1, pooled over the runs of `tag` on `seqs`."""
    betas = []
    for q in seqs:
        for r in per.get((tag, q), []):
            path = os.path.join(r["run"], "frames.csv")
            if not os.path.exists(path):
                continue
            u, gen = [], None
            with open(path) as fh:
                for row in csv.DictReader(fh):
                    if row.get("gen") != gen:
                        u, gen = [], row.get("gen")
                    try:
                        d2, kap = float(row["nis"]), float(row["kappa"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if not (np.isfinite(d2) and np.isfinite(kap)):
                        continue
                    w = u[-window:]
                    betas.append(max(1.0, float(np.median(w))) if len(w) >= window // 2 else 1.0)
                    u.append(d2 / CHI2_MED[2 if kap >= kappa_min else 1])
    if not betas:
        return None
    b = np.array(betas)
    return dict(n=int(b.size), median=float(np.median(b)), p90=float(np.percentile(b, 90)),
                frac_above_1=float(np.mean(b > 1.0)))


def holm(pdict):
    """Holm-Bonferroni adjusted p-values for a dict {name: p}."""
    items = [(k, p) for k, p in pdict.items() if p is not None]
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    out, run = {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def bootstrap_ci(diffs, n=10000, level=0.95, seed=0):
    d = np.asarray(diffs, float)
    if d.size < 3:
        return None
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, d.size, size=(n, d.size))].mean(1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return [float(lo), float(hi)]


UNC = ("sigma", "residual", "depth")


def unc_summary(per, tag, seqs):
    """Mean over sequences of the per-sequence uncertainty evaluation of `tag` (one run per sequence,
    ground-truth poses), the per-sequence values, the mean sparsification curves, and paired
    Wilcoxon tests of sigma_z against the residual-only and depth-only uncertainties (AUSE / AURG
    of AbsRel and RMSE). None unless every sequence has an evaluation."""
    from scipy.stats import wilcoxon
    us = []
    for q in seqs:
        rs = [r for r in per.get((tag, q), []) if r.get("unc_eval")]
        if not rs:
            return None
        us.append(rs[0]["unc_eval"])
    out = dict(n=len(us), sequences=list(seqs), px=int(sum(u["n_px"] for u in us)),
               frames=int(sum(u["n_frames"] for u in us)), fractions=us[0]["fractions"])
    per_seq = {}
    for m in ("absrel", "rmse"):
        out["full_" + m] = float(np.mean([u["pooled"][m]["full"] for u in us]))
        for a in ("ause", "aurg"):
            out[f"{a}_{m}"] = {k: float(np.mean([u["pooled"][m][a][k] for u in us])) for k in UNC}
            out[f"frame_{a}_{m}"] = {k: float(np.mean([u["per_frame"][m][a][k] for u in us])) for k in UNC}
            for q, u in zip(seqs, us):
                per_seq.setdefault(q, {})[f"{a}_{m}"] = u["pooled"][m][a]
        out["curve_" + m] = {k: np.mean([u["pooled"][m]["curve"][k] for u in us], 0).tolist()
                             for k in UNC + ("oracle",)}
    for c in ("k1", "k2", "median_ratio"):
        out["coverage_" + c] = {k: float(np.mean([u["coverage"][k][c] for u in us])) for k in ("sigma", "residual")}
    out["frame_spearman"] = {k: float(np.mean([u["frame_spearman"][k] for u in us])) for k in UNC}
    w = [u["within_frame_spearman_sigma_depth"] for u in us if u.get("within_frame_spearman_sigma_depth") is not None]
    out["within_frame_spearman"] = float(np.min(w)) if w else None
    tests = {}
    for m in ("absrel", "rmse"):
        for a in ("ause", "aurg"):
            x = np.array([u["pooled"][m][a]["sigma"] for u in us])
            for k in ("residual", "depth"):
                y = np.array([u["pooled"][m][a][k] for u in us])
                better = x < y if a == "ause" else x > y
                try:
                    pv = float(wilcoxon(x, y).pvalue) if np.any(x != y) else 1.0
                except ValueError:
                    pv = None
                tests[f"sigma_vs_{k}:{a}_{m}"] = dict(n=len(x), wins=int(better.sum()), p=pv,
                                                      mean_diff=float((x - y).mean()), ci=bootstrap_ci(x - y),
                                                      r_rb=rank_biserial((y - x) if a == "ause" else (x - y)))
    out["tests"] = tests
    out["per_seq"] = per_seq
    return out


def rank_biserial(d):
    """Matched-pairs rank-biserial correlation of the differences d (positive = first better):
    (R+ - R-) / (R+ + R-) over the ranks of |d|, zeros dropped, ties averaged."""
    from scipy.stats import rankdata
    d = np.asarray(d, float)
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    r = rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


ANCHOR_BINS = [(8, 30), (30, 100), (100, 10 ** 9)]
KAPPA_BINS = [(0.0, 0.03), (0.03, 0.1), (0.1, float("inf"))]


def strata(per, tag, seqs, min_frames=10):
    """For every frame of the ground-truth-pose runs of `tag` with a per-frame fit: AbsRel of the raw fit,
    of the filter output and of the best affine fit (diag.csv, same anchors), grouped by the number of
    anchors and by kappa (frames.csv). Per stratum: frame-pooled means, share of frames with the output
    better than the fit, and a paired Wilcoxon test over the sequences with >= min_frames frames."""
    import csv
    from scipy.stats import wilcoxon
    frames = []
    for q in seqs:
        for r in per.get((tag, q), []):
            dp, fp = os.path.join(r["run"], "diag.csv"), os.path.join(r["run"], "frames.csv")
            if not (os.path.exists(dp) and os.path.exists(fp)):
                continue
            with open(fp) as fh:
                anch = {int(x["i"]): int(x["anchors"] or 0) for x in csv.DictReader(fh)}
            with open(dp) as fh:
                for x in csv.DictReader(fh):
                    try:
                        fit, out, best, k = float(x["absrel_fit"]), float(x["absrel_out"]), float(x["absrel_best"]), float(x["kappa"])
                    except (TypeError, ValueError):
                        continue
                    if not (np.isfinite(fit) and np.isfinite(out) and np.isfinite(best) and np.isfinite(k)):
                        continue
                    frames.append(dict(seq=q, n=anch.get(int(x["i"]), 0), kappa=k, fit=fit, out=out, best=best))
    if not frames:
        return None

    def summarize(sel):
        if not sel:
            return dict(frames=0)
        f = np.array([x["fit"] for x in sel]); o = np.array([x["out"] for x in sel]); b = np.array([x["best"] for x in sel])
        ps = {}
        for x in sel:
            ps.setdefault(x["seq"], []).append(x)
        pf = [(np.mean([y["fit"] for y in v]), np.mean([y["out"] for y in v])) for v in ps.values() if len(v) >= min_frames]
        test = None
        if len(pf) >= 5:
            fa, oa = np.array(pf).T
            try:
                pv = float(wilcoxon(oa, fa).pvalue) if np.any(oa != fa) else 1.0
            except ValueError:
                pv = None
            test = dict(n=len(pf), wins=int((oa < fa).sum()), p=pv, mean_diff=float((oa - fa).mean()),
                        ci=bootstrap_ci(oa - fa), r_rb=rank_biserial(fa - oa))
        return dict(frames=int(len(sel)), sequences=len(ps), fit=float(f.mean()), out=float(o.mean()), best=float(b.mean()),
                    out_better=float(np.mean(o < f)), median_fit=float(np.median(f)), median_out=float(np.median(o)), test=test)

    out = dict(all=summarize(frames))
    out["anchors"] = [dict(lo=lo, hi=(None if hi >= 10 ** 9 else hi), **summarize([x for x in frames if lo <= x["n"] < hi])) for lo, hi in ANCHOR_BINS]
    out["kappa"] = [dict(lo=lo, hi=(None if not np.isfinite(hi) else hi), **summarize([x for x in frames if lo <= x["kappa"] < hi])) for lo, hi in KAPPA_BINS]
    return out
