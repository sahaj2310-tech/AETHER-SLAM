"""Generate the documentation charts (docs/*.png) from the measured results:
results/runtime/runtime.json (tip/runtime_bench.py) and results/measured_results.json
(tip/collect_results.py). No hand-entered numbers."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
RT = json.load(open(os.path.join(RES, "runtime", "runtime.json")))
MR = json.load(open(os.path.join(RES, "measured_results.json")))
GPU = RT["env"].get("gpu", "GPU")

BG, FG, CY, GR, OR, PU, GRID = "#0a0e17", "#e6edf3", "#00f3ff", "#00ff73", "#ff6200", "#9d00ff", "#1b2436"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": FG,
                     "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG, "axes.edgecolor": GRID,
                     "axes.titlecolor": FG, "font.size": 11, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, name), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# 1. live end-to-end: throughput and stage service times (uncapped vs 4 GB cap)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
runs = [("live", "uncapped"), ("live_cap4gb", "4 GB cap")]
stages = ["slam", "semantic", "fusion"]
for j, (k, lab) in enumerate(runs):
    L = RT.get(k)
    if not L:
        continue
    med = [L["stages"][s]["median_ms"] for s in stages]
    p95 = [L["stages"][s]["p95_ms"] for s in stages]
    x = [i + (j - 0.5) * 0.38 for i in range(3)]
    axes[0].bar(x, med, width=0.38, color=[CY, GR][j], label=f"median ({lab})")
    axes[0].scatter(x, p95, color=OR, zorder=3, s=14, label="p95" if j == 0 else None)
axes[0].set_xticks(range(3)); axes[0].set_xticklabels(["tracking", "depth+align", "fusion"])
axes[0].set_ylabel("service time (ms)"); axes[0].set_title("Stage service times (live)")
axes[0].legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=8)
vals = [(lab, RT[k]["stages"]["fusion"]["fps_mean"], RT[k]["stages"]["slam"]["fps_mean"]) for k, lab in runs if RT.get(k)]
axes[1].bar([v[0] + "\nto fusion" for v in vals], [v[1] for v in vals], color=GR)
axes[1].bar([v[0] + "\ntracked" for v in vals], [v[2] for v in vals], color=CY)
for i, v in enumerate([v[1] for v in vals] + [v[2] for v in vals]):
    axes[1].text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=9)
axes[1].set_ylabel("frames / s"); axes[1].set_title(f"End-to-end throughput, {GPU}")
save(fig, "live_runtime.png")

# 2. per-component latency (isolated)
comp = RT["components"]
items = [("DISK+LightGlue", "DISK (1024) + LightGlue"), ("YOLOv8n-seg", "YOLOv8n-seg"), ("DPT_Hybrid", "DPT-Hybrid"),
         ("DPT_Large", "DPT-Large"), ("MiDaS_small", "MiDaS-small"), ("DepthAnythingV2_Small", "Depth Anything V2-S"),
         ("Weighted TSDF integration (GPU)", "weighted TSDF (GPU)"), ("Marching cubes (CPU)", "marching cubes (CPU)")]
items = [(k, n) for k, n in items if k in comp and "median_ms" in comp[k]]
fig, ax = plt.subplots(figsize=(8.5, 3.8))
lat = [comp[k]["median_ms"] for k, _ in items]
bars = ax.barh([n for _, n in items][::-1], lat[::-1], color=CY)
for b, v in zip(bars, lat[::-1]):
    ax.text(v + 0.5, b.get_y() + b.get_height() / 2, f"{v:.1f} ms", va="center", fontsize=9)
ax.set_xlabel("median latency (ms, isolated, after GPU sync)"); ax.set_title(f"Per-component latency, {GPU}")
save(fig, "model_latency.png")

# 3. depth networks: latency, memory, TUM AbsRel of the full system
bb = {r["depth_model"]: r for r in MR["tables"].get("table6_backbones", [])}
dm = [k for k in ("MiDaS_small", "DPT_Hybrid", "DPT_Large", "DepthAnythingV2_Small") if k in comp and "median_ms" in comp[k]]
fig, ax1 = plt.subplots(figsize=(8.5, 3.6))
x = range(len(dm))
ax1.bar([i - 0.25 for i in x], [comp[k]["median_ms"] for k in dm], width=0.25, color=GR, label="latency (ms)")
ax1.bar([i for i in x], [comp[k].get("peak_mb", 0) / 10 for k in dm], width=0.25, color=CY, label="memory (MB / 10)")
ax1.bar([i + 0.25 for i in x], [(bb.get(k) or {}).get("TUM_AbsRel_pct") or 0 for k in dm], width=0.25, color=OR, label="TUM AbsRel (%)")
ax1.set_xticks(list(x)); ax1.set_xticklabels(dm, fontsize=9)
ax1.set_title("Depth networks: latency, memory, accuracy of the full system")
ax1.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=8)
save(fig, "depth_tradeoff.png")

# 4. ablation: TUM AbsRel / SC and Replica F-score per variant
t3 = MR["tables"]["table3_ablation"]
labs = [r["label"].split(")")[0] + ")" for r in t3]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
for ax, key, title, col in [(axes[0], "TUM_AbsRel_pct", "TUM fr1 AbsRel (%)  ↓", CY),
                            (axes[1], "TUM_SC_x1e2", "TUM fr1 scale consistency SC (×10⁻²)  ↓", GR),
                            (axes[2], "Replica_F_pct", "Replica F-score @5 cm (%)  ↑", OR)]:
    v = [r[key] if r[key] is not None else 0 for r in t3]
    ax.bar(labs, v, color=[col if l != "(k)" else PU for l in labs])
    ax.set_title(title, fontsize=10)
fig.suptitle("Full-system ablation of the depth alignment (mean over seeds)", color=FG)
save(fig, "ablation.png")

for old in ("fps_milestones.png", "optimization_impact.png"):
    p = os.path.join(HERE, old)
    if os.path.exists(p):
        os.remove(p)
        print("removed outdated", old)
print("done")
