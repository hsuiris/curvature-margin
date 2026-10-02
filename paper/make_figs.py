"""Figures for the paper. Reads result CSVs and score pickles via cmargin.paths (set CMARGIN_DATA and CMARGIN_RESULTS).
Usage: python paper/make_figs.py [output_dir]"""
import sys, glob, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("pdf")
import matplotlib.pyplot as plt
from scipy.stats import norm

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from cmargin.paths import DATA, RESULTS
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "figs")
os.makedirs(OUT, exist_ok=True)
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "violet": "#4a3aa7", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df"}
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.edgecolor": C["ink2"], "axes.linewidth": 0.6, "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.5, "axes.axisbelow": True,
                     "lines.linewidth": 1.4, "lines.markersize": 4, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
GENS = ["Qwen2.5-3B", "phi-2", "SmolLM2-1.7B", "pythia-2.8b"]
res = lambda f: pd.read_csv(os.path.join(RESULTS, f))
m1, m2, ea = res("e1_scaled_sweep/metrics.csv"), res("e3_falcon_dual/metrics.csv"), res("e5_heldout_pythia/metrics.csv")
fal = pd.concat([m2, ea], ignore_index=True)
xing = pd.concat([res("e3_falcon_dual/crossing.csv"), res("e5_heldout_pythia/crossing.csv")], ignore_index=True).set_index("generator")

def cross(x, y, level=0.0):
    x, y = np.asarray(x, float), np.asarray(y, float) - level
    for i in range(len(x) - 1):
        if y[i] > 0 >= y[i + 1] or y[i] < 0 <= y[i + 1]:
            return x[i] + (x[i + 1] - x[i]) * y[i] / (y[i] - y[i + 1])
    return None

def half(ax):
    ax.axhline(0.5, color=C["ink2"], lw=0.7, ls=(0, (3, 2)), zorder=1)

def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight"); plt.close(fig); print("saved", name)

# ---------- 06_scaled_sweep curvature margins (scorer entropy − surprisal), computed from pickles ----------
rows = []
for f in sorted(glob.glob(os.path.join(DATA, "06_scaled_sweep", "scores_*.pkl"))):
    s = pd.read_pickle(f); kind = "GPT-2 XL" if "gpt2-xl" in f else "self"
    s["curv"] = [(a[1] - a[0]).mean() for a in s.arr]; h = s[s.who == "human"]
    for dec, g in s[s.who == "ai"].groupby("decoding"):
        rows.append(dict(gen=s.model.iloc[0].split("/")[-1], scorer=kind, T=float(dec[1:]), cm=g.curv.mean() - h.curv.mean()))
cm1 = pd.DataFrame(rows)
m1 = m1.merge(cm1, left_on=["generator", "scorer", "temperature"], right_on=["gen", "scorer", "T"], how="left")

# ---------- Fig 1: AUROC vs temperature ----------
fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.95), sharey=True)
for ax, g in zip(axes, GENS):
    half(ax)
    for kind, col, mk, lab in (("GPT-2 XL", C["orange"], "s", "GPT-2 XL (single)"), ("self", C["aqua"], "^", "Generator itself")):
        d = m1[(m1.generator == g) & (m1.scorer == kind)].sort_values("temperature")
        if len(d): ax.plot(d.temperature, d.AUROC, marker=mk, color=col, label=lab, ms=3.5)
    d = fal[fal.generator == g].sort_values("temperature")
    ax.plot(d.temperature, d.AUROC_FDG, marker="o", color=C["blue"], label="Falcon dual (Fast-DetectGPT)", ms=3.5)
    ax.fill_between(d.temperature, d.FDG_CI_low, d.FDG_CI_high, color=C["blue"], alpha=0.15, lw=0)
    p = xing.loc[g, "curvature_margin_dual_pred"]; ax.axvline(p, color=C["violet"], lw=1.0, ls=(0, (1, 1.5)))
    ax.set_title(g + (" (held out)" if g == "pythia-2.8b" else "")); ax.set_ylim(-0.02, 1.02)
axes[0].set_ylabel("AUROC"); axes[0].plot([], [], color=C["violet"], lw=1.0, ls=(0, (1, 1.5)), label="Zero of curvature margin $K$")
fig.supxlabel("Sampling temperature", fontsize=8, y=-0.04)
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.1))
save(fig, "fig1_auroc_temperature.pdf")

# ---------- Fig 2: margins vs AUROC ----------
pts = [dict(sm=r.surprisal_margin, cm=r.cm, au=r.AUROC, cfg="GPT-2 XL" if r.scorer == "GPT-2 XL" else "Generator itself") for r in m1.itertuples()]
pts += [dict(sm=r.surprisal_margin, cm=r.curvature_margin_dual, au=r.AUROC_FDG, cfg="Falcon dual") for r in fal.itertuples()]
P = pd.DataFrame(pts); sty = {"GPT-2 XL": (C["orange"], "s"), "Generator itself": (C["aqua"], "^"), "Falcon dual": (C["blue"], "o")}
fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.3), sharey=True)
for ax, key, lab in ((axes[0], "sm", "Surprisal margin $M$ (nats)"), (axes[1], "cm", "Curvature margin $K$ (nats)")):
    half(ax); ax.axvline(0, color=C["ink2"], lw=0.7, ls=(0, (3, 2)))
    for cfg, (col, mk) in sty.items():
        d = P[P.cfg == cfg]; ax.scatter(d[key], d.au, s=12, color=col, marker=mk, label=cfg, edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_xlabel(lab)
axes[0].set_ylabel("AUROC"); axes[0].legend(loc="upper left", frameon=False, markerscale=1.4)
save(fig, "fig2_margin_scatter.pdf")

# ---------- Fig 3: failure temperature by statistic ----------
bx = res("e6_zero_shot_benchmark/crossing.csv"); tx = res("e7_trained_baselines/crossing.csv")
order = ["Entropy", "Likelihood", "LogRank", "LRR", "DMAP_position", "Binoculars", "FastDetectGPT"]
names = {"Entropy": "Entropy$^\\dagger$", "Likelihood": "Likelihood", "LogRank": "LogRank", "LRR": "LRR", "DMAP_position": "DMAP position", "Binoculars": "Binoculars", "FastDetectGPT": "Fast-DetectGPT"}
fig, ax = plt.subplots(figsize=(4.2, 2.4)); cols = [C["blue"], C["orange"], C["aqua"], C["violet"]]; mks = ["o", "s", "^", "D"]
for j, (g, col, mk) in enumerate(zip(GENS, cols, mks)):
    d = bx[bx.generator == g].set_index("method")
    y = np.arange(len(order)) + (j - 1.5) * 0.17
    ax.errorbar([d.loc[o, "fail_temp"] for o in order], y, xerr=[[d.loc[o, "fail_temp"] - d.loc[o, "CI_low"] for o in order], [d.loc[o, "CI_high"] - d.loc[o, "fail_temp"] for o in order]],
                fmt=mk, color=col, ms=3.5, elinewidth=0.8, capsize=0, label=g)
ax.set_yticks(range(len(order))); ax.set_yticklabels([names[o] for o in order]); ax.set_xlabel("Failure temperature (AUROC = 0.5)")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False); ax.grid(axis="y", visible=False)
save(fig, "fig3_failure_by_statistic.pdf")

# ---------- Fig 4: per-domain failure temperature ----------
dx = res("e6_zero_shot_benchmark/domain_crossing.csv")
fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.9), sharex=True)
for ax, g in zip(axes, GENS):
    d = dx[dx.generator == g]; doms = sorted(d.domain.unique())
    for meth, col, mk, lab in (("FastDetectGPT", C["blue"], "o", "Fast-DetectGPT (Falcon dual)"), ("LogRank", C["orange"], "s", "LogRank (Falcon-7B-Instruct)")):
        dd = d[d.method == meth].set_index("domain").reindex(doms)
        ax.scatter(dd.fail_temp, range(len(doms)), color=col, marker=mk, s=14, label=lab, zorder=3)
    ax.set_yticks(range(len(doms))); ax.set_yticklabels(doms); ax.set_title(g); ax.grid(axis="y", visible=False)
plt.tight_layout(); fig.supxlabel("Failure temperature (AUROC = 0.5)", fontsize=8, y=-0.04)
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.1))
save(fig, "fig4_domain_failure.pdf")

# ---------- Fig 5: M4GT cells ----------
mc = res("e4_m4gt_natural/cells.csv")
fig, ax = plt.subplots(figsize=(3.4, 2.4)); half(ax); ax.axvline(0, color=C["ink2"], lw=0.7, ls=(0, (3, 2)))
b = mc.generator.eq("bloomz")
ax.scatter(mc[~b].curvature_margin, mc[~b].AUROC_FDG, s=14, color=C["blue"], marker="o", label="Other generators", edgecolor="white", linewidth=0.4, zorder=3)
ax.scatter(mc[b].curvature_margin, mc[b].AUROC_FDG, s=18, color=C["orange"], marker="s", label="BLOOMZ", edgecolor="white", linewidth=0.4, zorder=3)
for r in mc[b].itertuples(): ax.annotate(r.domain, (r.curvature_margin, r.AUROC_FDG), xytext=(4, -2), textcoords="offset points", fontsize=6, color=C["ink2"])
ax.set_xlabel("Curvature margin $K$ (nats)"); ax.set_ylabel("Fast-DetectGPT AUROC"); ax.legend(loc="lower right", frameon=False)
save(fig, "fig5_m4gt_cells.pdf")

# ---------- Fig 6: TPR at 1% FPR vs temperature ----------
bm = res("e6_zero_shot_benchmark/metrics.csv")
fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.9), sharey=True)
for ax, g in zip(axes, GENS):
    d = fal[fal.generator == g].sort_values("temperature"); ax.plot(d.temperature, d["TPR_1pct_FDG"], marker="o", color=C["blue"], ms=3.5, label="Fast-DetectGPT (Falcon dual)")
    d = m1[(m1.generator == g) & (m1.scorer == "GPT-2 XL")].sort_values("temperature")
    if len(d): ax.plot(d.temperature, d["TPR_1pct"], marker="s", color=C["orange"], ms=3.5, label="Fast-DetectGPT (GPT-2 XL)")
    d = bm[(bm.generator == g) & (bm.method == "LogRank")].sort_values("temperature"); ax.plot(d.temperature, d["TPR_1pct"], marker="^", color=C["aqua"], ms=3.5, label="LogRank (Falcon-7B-Instruct)")
    ax.axhline(0.01, color=C["ink2"], lw=0.7, ls=(0, (3, 2))); ax.set_title(g)
axes[0].set_ylabel("TPR at 1% FPR")
fig.supxlabel("Sampling temperature", fontsize=8, y=-0.04)
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.1))
save(fig, "fig6_tpr_1pct.pdf")

# ---------- Fig 7: trained baselines ----------
tm = res("e7_trained_baselines/metrics.csv")
spec = [("ada_identity", C["blue"], "o", "Fast-DetectGPT (untrained witness)"), ("ada_all3", C["violet"], "D", "AdaDetectGPT (trained)"),
        ("mage_longformer", C["orange"], "s", "MAGE Longformer"), ("diveye_all3", C["yellow"], "v", "DivEye"), ("roberta_openai", C["aqua"], "^", "RoBERTa OpenAI detector")]
fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.95), sharey=True)
for ax, g in zip(axes, GENS):
    half(ax)
    for det, col, mk, lab in spec:
        d = tm[(tm.generator == g) & (tm.detector == det)].sort_values("temperature")
        if len(d): ax.plot(d.temperature, d.AUROC, marker=mk, color=col, ms=3.2, label=lab)
    ax.set_title(g); ax.set_ylim(-0.02, 1.02)
axes[0].set_ylabel("AUROC")
fig.supxlabel("Sampling temperature", fontsize=8, y=-0.04)
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.1))
save(fig, "fig7_trained_baselines.pdf")

# numbers used in text
print("Spearman sm/cm vs AUROC:", P[["sm", "cm", "au"]].corr(method="spearman").round(3).to_dict())
print("n points", len(P))

# ---------- Appendix: human-text-only estimates versus observed failure temperatures.
estimate_first = res("e9_theory_prediction/test.csv").rename(columns={
    "generator": "model",
    "fail_temp_predicted": "T_hat",
    "fail_temp_observed": "T_star",
})
estimate_replication = res("e12_replication_test/per_model.csv")
estimate_all = pd.concat([
    estimate_first[["model", "T_hat", "T_star"]],
    estimate_replication[["model", "T_hat", "T_star"]],
], ignore_index=True)
assert (len(estimate_first), len(estimate_replication)) == (4, 12)
assert estimate_all.model.is_unique
assert np.isfinite(estimate_all[["T_hat", "T_star"]].to_numpy()).all()

estimate_values = estimate_all[["T_hat", "T_star"]].to_numpy()
estimate_pad = np.ptp(estimate_values) * 0.08
estimate_limits = (estimate_values.min() - estimate_pad,
                   estimate_values.max() + estimate_pad)
fig, ax = plt.subplots(figsize=(3.05, 2.85))
fig.subplots_adjust(left=0.19, right=0.98, bottom=0.18, top=0.97)
ax.plot(estimate_limits, estimate_limits, color=C["ink2"],
        lw=0.7, ls=(0, (3, 2)), zorder=1)
for estimate_data, estimate_marker, estimate_label in (
    (estimate_first, "o", "First test"),
    (estimate_replication, "s", "Replication"),
):
    ax.scatter(estimate_data.T_hat, estimate_data.T_star, s=12,
               color=C["aqua"], marker=estimate_marker,
               edgecolor="white", linewidth=0.4, zorder=3,
               label=f"{estimate_label} ({len(estimate_data)} models)")
ax.set_xlim(estimate_limits)
ax.set_ylim(estimate_limits)
ax.set_aspect("equal", adjustable="box")
# Mathtext scales superscripts to 70%; 10 pt keeps the star at 7 pt.
ax.set_xlabel(r"Estimate $\hat{T}$ (human text only)", fontsize=10)
ax.set_ylabel(r"Observed failure temperature $T^{*}$", fontsize=10)
ax.legend(loc="upper left", frameon=False, markerscale=1.4)

estimate_labels = {
    "facebook/xglm-1.7B": ("XGLM-1.7B", (5, 2), "left", "bottom"),
    "google/gemma-3-1b-pt": ("Gemma-3-1B", (3, -6), "right", "top"),
    "PleIAs/Pleias-1.2b-Preview": ("Pleias-1.2B", (5, -6), "left", "top"),
}
estimate_largest = estimate_replication.assign(
    abs_error=(estimate_replication.T_hat - estimate_replication.T_star).abs()
).nlargest(3, "abs_error")
assert set(estimate_largest.model) == set(estimate_labels)
for estimate_row in estimate_largest.itertuples():
    estimate_name, estimate_offset, estimate_ha, estimate_va = estimate_labels[estimate_row.model]
    ax.annotate(estimate_name, (estimate_row.T_hat, estimate_row.T_star),
                xytext=estimate_offset, textcoords="offset points",
                ha=estimate_ha, va=estimate_va, fontsize=7, color=C["ink2"])
save(fig, "fig_estimate_vs_observed.pdf")
