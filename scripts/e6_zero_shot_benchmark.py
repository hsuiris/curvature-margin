# Section 5.3 / Table 5: eight zero-shot statistics with official formulas and orientation.
# Formal zero-shot baselines (design doc §8.8.15) on the Falcon dual-model per-token rows: 07_falcon_dual_scoring (Qwen2.5-3B, phi-2,
# SmolLM2-1.7B on MAGE humans) and 09_heldout_pythia (pythia-2.8b on M4GT humans). Every score is computed from the stored 12 rows with
# the official formula and the official direction (higher = AI): Likelihood / LogRank / Entropy (fast-detect-gpt
# scripts/baselines.py @971b0520), LRR (DetectLLM, Su et al. 2023; scripts/detect_llm.py), Fast-DetectGPT analytic
# (scripts/fast_detect_gpt.py @971b0520; scorer falcon-7b-instruct, reference falcon-7b), Binoculars (ahans30/Binoculars
# binoculars/metrics.py @c8ae2f90; performer falcon-7b-instruct, observer falcon-7b). DMAP (Featurespace/dmap @f13c10b6) defines
# no detector; two derived statistics are included and labelled as ours. Protocol as e3_falcon_dual.py: AUROC + 1,000-bootstrap CI,
# 1% threshold from even-id humans (actual FPR on odd-id), crossing by linear interpolation + bootstrap CI, per-domain AUROC.
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1 if 'prepare' not in __file__ else 2]))
from cmargin.paths import DATA, RESULTS
(RESULTS / "e6_zero_shot_benchmark").mkdir(parents=True, exist_ok=True)
import glob
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from cmargin.crossing import cross

N_BOOT, FPR, NBIN = 1000, 0.01, 40
SETS = {"main2a": f"{DATA}/07_falcon_dual_scoring/scores_falcon_*.pkl", "extA": f"{DATA}/09_heldout_pythia/scores_falcon_*.pkl"}
METHODS = ["Likelihood", "LogRank", "Entropy", "LRR", "FastDetectGPT", "Binoculars", "DMAP_position", "DMAP_chi2"]


def dmap_stats(a):
    # DMAP interval per token: [a_i, b_i] with a_i = mass of more-likely tokens, b_i = a_i + p(w_i); our row 8 is the mass of
    # less-likely tokens, so b_i = 1 - cdf, a_i = b_i - p. DMAP_position = minus the mean interval midpoint (head-biased = AI, higher = AI).
    # DMAP_chi2 = entropy-weighted (clip 2) 40-bin histogram (paper §3.2) compared with the uniform density, NBIN * sum (D_b - 1/NBIN)^2.
    s, H, cdf = a[0], a[1], a[8]
    p = np.exp(-s); b = np.clip(1 - cdf, 0, 1); lo = np.clip(b - p, 0, 1)
    w = np.minimum(H, 2.0); w = w / w.sum() if w.sum() > 0 else np.full_like(w, 1 / len(w))
    edges = np.linspace(0, 1, NBIN + 1); L = np.maximum(b - lo, 1e-9)
    ov = np.clip(np.minimum(b[None, :], edges[1:, None]) - np.maximum(lo[None, :], edges[:-1, None]), 0, None)  # [NBIN, T]
    hist = (ov / L[None, :] * w[None, :]).sum(1)
    return -((lo + b) / 2).mean(), NBIN * ((hist - 1 / NBIN) ** 2).sum()


def feats(s):
    A = s.arr; s = s.copy()
    s["model"] = s.model.str.split("/").str[-1]
    s["T"] = [float(d[1:]) if d != "human" else np.nan for d in s.decoding]
    s["Likelihood"] = [-a[0].mean() for a in A]
    s["LogRank"] = [-np.log(a[7] + 1).mean() for a in A]
    s["Entropy"] = [a[1].mean() for a in A]
    s["LRR"] = [a[0].mean() / max(np.log(a[7] + 1).mean(), 1e-6) for a in A]
    s["FastDetectGPT"] = [(-a[0].sum() - a[4].sum()) / np.sqrt(a[5].sum()) for a in A]
    s["Binoculars"] = [-(a[0].mean() / -a[4].mean()) for a in A]
    d = np.array([dmap_stats(a) for a in A]); s["DMAP_position"], s["DMAP_chi2"] = d[:, 0], d[:, 1]
    return s.drop(columns=["arr"])


def auroc(g, h):
    return roc_auc_score(np.r_[np.ones(len(g)), np.zeros(len(h))], np.r_[g, h])


def main():
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 300); pd.set_option("display.max_rows", 500); pd.set_option("display.max_columns", 40)
    M, D, X = [], [], []
    for setname, pat in SETS.items():
        for f in sorted(glob.glob(pat)):
            d = feats(pd.read_pickle(f)); m = d.model.iloc[0]
            h = d[d.who == "human"]; ai = d[d.who == "ai"]; h_cal, h_test = h[h.id % 2 == 0], h[h.id % 2 == 1]
            Ts = sorted(ai["T"].unique()); G = {T: ai[np.isclose(ai["T"], T)] for T in Ts}
            for meth in METHODS:
                thr = np.quantile(h_cal[meth], 1 - FPR); hv = h[meth].values; curve = []
                for T in Ts:
                    g = G[T][meth].values
                    au = auroc(g, hv); curve.append(au)
                    b = [auroc(rng.choice(g, len(g)), rng.choice(hv, len(hv))) for _ in range(N_BOOT)]
                    tb = [(rng.choice(g, len(g)) > thr).mean() for _ in range(N_BOOT)]
                    M.append(dict(dataset=setname, generator=m, method=meth, temperature=T, n_ai=len(g), AUROC=au, CI_low=np.percentile(b, 2.5), CI_high=np.percentile(b, 97.5),
                                  TPR_1pct=(g > thr).mean(), TPR_CI_low=np.percentile(tb, 2.5), TPR_CI_high=np.percentile(tb, 97.5), FPR_realized=(h_test[meth] > thr).mean()))
                    for dom, gg in G[T].groupby("domain"):
                        D.append(dict(dataset=setname, generator=m, method=meth, domain=dom, temperature=T, n_ai=len(gg), AUROC=auroc(gg[meth].values, h[h.domain == dom][meth].values)))
                ac = cross(Ts, curve, 0.5); boots = []
                for _ in range(N_BOOT):
                    hb = rng.choice(hv, len(hv))
                    c = cross(Ts, [auroc(rng.choice(G[T][meth].values, len(G[T])), hb) for T in Ts], 0.5)
                    if c is not None: boots.append(c)
                X.append(dict(dataset=setname, generator=m, method=meth, fail_temp=ac, CI_low=np.percentile(boots, 2.5) if boots else np.nan, CI_high=np.percentile(boots, 97.5) if boots else np.nan,
                              share_bootstrap_crossing=len(boots) / N_BOOT, monotone_decreasing="yes" if (np.diff(curve) < 0).all() else "no", AUROC_lowest_temp=curve[0], AUROC_highest_temp=curve[-1]))
            print("done", setname, m, flush=True)
    M, D, X = pd.DataFrame(M), pd.DataFrame(D), pd.DataFrame(X)
    # per-domain crossing per method
    rows = []
    for (setname, m, meth, dom), g in D.groupby(["dataset", "generator", "method", "domain"]):
        g = g.sort_values("temperature"); rows.append(dict(dataset=setname, generator=m, method=meth, domain=dom, fail_temp=cross(g.temperature, g.AUROC, 0.5)))
    DX = pd.DataFrame(rows)
    spread = DX.groupby(["generator", "method"]).fail_temp.agg(lambda x: x.max() - x.min() if x.notna().all() else np.nan).unstack()
    print("\n== Failure temperature (linear interpolation at AUROC = 0.5; CIs from 1,000 bootstrap samples) ==")
    print(X.pivot_table(index="generator", columns="method", values="fail_temp").round(3).to_string())
    print("\n== CI half-width of the failure temperature ==")
    X["CI_half_width"] = (X.CI_high - X.CI_low) / 2; print(X.pivot_table(index="generator", columns="method", values="CI_half_width").round(3).to_string())
    print("\n== AUROC at the lowest temperature (0.6 or 0.8) ==")
    print(M[M.groupby(["generator", "method"]).temperature.transform("min") == M.temperature].pivot_table(index="generator", columns="method", values="AUROC").round(3).to_string())
    print("\n== TPR at 1% FPR, T = 0.8 ==")
    print(M[np.isclose(M.temperature, 0.8)].pivot_table(index="generator", columns="method", values="TPR_1pct").round(3).to_string())
    print("\n== AUROC at T = 1.0 ==")
    print(M[np.isclose(M.temperature, 1.0)].pivot_table(index="generator", columns="method", values="AUROC").round(3).to_string())
    print("\n== Largest cross-domain spread of the failure temperature per method ==")
    print(spread.round(3).to_string())
    print("\n== Monotone decreasing ==")
    print(X.pivot_table(index="generator", columns="method", values="monotone_decreasing", aggfunc="first").to_string())
    M.to_csv(f"{RESULTS}/e6_zero_shot_benchmark/metrics.csv", index=False); D.to_csv(f"{RESULTS}/e6_zero_shot_benchmark/domains.csv", index=False)
    X.to_csv(f"{RESULTS}/e6_zero_shot_benchmark/crossing.csv", index=False); DX.to_csv(f"{RESULTS}/e6_zero_shot_benchmark/domain_crossing.csv", index=False)


if __name__ == "__main__":
    main()
