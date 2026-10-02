# Section 5.2 / Table 4 (held-out row): pythia-2.8b on M4GT human prompts; pre-registered test.
# External validation A (design doc §8.8.14 / external_validation_protocol.md A): pythia-2.8b continuations of M4GT humans,
# Falcon-7B-Instruct scorer + Falcon-7B observer. Same computations as e3_falcon_dual.py, without the GPT-2 XL comparator.
# Same protocol as e1_scaled_sweep.py / e2_curvature_margin.py: AUROC per (generator, temperature) against all 1,000 humans with a 1,000-sample bootstrap CI;
# 1% FPR threshold from the even-id 500 humans (np.quantile 0.99, strict >), actual FPR on the odd-id 500; crossings by linear
# interpolation on the temperature grid; margins from means over all 1,000 humans. Covers E1-E4 of 8.8.12, adds crossing CIs,
# TPR CIs and per-domain crossings.
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1 if 'prepare' not in __file__ else 2]))
from cmargin.paths import DATA, RESULTS
(RESULTS / "e5_heldout_pythia").mkdir(parents=True, exist_ok=True)
import glob
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from cmargin.crossing import cross

N_BOOT, FPR = 1000, 0.01


def feats(s):
    s = s.copy(); A = s.arr
    s["model"] = s.model.str.split("/").str[-1]
    s["T"] = [float(d[1:]) if d != "human" else np.nan for d in s.decoding]
    s["surprisal"] = [a[0].mean() for a in A]
    s["curv_self"] = [(a[1] - a[0]).mean() for a in A]       # scorer entropy − surprisal (e2_curvature_margin.py definition)
    s["curv_dual"] = [(-a[4] - a[0]).mean() for a in A]      # observer cross-entropy − surprisal (Fast-DetectGPT numerator)
    s["fdg"] = [(-a[0].sum() - a[4].sum()) / np.sqrt(a[5].sum()) for a in A]
    s["bino"] = [-(a[0].mean() / -a[4].mean()) for a in A]  # higher = more AI-like
    return s.drop(columns=["arr"])


def auroc(g, h):
    return roc_auc_score(np.r_[np.ones(len(g)), np.zeros(len(h))], np.r_[g, h])


def auroc_ci(g, h, rng):
    b = [auroc(rng.choice(g, len(g)), rng.choice(h, len(h))) for _ in range(N_BOOT)]
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def crossing_ci(Ts, G, h, rng):
    # resample humans once and each temperature's AI texts, recompute the AUROC curve, take its 0.5 crossing
    out = []
    for _ in range(N_BOOT):
        hb = rng.choice(h, len(h))
        c = cross(Ts, [auroc(rng.choice(g, len(g)), hb) for g in G], 0.5)
        if c is not None: out.append(c)
    return np.percentile(out, 2.5), np.percentile(out, 97.5), len(out)


def main():
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 300); pd.set_option("display.max_rows", 200); pd.set_option("display.max_columns", 50)
    rows, X, Dm = [], [], []
    for f in sorted(glob.glob(f"{DATA}/09_heldout_pythia/scores_falcon_*.pkl")):
        d = feats(pd.read_pickle(f)); m = d.model.iloc[0]
        h = d[d.who == "human"]; ai = d[d.who == "ai"]
        h_cal, h_test = h[h.id % 2 == 0], h[h.id % 2 == 1]
        thr = {c: np.quantile(h_cal[c], 1 - FPR) for c in ("fdg", "bino")}
        thr_all = {c: np.quantile(h[c], 1 - FPR) for c in ("fdg", "bino")}
        for T, g in ai.groupby("T"):
            r = dict(generator=m, temperature=T, n_ai=len(g), surprisal_margin=h.surprisal.mean() - g.surprisal.mean(),
                     curvature_margin_self=g.curv_self.mean() - h.curv_self.mean(), curvature_margin_dual=g.curv_dual.mean() - h.curv_dual.mean())
            for c, name in (("fdg", "FDG"), ("bino", "Bino")):
                lo, hi = auroc_ci(g[c].values, h[c].values, rng)
                tb = [(rng.choice(g[c].values, len(g)) > thr[c]).mean() for _ in range(N_BOOT)]
                r.update({f"AUROC_{name}": auroc(g[c].values, h[c].values), f"{name}_CI_low": lo, f"{name}_CI_high": hi, f"{name}_CI_half_width": (hi - lo) / 2,
                          f"TPR_1pct_{name}": (g[c] > thr[c]).mean(), f"{name}_TPR_CI_low": np.percentile(tb, 2.5), f"{name}_TPR_CI_high": np.percentile(tb, 97.5),
                          f"FPR_realized_{name}": (h_test[c] > thr[c]).mean(), f"TPR_1pct_same_data_{name}": (g[c] > thr_all[c]).mean()})
            rows.append(r)
        R = pd.DataFrame([r for r in rows if r["generator"] == m]).sort_values("temperature")
        Ts = R.temperature.values; G = [ai[np.isclose(ai["T"], T)] for T in Ts]
        ac_f, ac_b = cross(Ts, R.AUROC_FDG, 0.5), cross(Ts, R.AUROC_Bino, 0.5)
        lo, hi, nb = crossing_ci(Ts, [g.fdg.values for g in G], h.fdg.values, rng)
        near = R[(R.AUROC_FDG > 0.3) & (R.AUROC_FDG < 0.7)]
        x = dict(generator=m, fail_temp_observed_FDG=ac_f, FDG_CI_low=lo, FDG_CI_high=hi, fail_temp_observed_Bino=ac_b, fail_temp_diff_FDG_Bino=abs(ac_f - ac_b),
                 monotone_decreasing_FDG="yes" if (np.diff(R.AUROC_FDG) < 0).all() else "no",
                 monotone_decreasing_Bino="yes" if (np.diff(R.AUROC_Bino) < 0).all() else "no", max_CI_half_width_near_crossing_FDG=near.FDG_CI_half_width.max() if len(near) else np.nan)
        for col, name in (("surprisal_margin", "surprisal_margin"), ("curvature_margin_self", "curvature_margin_self"), ("curvature_margin_dual", "curvature_margin_dual")):
            pc = cross(Ts, R[col]); x[f"{name}_pred"] = pc; x[f"{name}_error"] = None if pc is None else ac_f - pc
        X.append(x)
        for dom, dd in d.groupby("domain"):
            hh = dd[dd.who == "human"]; pts = sorted((T, auroc(g.fdg.values, hh.fdg.values), g.curv_dual.mean() - hh.curv_dual.mean(), len(g))
                                                     for T, g in dd[dd.who == "ai"].groupby("T"))
            Dm.append(dict(generator=m, domain=dom, n_human=len(hh), n_ai_per_temperature=pts[0][3], fail_temp_observed_FDG=cross([p[0] for p in pts], [p[1] for p in pts], 0.5),
                           curvature_margin_dual_pred=cross([p[0] for p in pts], [p[2] for p in pts])))
    R, X, Dm = pd.DataFrame(rows).sort_values(["generator", "temperature"]), pd.DataFrame(X), pd.DataFrame(Dm)
    spread = Dm.groupby("generator").fail_temp_observed_FDG.agg(lambda x: x.max() - x.min())
    print("== Per condition (300 AI vs 1,000 human texts; 1% threshold set on the 500 even-id humans, realized FPR on the 500 odd-id humans; CIs from 1,000 bootstrap samples) ==")
    print(R.round(3).to_string(index=False))
    print("\n== E1/E2 failure temperature (observed, CI, difference between the two detectors) and E3 prediction errors of the three margins (positive = predicted too low) ==")
    print(X.round(3).to_string(index=False))
    print("\n== Failure temperature per domain (Falcon Fast-DetectGPT) ==")
    print(Dm.round(3).to_string(index=False)); print("Largest cross-domain spread:", spread.round(3).to_dict())
    R.to_csv(f"{RESULTS}/e5_heldout_pythia/metrics.csv", index=False); X.to_csv(f"{RESULTS}/e5_heldout_pythia/crossing.csv", index=False)
    Dm.to_csv(f"{RESULTS}/e5_heldout_pythia/domains.csv", index=False)
    for _, x in X.iterrows():
        print(f"\nPrimary endpoint (pre-registered criterion: dual curvature-margin prediction error <= 0.02): {x.generator} error {abs(x.curvature_margin_dual_error):.3f} -> {'met' if abs(x.curvature_margin_dual_error) <= 0.02 else 'not met'}; secondary endpoint, FDG vs Bino crossing difference {x.fail_temp_diff_FDG_Bino:.3f}")


if __name__ == "__main__":
    main()
