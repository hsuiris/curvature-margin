# Section 5.2 / Table 4 (GPT-2 XL and self rows): scaled sweep, 300 vs 1,000 texts, 0.02 grid.
# Main experiment 1 analysis (design doc §8.8.11): 300 texts per condition, 1,000 humans, 7 domains, 0.02 grid.
# E1 crossing prediction; E2 CI half-width near the crossing; E3 TPR at 1% FPR with a held-out human calibration
# split; E4 per-domain crossing spread; E5 forward prediction with the old sigma; plus a refit of sigma.
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1 if 'prepare' not in __file__ else 2]))
from cmargin.paths import DATA, RESULTS
(RESULTS / "e1_scaled_sweep").mkdir(parents=True, exist_ok=True)
import glob
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm, spearmanr
from sklearn.metrics import roc_auc_score
from cmargin.crossing import cross

SIGMA_OLD, N_BOOT, FPR = 0.638, 1000, 0.01


def load():
    frames = []
    for f in sorted(glob.glob(f"{DATA}/06_scaled_sweep/scores_*.pkl")):
        s = pd.read_pickle(f)
        s["scorer_kind"] = "GPT-2 XL" if "gpt2-xl" in f else "self"
        s["surprisal"] = [a[0].mean() for a in s.arr]
        s["fdg"] = [(a[1].sum() - a[0].sum()) / np.sqrt(a[2].sum()) for a in s.arr]
        s["y"] = (s.who == "ai").astype(int)
        s["T"] = [float(d[1:]) if d != "human" else np.nan for d in s.decoding]
        s["model"] = s.model.str.split("/").str[-1]
        frames.append(s.drop(columns=["arr"]))
    return pd.concat(frames, ignore_index=True)


def cell_stats(g, h, rng, boot=True):
    y = np.r_[np.ones(len(g)), np.zeros(len(h))]; sc = np.r_[g.fdg.values, h.fdg.values]
    au = roc_auc_score(y, sc)
    if not boot:
        return au, None, None
    pos, neg = np.arange(len(g)), len(g) + np.arange(len(h))
    b = [roc_auc_score(y[i], sc[i]) for i in (np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))] for _ in range(N_BOOT))]
    return au, np.percentile(b, 2.5), np.percentile(b, 97.5)


def main():
    rng = np.random.default_rng(0)
    d = load()
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 200)
    rows = []
    for (m, sk), dd in d.groupby(["model", "scorer_kind"]):
        h = dd[dd.who == "human"]
        cal_ids = set(h.id[h.id % 2 == 0]); h_cal, h_test = h[h.id.isin(cal_ids)], h[~h.id.isin(cal_ids)]
        thr = np.quantile(h_cal.fdg, 1 - FPR)
        for T, g in dd[dd.who == "ai"].groupby("T"):
            au, lo, hi = cell_stats(g, h, rng)
            rows.append(dict(generator=m, scorer=sk, temperature=T, n_ai=len(g), surprisal_margin=h.surprisal.mean() - g.surprisal.mean(),
                             AUROC=au, CI_low=lo, CI_high=hi, CI_half_width=(hi - lo) / 2,
                             TPR_1pct=(g.fdg > thr).mean(), FPR_realized=(h_test.fdg > thr).mean()))
    R = pd.DataFrame(rows).sort_values(["generator", "scorer", "temperature"])
    R["AUROC_predicted_pilot_sigma"] = norm.cdf(R.surprisal_margin / SIGMA_OLD)
    # refit sigma on the new GPT-2 XL cells
    gx = R[R.scorer == "GPT-2 XL"]
    fit = minimize_scalar(lambda s: ((norm.cdf(gx.surprisal_margin / s) - gx.AUROC) ** 2).sum(), bounds=(0.05, 5), method="bounded")

    X = []
    for (m, sk), g in R.groupby(["generator", "scorer"]):
        g = g.sort_values("temperature")
        pc, ac = cross(g.temperature, g.surprisal_margin), cross(g.temperature, g.AUROC, 0.5)
        near = g[(g.AUROC > 0.3) & (g.AUROC < 0.7)]
        X.append(dict(generator=m, scorer=sk, fail_temp_predicted=pc, fail_temp_observed=ac, diff=None if pc is None or ac is None else abs(pc - ac),
                      max_CI_half_width_near_crossing=near.CI_half_width.max() if len(near) else np.nan, monotone_decreasing="yes" if (np.diff(g.AUROC) < 0).all() else "no"))
    X = pd.DataFrame(X)

    # per-domain crossing (GPT-2 XL)
    Dm = []
    for m, dd in d[d.scorer_kind == "GPT-2 XL"].groupby("model"):
        for dom, ddd in dd.groupby("domain"):
            h = ddd[ddd.who == "human"]; pts = []
            for T, g in ddd[ddd.who == "ai"].groupby("T"):
                au, _, _ = cell_stats(g, h, rng, boot=False)
                pts.append((T, au, h.surprisal.mean() - g.surprisal.mean()))
            pts = sorted(pts)
            Dm.append(dict(generator=m, domain=dom, n_ai_per_temperature=len(g), fail_temp_observed=cross([p[0] for p in pts], [p[1] for p in pts], 0.5),
                           fail_temp_predicted=cross([p[0] for p in pts], [p[2] for p in pts])))
    Dm = pd.DataFrame(Dm)
    spread = Dm.groupby("generator").fail_temp_observed.agg(lambda x: x.max() - x.min())

    print("== Per condition (300 AI vs 1,000 human texts; 1% threshold set on 500 humans, realized FPR on the other 500) ==")
    print(R.round(3).to_string(index=False))
    print("\n== E1 failure temperature: predicted vs observed; E2 largest CI half-width near the crossing ==")
    print(X.round(3).to_string(index=False))
    print(f"\n== E5 forward prediction (pilot sigma={SIGMA_OLD}): GPT-2 XL, 27 conditions, mean absolute error {(gx.AUROC_predicted_pilot_sigma - gx.AUROC).abs().mean():.3f}, max {(gx.AUROC_predicted_pilot_sigma - gx.AUROC).abs().max():.3f}; "
          f"self-scored {(R[R.scorer == 'self'].AUROC_predicted_pilot_sigma - R[R.scorer == 'self'].AUROC).abs().mean():.3f}")
    print(f"sigma refitted on the new data = {fit.x:.3f}; Spearman correlation, margin vs AUROC (all 54 conditions) {spearmanr(R.surprisal_margin, R.AUROC).statistic:.3f}")
    print("\n== E4 failure temperature per domain (GPT-2 XL) ==")
    print(Dm.round(3).to_string(index=False))
    print("Largest cross-domain spread of the failure temperature per generator:", spread.round(3).to_dict())
    R.to_csv(f"{RESULTS}/e1_scaled_sweep/metrics.csv", index=False); X.to_csv(f"{RESULTS}/e1_scaled_sweep/crossing.csv", index=False)
    Dm.to_csv(f"{RESULTS}/e1_scaled_sweep/domains.csv", index=False)


if __name__ == "__main__":
    main()
