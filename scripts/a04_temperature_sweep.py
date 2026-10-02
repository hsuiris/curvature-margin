# Pilot temperature sweep of Qwen2.5-3B with three scorers; first margin-based crossing prediction.
# Pilot 5 analysis (design doc §8.8): temperature sweep of Qwen2.5-3B (top-p 1.0), three scorers.
# Pre-registered checks: (1) AUROC falls with temperature and crosses 0.5 near margin = 0;
# (2) the crossing temperature differs by scorer; (3) two-sided ensemble over scorers >= best single - 0.02.
# Forward prediction: probit scale fitted on the six pilot-3/4 GPT-2 XL conditions, applied to the 18 new points.
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1 if 'prepare' not in __file__ else 2]))
from cmargin.paths import DATA, RESULTS
(RESULTS / "a04_temperature_sweep").mkdir(parents=True, exist_ok=True)
import glob
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm, spearmanr
from sklearn.metrics import roc_auc_score

N_BOOT = 1000


def feats(s):
    s = s.copy(); a = s.arr
    s["fdg"] = [(x[1].sum() - x[0].sum()) / np.sqrt(x[2].sum()) for x in a]
    s["surprisal"] = [x[0].mean() for x in a]
    s["y"] = (s.who == "ai").astype(int)
    s["T"] = s.decoding.replace("pure", "T1.0").str[1:].astype(float)   # T=1.0 file was copied from pilot 3
    return s.drop(columns=["arr"])


def cross(x, y, level=0.0):
    """Temperature where y crosses `level`, linear interpolation on the sorted grid; None if no sign change."""
    x, y = np.asarray(x), np.asarray(y) - level
    for i in range(len(x) - 1):
        if y[i] > 0 >= y[i + 1] or y[i] < 0 <= y[i + 1]:
            return x[i] + (x[i + 1] - x[i]) * y[i] / (y[i] - y[i + 1])
    return None


def main():
    rng = np.random.default_rng(0)
    d = pd.concat([feats(pd.read_pickle(f)) for f in sorted(glob.glob(f"{DATA}/03_pilot_temperature/scores_*.pkl"))], ignore_index=True)
    d["scorer"] = d.scorer.str.split("/").str[-1]

    rows = []
    for (sc, T), g in d.groupby(["scorer", "T"]):
        h, ai = g[g.y == 0], g[g.y == 1]
        boot = [roc_auc_score(g.y.values[i], g.fdg.values[i]) for i in
                (rng.integers(0, len(g), len(g)) for _ in range(N_BOOT))]
        rows.append(dict(scorer=sc, temperature=T, surprisal_margin=h.surprisal.mean() - ai.surprisal.mean(),
                         AUROC=roc_auc_score(g.y, g.fdg), CI_low=np.percentile(boot, 2.5), CI_high=np.percentile(boot, 97.5)))
    R = pd.DataFrame(rows).sort_values(["scorer", "temperature"])

    # (1)(2) predicted vs actual crossing temperature per scorer
    X = []
    for sc, g in R.groupby("scorer"):
        X.append(dict(scorer=sc, fail_temp_predicted=cross(g.temperature, g.surprisal_margin), fail_temp_observed=cross(g.temperature, g.AUROC, 0.5),
                      AUROC_monotone_decreasing="yes" if (np.diff(g.AUROC) < 0).all() else "no"))
    X = pd.DataFrame(X)
    rho = spearmanr(R.surprisal_margin, R.AUROC)

    # forward prediction: fit one probit scale on pilot 3/4 GPT-2 XL conditions, then predict all 18 new points
    P = pd.read_csv(f"{RESULTS}/a03_scorer_family/decomposition.csv")
    fit = minimize_scalar(lambda s: ((norm.cdf(P.surprisal_margin / s) - P.AUROC_GPT2XL_scorer) ** 2).sum(), bounds=(0.05, 5), method="bounded")
    R["AUROC_predicted"] = norm.cdf(R.surprisal_margin / fit.x)
    R["error"] = R.AUROC_predicted - R.AUROC

    # (3) two-sided per scorer, calibrated on the 100 human texts of that scorer, then max over scorers
    Z = {}
    for sc, g in d.groupby("scorer"):
        h = g[g.y == 0].fdg
        med, scl = h.median(), (h.quantile(.75) - h.quantile(.25)) / 1.349
        Z[sc] = g.assign(z=(g.fdg - med).abs() / scl)[["id", "T", "who", "y", "z"]]
    two = []
    for T in sorted(d["T"].unique()):
        r = {"temperature": T}
        parts = {sc: z[z["T"] == T].set_index(["id", "who"]) for sc, z in Z.items()}
        for sc, z in parts.items():
            r[f"two_sided_{sc}"] = roc_auc_score(z.y, z.z)
        common = pd.concat([z.z.rename(sc) for sc, z in parts.items()], axis=1, join="inner")
        yy = parts["gpt2-xl"].loc[common.index].y
        r["two_sided_max_of_three"] = roc_auc_score(yy, common.max(axis=1))
        r["best_single_one_sided"] = R[R.temperature == T].AUROC.max()
        r["best_single_two_sided"] = max(r[f"two_sided_{sc}"] for sc in parts)
        r["combination_minus_best_single"] = r["two_sided_max_of_three"] - r["best_single_two_sided"]
        two.append(r)
    W = pd.DataFrame(two)

    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 30)
    print("== Per scorer x temperature: margin, AUROC, forward prediction ==")
    print(R.round(3).to_string(index=False))
    print(f"\nSpearman correlation, margin vs AUROC: {rho.statistic:.3f} (p={rho.pvalue:.2e})")
    print(f"Forward prediction: probit scale sigma={fit.x:.3f} (fitted on the 6 conditions of 01/02); mean absolute error on the 18 new conditions {R.error.abs().mean():.3f}, max {R.error.abs().max():.3f}")
    print("\n== Failure temperature: predicted (margin = 0) vs observed (AUROC = 0.5) ==")
    print(X.round(3).to_string(index=False))
    print("\n== Two-sided scores and the max-combination of three scorers ==")
    print(W.round(3).to_string(index=False))
    R.to_csv(f"{RESULTS}/a04_temperature_sweep/metrics.csv", index=False)
    X.to_csv(f"{RESULTS}/a04_temperature_sweep/crossing.csv", index=False)
    W.to_csv(f"{RESULTS}/a04_temperature_sweep/two_sided.csv", index=False)


def check():
    assert abs(cross([0.9, 1.0], [0.5, -0.5]) - 0.95) < 1e-9
    assert cross([0.9, 1.0], [0.5, 0.2]) is None
    print("check ok")


if __name__ == "__main__":
    import sys
    check() if sys.argv[1:] == ["check"] else main()
