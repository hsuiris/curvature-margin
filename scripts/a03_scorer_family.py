# Appendix A (scorer family): same texts scored by the generator itself; margin decomposition.
# Pilot 4 analysis plus the first test of the predicted-failure-point idea.
# Data: 01_pilot_decoding/scores.pkl (scorer = GPT-2 XL) and 02_pilot_scorer_family/scores_*.pkl (scorer = the generator itself).
# Rows per position: 0 surprisal, 1 entropy, 2 variance of log-prob, 3 top-10 mass.
# Prediction under test: the detector's margin, and therefore its AUROC, follows
#   margin = (human surprisal under the scorer) - (effective entropy of the generator + mismatch),
# so AUROC should be predictable from the mean and spread of the per-text scores alone.
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1 if 'prepare' not in __file__ else 2]))
from cmargin.paths import DATA, RESULTS
(RESULTS / "a03_scorer_family").mkdir(parents=True, exist_ok=True)
import glob
import numpy as np, pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.metrics import roc_auc_score

N_BOOT = 1000


def feats(s, scorer):
    a = s.arr
    s = s.copy()
    s["scorer"] = scorer
    s["fdg"] = [(x[1].sum() - x[0].sum()) / np.sqrt(x[2].sum()) for x in a]
    s["surprisal"] = [x[0].mean() for x in a]
    s["entropy"] = [x[1].mean() for x in a]
    s["y"] = (s.who == "ai").astype(int)
    return s.drop(columns=["arr"])


def load():
    p3 = feats(pd.read_pickle(f"{DATA}/01_pilot_decoding/scores.pkl"), "GPT-2 XL")
    p4 = pd.concat([feats(pd.read_pickle(f), "self") for f in sorted(glob.glob(f"{DATA}/02_pilot_scorer_family/scores_Qwen2.5-3B*.pkl")) +
                    sorted(glob.glob(f"{DATA}/02_pilot_scorer_family/scores_opt-2.7b.pkl"))], ignore_index=True)
    return pd.concat([p3, p4], ignore_index=True)


def main():
    rng = np.random.default_rng(0)
    d = load()
    rows = []
    for (model, dec, scorer), g in d.groupby(["model", "decoding", "scorer"]):
        h, ai = g[g.y == 0], g[g.y == 1]
        auroc = roc_auc_score(g.y, g.fdg)
        # normal approximation: AUROC = Phi(margin / sqrt(var_ai + var_human))
        margin = ai.fdg.mean() - h.fdg.mean()
        pred = norm.cdf(margin / np.sqrt(ai.fdg.var(ddof=1) + h.fdg.var(ddof=1)))
        boot = [roc_auc_score(g.y.values[i], g.fdg.values[i]) for i in
                (rng.integers(0, len(g), len(g)) for _ in range(N_BOOT))]
        rows.append(dict(generator=model.split("/")[-1], decoding=dec, scorer=scorer, AUROC=auroc,
                         CI=f"[{np.percentile(boot, 2.5):.3f}, {np.percentile(boot, 97.5):.3f}]",
                         AUROC_predicted=pred, error=pred - auroc,
                         surprisal_ai=ai.surprisal.mean(), surprisal_human=h.surprisal.mean(),
                         scorer_entropy=ai.entropy.mean(), score_ai=ai.fdg.mean(), score_human=h.fdg.mean()))
    R = pd.DataFrame(rows).sort_values(["generator", "decoding", "scorer"])

    # decomposition: effective entropy (surprisal under its own model) + mismatch (extra surprisal under GPT-2 XL)
    dec_rows = []
    for (model, dec), g in d[d.y == 1].groupby(["model", "decoding"]):
        own = g[g.scorer == "self"].surprisal.mean()
        mism = g[g.scorer == "GPT-2 XL"].surprisal.mean() - own
        hum = d[(d.y == 0) & (d.scorer == "GPT-2 XL") & (d.model == model) & (d.decoding == dec)].surprisal.mean()
        au = R[(R.generator == model.split("/")[-1]) & (R.decoding == dec) & (R.scorer == "GPT-2 XL")].AUROC.iloc[0]
        dec_rows.append(dict(generator=model.split("/")[-1], decoding=dec, effective_entropy=own, family_increase=mism,
                             surprisal_ai=own + mism, surprisal_human=hum, surprisal_margin=hum - (own + mism),
                             AUROC_GPT2XL_scorer=au))
    D = pd.DataFrame(dec_rows).sort_values("surprisal_margin")

    rho_all = spearmanr(R.AUROC, R.AUROC_predicted)
    rho_margin = spearmanr(D.surprisal_margin, D.AUROC_GPT2XL_scorer)
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 30)
    print("== Per condition: observed AUROC and the normal-approximation prediction ==")
    print(R.round(3).to_string(index=False))
    print(f"\nSpearman correlation, predicted vs observed: {rho_all.statistic:.3f} (p={rho_all.pvalue:.4f}); mean absolute error {R.error.abs().mean():.3f}")
    print("\n== Decomposition: effective entropy, increase from the scorer family, and margin to human texts (scorer GPT-2 XL) ==")
    print(D.round(3).to_string(index=False))
    print(f"\nSpearman correlation, margin vs AUROC: {rho_margin.statistic:.3f} (p={rho_margin.pvalue:.4f})")
    R.to_csv(f"{RESULTS}/a03_scorer_family/metrics.csv", index=False)
    D.to_csv(f"{RESULTS}/a03_scorer_family/decomposition.csv", index=False)


if __name__ == "__main__":
    main()
