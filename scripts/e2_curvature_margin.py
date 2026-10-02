# Section 5.2 / Table 4: curvature-margin vs surprisal-margin prediction on the scaled sweep.
# Post hoc analysis of main experiment 1: the surprisal margin predicts the crossing with a systematic bias (predicted
# temperature below actual). The Fast-DetectGPT statistic also carries the scorer-entropy term, so test the
# curvature margin  E_h[H - s] - E_g[H - s]  as the predictor (still computable before looking at AUROC).
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1 if 'prepare' not in __file__ else 2]))
from cmargin.paths import DATA, RESULTS
(RESULTS / "e2_curvature_margin").mkdir(parents=True, exist_ok=True)
import glob
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from cmargin.crossing import cross


def main():
    rows = []
    for f in sorted(glob.glob(f"{DATA}/06_scaled_sweep/scores_*.pkl")):
        s = pd.read_pickle(f)
        sk = "GPT-2 XL" if "gpt2-xl" in f else "self"
        s["surprisal"] = [a[0].mean() for a in s.arr]; s["entropy"] = [a[1].mean() for a in s.arr]
        s["curv"] = s.entropy - s.surprisal
        s["fdg"] = [(a[1].sum() - a[0].sum()) / np.sqrt(a[2].sum()) for a in s.arr]
        h = s[s.who == "human"]; m = s.model.iloc[0].split("/")[-1]
        for dec, g in s[s.who == "ai"].groupby("decoding"):
            y = np.r_[np.ones(len(g)), np.zeros(len(h))]
            rows.append(dict(generator=m, scorer=sk, temperature=float(dec[1:]), surprisal_margin=h.surprisal.mean() - g.surprisal.mean(),
                             entropy_gap=h.entropy.mean() - g.entropy.mean(), curvature_margin=g.curv.mean() - h.curv.mean(),
                             AUROC=roc_auc_score(y, np.r_[g.fdg.values, h.fdg.values])))
    R = pd.DataFrame(rows).sort_values(["generator", "scorer", "temperature"])
    X = []
    for (m, sk), g in R.groupby(["generator", "scorer"]):
        pc, cc, ac = cross(g.temperature, g.surprisal_margin), cross(g.temperature, g.curvature_margin), cross(g.temperature, g.AUROC, 0.5)
        near = g.iloc[(g.AUROC - 0.5).abs().argsort()[:2]]
        X.append(dict(generator=m, scorer=sk, fail_temp_observed=ac, surprisal_margin_pred=pc, surprisal_margin_error=ac - pc, curvature_margin_pred=cc, curvature_margin_error=ac - cc,
                      entropy_gap_near_crossing=near.entropy_gap.mean()))
    X = pd.DataFrame(X)
    pd.set_option("display.width", 250)
    print("== Post hoc: predicting the failure temperature with the curvature margin (AI minus human mean of entropy - surprisal) ==")
    print(X.round(3).to_string(index=False))
    print(f"\nMean error of the surprisal-margin prediction {X.surprisal_margin_error.mean():+.3f} (mean absolute {X.surprisal_margin_error.abs().mean():.3f}); curvature-margin prediction {X.curvature_margin_error.mean():+.3f} (mean absolute {X.curvature_margin_error.abs().mean():.3f})")
    X.to_csv(f"{RESULTS}/e2_curvature_margin/crossing_curvature.csv", index=False)


if __name__ == "__main__":
    main()
