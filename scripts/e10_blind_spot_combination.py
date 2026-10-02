# Blind-spot combination (design doc §8.8.19; amended in 8.8.19.1 at 21:14 on 2026-09-29, before GPT-Neo and BLOOM text existed).
# Statistics whose population-mean margins vanish at different temperatures are combined without training: two-sided robust scores
# |x - median| / MAD of Fast-DetectGPT and of Likelihood, calibrated on even-id humans, combined by the maximum; AUROC on odd-id
# humans against each temperature's machine texts. Criteria: (1) the combination's lowest AUROC over temperatures exceeds the
# better single two-sided statistic's lowest AUROC by >= 0.05; (2) at T = 0.92 it loses <= 0.03 against one-sided Fast-DetectGPT.
# Test: prospective for gpt-neo-1.3B and bloom-1b7 (both must meet 1 and 2); OLMo-2 and Granite are supplementary.
# Development (exploratory): the same combination on the 06 (self, GPT-2 XL) and 07/09 (Falcon dual) scores.
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1 if 'prepare' not in __file__ else 2]))
from cmargin.paths import DATA, RESULTS
(RESULTS / "e10_blind_spot_combination").mkdir(parents=True, exist_ok=True)
import glob
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

OUT = RESULTS / "e10_blind_spot_combination"
PROSPECTIVE = ("gpt-neo-1.3B", "bloom-1b7")
METHODS = ("FDG one-sided", "FDG two-sided", "Likelihood two-sided", "max two-sided")


def load(f, dual):
    s = pd.read_pickle(f); A = s.arr
    s["FDG"] = [(-a[0].sum() - a[4].sum()) / np.sqrt(a[5].sum()) if dual else (a[1] - a[0]).sum() / np.sqrt(a[2].sum()) for a in A]
    s["Likelihood"] = [-a[0].mean() for a in A]
    s["T"] = [float(d[1:]) if d != "human" else np.nan for d in s.decoding]
    return s.drop(columns=["arr"])


def curves(s):
    h = s[s.who == "human"]; cal, test = h[h.id % 2 == 0], h[h.id % 2 == 1]; ai = s[s.who == "ai"]
    z = lambda fr, m: ((fr[m] - cal[m].median()) / (cal[m] - cal[m].median()).abs().median()).abs()
    score = lambda fr: {"FDG one-sided": fr.FDG, "FDG two-sided": z(fr, "FDG"), "Likelihood two-sided": z(fr, "Likelihood"),
                        "max two-sided": np.maximum(z(fr, "FDG"), z(fr, "Likelihood"))}
    st, rows = score(test), []
    for T in sorted(ai["T"].unique()):
        g = ai[np.isclose(ai["T"], T)]; sg = score(g)
        rows.append({"T": T, **{m: roc_auc_score(np.r_[np.ones(len(g)), np.zeros(len(test))], np.r_[sg[m], st[m]]) for m in METHODS}})
    return pd.DataFrame(rows)


def judge(name, c):
    lo = c[list(METHODS)].min(); best_single = max(lo["FDG two-sided"], lo["Likelihood two-sided"])
    at = c[np.isclose(c["T"], c["T"].min())].iloc[0]
    return dict(setting=name, lowest_T=c["T"].min(), **{f"min_{m}": lo[m] for m in METHODS}, gain=lo["max two-sided"] - best_single,
                loss_at_lowest_T=at["FDG one-sided"] - at["max two-sided"])


def main():
    pd.set_option("display.width", 220)
    dev = []
    for f in sorted(glob.glob(f"{DATA}/06_scaled_sweep/scores_*.pkl")):
        dev.append(judge(f.split("scores_")[1][:-4], curves(load(f, dual=False))))
    for f in sorted(glob.glob(f"{DATA}/07_falcon_dual_scoring/scores_falcon_*.pkl")) + sorted(glob.glob(f"{DATA}/09_heldout_pythia/scores_falcon_*.pkl")):
        dev.append(judge(f.split("scores_")[1][:-4], curves(load(f, dual=True))))
    dev = pd.DataFrame(dev); dev.to_csv(OUT / "development.csv", index=False)
    print("== Development (exploratory) =="); print(dev.round(3).to_string(index=False))
    test_dir = pathlib.Path(DATA) / "13_theory_prediction_test"
    files = sorted(glob.glob(f"{test_dir}/scores_self_*.pkl"))
    if not files:
        print("\nTest data not found yet."); return
    rows, per_t = [], []
    for f in files:
        tag = f.split("scores_self_")[1][:-4]; c = curves(load(f, dual=False))
        per_t.append(c.assign(generator=tag))
        r = judge(tag, c); r["criterion1"] = r["gain"] >= 0.05; r["criterion2"] = r["loss_at_lowest_T"] <= 0.03
        r["role"] = "prospective" if tag in PROSPECTIVE else "supplementary"; rows.append(r)
    test = pd.DataFrame(rows); test.to_csv(OUT / "test.csv", index=False); pd.concat(per_t).to_csv(OUT / "test_curves.csv", index=False)
    print("\n== Test (notebook 13, self-scoring) =="); print(test.round(3).to_string(index=False))
    pro = test[test.role == "prospective"]
    print("\nprospective verdict (gpt-neo and bloom must both meet criteria 1 and 2):",
          "met" if len(pro) == 2 and (pro.criterion1 & pro.criterion2).all() else "not met")


if __name__ == "__main__":
    main()
