# Appendix A (paraphrase): paraphrase groups vs human originals, whole-text scoring.
# Pilot 11 analysis (design doc §8.8.10): paraphrase sets vs human originals, whole-text scoring.
# Part A: per group (source x paraphrase decoding) margin, one-sided AUROC, two-sided AUROC, forward prediction.
# Part B: stack / gate trained on the Qwen continuation run (03_pilot_temperature, GPT-2 XL, continuation-only features) applied
#         unchanged to the whole-text paraphrase sets; ids split by fold. Features on whole text = approximation.
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1 if 'prepare' not in __file__ else 2]))
from cmargin.paths import DATA, RESULTS
(RESULTS / "a12_paraphrase").mkdir(parents=True, exist_ok=True)
import glob
import numpy as np, pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from cmargin.diveye import feats, FEATS9
from cmargin.blindspot import build as build5, ENT

SIGMA, N_BOOT = 0.638, 1000
FEAT = FEATS9 + ENT
ORDER = ["ai_T0.8", "ai_T1.0", "para_human_greedy", "para_human_sample", "para_ai_T0.8_greedy", "para_ai_T0.8_sample",
         "para_ai_T1.0_greedy", "para_ai_T1.0_sample"]
NAMES = {"ai_T0.8": "AI continuation T0.8 (not paraphrased, reference)", "ai_T1.0": "AI continuation T1.0 (not paraphrased, blind spot, reference)",
         "para_human_greedy": "human original -> greedy paraphrase", "para_human_sample": "human original -> sampled paraphrase",
         "para_ai_T0.8_greedy": "AI T0.8 -> greedy paraphrase", "para_ai_T0.8_sample": "AI T0.8 -> sampled paraphrase",
         "para_ai_T1.0_greedy": "AI T1.0 -> greedy paraphrase", "para_ai_T1.0_sample": "AI T1.0 -> sampled paraphrase"}


def build11(path):
    s = pd.read_pickle(path)
    rows = []
    for r in s.itertuples():
        a = r.arr.astype(np.float64); L, E, M = a[0], a[1], a[3]
        f = dict(zip(FEATS9, feats(L, np.ones(len(L), bool))))
        f.update(ent_mean=E.mean(), ent_std=E.std(), m_mean=M.mean(), m_std=M.std(), surprisal=L.mean(),
                 fdg=(a[1].sum() - a[0].sum()) / np.sqrt(a[2].sum()), id=r.id, group=r.group, y=int(r.group != "human"))
        rows.append(f)
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def two_sided(h, x):
    med, scl = h.median(), (h.quantile(.75) - h.quantile(.25)) / 1.349
    return np.abs(x - med) / scl


def lr(tr, te, cols):
    sc = StandardScaler().fit(tr[cols]); m = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(tr[cols]), tr.y)
    return m.decision_function(sc.transform(te[cols]))


def part_a(d, scorer, rng):
    h = d[d.group == "human"]
    rows = []
    for g in ORDER:
        x = d[d.group == g]
        y = np.r_[np.ones(len(x)), np.zeros(len(h))]
        one = np.r_[x.fdg.values, h.fdg.values]
        two = np.r_[two_sided(h.fdg, x.fdg.values), two_sided(h.fdg, h.fdg.values)]
        boot = [roc_auc_score(y[i], one[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(N_BOOT))]
        m = h.surprisal.mean() - x.surprisal.mean()
        rows.append(dict(scorer=scorer, group=NAMES[g], n_texts=len(x), surprisal_margin=m, AUROC_one_sided=roc_auc_score(y, one),
                         CI=f"[{np.percentile(boot, 2.5):.3f}, {np.percentile(boot, 97.5):.3f}]",
                         AUROC_two_sided=roc_auc_score(y, two), AUROC_predicted=norm.cdf(m / SIGMA)))
    return pd.DataFrame(rows)


def part_b(d):
    """Train on the Qwen continuation run (all temperatures), test on the paraphrase sets; ids split by fold."""
    qwen = build5(f"{DATA}/03_pilot_temperature/scores_gpt2-xl.pkl")
    ids = np.sort(d.id.unique()); folds = list(GroupKFold(5).split(ids, groups=ids))
    h_all = d[d.group == "human"]
    out = []
    for g in ORDER:
        x = d[d.group == g]
        te_all = pd.concat([x, h_all]).reset_index(drop=True)
        sc = {k: np.zeros(len(te_all)) for k in ("two_sided", "feature_classifier", "stack", "gate")}
        for tr_i, te_i in folds:
            tr = qwen[qwen.id.isin(ids[tr_i])].copy()
            te_mask = te_all.id.isin(ids[te_i]).values; te = te_all[te_mask].copy()
            hh = tr[tr.y == 0].fdg
            tr["two"], te["two"] = two_sided(hh, tr.fdg), two_sided(hh, te.fdg)
            sc["two_sided"][te_mask] = te.two
            feat = lr(tr, te, FEAT); sc["feature_classifier"][te_mask] = feat
            sc["stack"][te_mask] = lr(tr, te, FEAT + ["fdg", "two"])
            thr = np.quantile(tr[tr.y == 0].two, 0.9); far = te.two.values > thr
            fz = (feat - feat.mean()) / (feat.std() + 1e-9)
            sc["gate"][te_mask] = np.where(far, 10 + te.two.values, fz)
        r = {"group": NAMES[g]}
        for k, v in sc.items():
            r[k] = roc_auc_score(te_all.y, v)
        out.append(r)
    return pd.DataFrame(out)


def main():
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 100)
    allA = []
    for scorer in ("gpt2-xl", "Qwen2.5-3B"):
        d = build11(f"{DATA}/05_pilot_paraphrase/scores_{scorer}.pkl")
        A = part_a(d, scorer, rng); allA.append(A)
        print(f"\n== A. Scorer {scorer}: each group vs the human originals (whole-text scoring) ==")
        print(A.round(3).to_string(index=False))
        para = A[A.group.str.contains("->")]   # the six paraphrase groups (the two references are not paraphrased)
        rho = spearmanr(para.surprisal_margin, para.AUROC_one_sided)
        print(f"Six paraphrase groups: Spearman correlation, margin vs one-sided AUROC {rho.statistic:.3f} (p={rho.pvalue:.3f}); forward-prediction mean absolute error {(para.AUROC_predicted - para.AUROC_one_sided).abs().mean():.3f}")
        if scorer == "gpt2-xl":
            B = part_b(d)
            print("\n== B. Combinations trained on the Qwen continuations, applied to the paraphrase groups (GPT-2 XL) ==")
            print(B.round(3).to_string(index=False))
            B.to_csv(f"{RESULTS}/a12_paraphrase/transfer.csv", index=False)
    pd.concat(allA).to_csv(f"{RESULTS}/a12_paraphrase/metrics.csv", index=False)


if __name__ == "__main__":
    main()
