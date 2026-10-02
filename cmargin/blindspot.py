# Shared blind-spot feature builder (from the pilot feature study); run scripts/a05_blindspot_features.py.
# Blind-spot features (design doc §8.8.2): on the blind spot (T = 1.0) can trained features detect what
# curvature scores cannot? Data: 03_pilot_temperature per-position arrays (rows 0 surprisal, 1 entropy, 2 var, 3 top-10).
# Two evaluations, both grouped by text id so a human text and its AI continuation never straddle a fold:
#   within : 5-fold CV on the T = 1.0 set only
#   transfer: train on the other five temperatures, test on T = 1.0, human ids split by fold
from cmargin.paths import DATA, RESULTS
import glob
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from cmargin.diveye import feats, FEATS9

N_BOOT, TARGET = 1000, 1.0
ENT = ["ent_mean", "ent_std", "m_mean", "m_std"]
SETS = {"DivEye-9": FEATS9, "DivEye-9+entropy": FEATS9 + ENT, "DivEye-9+entropy+FDG": FEATS9 + ENT + ["fdg"], "entropy-4": ENT}
XGB = dict(max_depth=4, n_estimators=200, subsample=0.7, colsample_bytree=0.8, min_child_weight=5, gamma=1.0,
           learning_rate=0.1, random_state=0, n_jobs=4)


def build(path):
    s = pd.read_pickle(path)
    rows = []
    for r in s.itertuples():
        a = r.arr.astype(np.float64); L, E, M = a[0], a[1], a[3]
        f = dict(zip(FEATS9, feats(L, np.ones(len(L), bool))))
        f.update(ent_mean=E.mean(), ent_std=E.std(), m_mean=M.mean(), m_std=M.std(),
                 fdg=(a[1].sum() - a[0].sum()) / np.sqrt(a[2].sum()),
                 id=r.id, who=r.who, y=int(r.who == "ai"),
                 T=float(r.decoding.replace("pure", "T1.0")[1:]), scorer=r.scorer.split("/")[-1])
        rows.append(f)
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def models():
    return {"XGB": lambda: XGBClassifier(**XGB), "LR": lambda: None}


def fit_predict(kind, cols, tr, te):
    if kind == "XGB":
        m = XGBClassifier(**XGB).fit(tr[cols].values, tr.y)
        return m.predict(te[cols].values, output_margin=True)
    sc = StandardScaler().fit(tr[cols]); m = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(tr[cols]), tr.y)
    return m.decision_function(sc.transform(te[cols]))


def evaluate(d, rng):
    """d: one scorer. Returns pooled out-of-fold scores per method for the T = TARGET test set."""
    tgt = d[d["T"] == TARGET].copy()
    ids = np.sort(tgt.id.unique())
    folds = list(GroupKFold(5).split(ids, groups=ids))
    out = {}
    for setting in ("within_blind_spot", "cross_temperature"):
        for name, cols in SETS.items():
            for kind in ("XGB", "LR"):
                out[(setting, f"{kind} {name}")] = np.zeros(len(tgt))
        out[(setting, "Fast-DetectGPT one-sided")] = tgt.fdg.values.copy()
        out[(setting, "Fast-DetectGPT two-sided")] = np.zeros(len(tgt))
    for tr_i, te_i in folds:
        tr_ids, te_ids = ids[tr_i], ids[te_i]
        te_mask = tgt.id.isin(te_ids).values
        te = tgt[te_mask]
        train = {"within_blind_spot": tgt[tgt.id.isin(tr_ids)],
                 "cross_temperature": d[(d["T"] != TARGET) & d.id.isin(tr_ids)]}
        for setting, tr in train.items():
            for name, cols in SETS.items():
                for kind in ("XGB", "LR"):
                    out[(setting, f"{kind} {name}")][te_mask] = fit_predict(kind, cols, tr, te)
            h = tr[tr.y == 0].fdg
            med, scl = h.median(), (h.quantile(.75) - h.quantile(.25)) / 1.349
            out[(setting, "Fast-DetectGPT two-sided")][te_mask] = np.abs(te.fdg.values - med) / scl
    y = tgt.y.values
    rows = []
    for (setting, meth), sc in out.items():
        boot = [roc_auc_score(y[i], sc[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(N_BOOT))]
        rows.append(dict(evaluation=setting, method=meth, AUROC=roc_auc_score(y, sc),
                         CI=f"[{np.percentile(boot, 2.5):.3f}, {np.percentile(boot, 97.5):.3f}]"))
    return pd.DataFrame(rows)


def main():
    (RESULTS / "a05_blindspot_features").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    D = {p.split("scores_")[1][:-4]: build(p) for p in sorted(glob.glob(f"{DATA}/03_pilot_temperature/scores_*.pkl"))}
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
    allr = []
    for sc in ("gpt2-xl", "gpt2", "Qwen2.5-3B"):
        R = evaluate(D[sc], rng); R.insert(0, "scorer", sc); allr.append(R)
        print(f"\n== Scorer {sc}, test temperature {TARGET} ==")
        print(R.round(3).to_string(index=False))
    A = pd.concat(allr, ignore_index=True)
    A.to_csv(f"{RESULTS}/a05_blindspot_features/metrics.csv", index=False)

    # post-hoc diagnostic: which single features separate human from AI at T = 1.0 (GPT-2 XL)?
    t = D["gpt2-xl"]; t = t[t["T"] == TARGET]
    diag = {c: roc_auc_score(t.y, t[c]) for c in FEATS9 + ENT + ["fdg"]}
    S = pd.Series(diag).sort_values()
    print("\n== Additional diagnostic: T = 1.0, GPT-2 XL scorer, AUROC of each single feature (below 0.5 means lower for AI) ==")
    print(S.round(3).to_string())
    S.to_csv(f"{RESULTS}/a05_blindspot_features/single_feature.csv")


if __name__ == "__main__":
    main()
