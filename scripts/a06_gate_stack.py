# Appendix A (gate and stacking): leave-one-temperature-out combination of zero-shot score and features.
# Gate and stack (design doc §8.8.4): anchored combination, leave-one-temperature-out on 03_pilot_temperature data.
# Methods: Fast-DetectGPT one-sided, two-sided, feature LR (9 + entropy), stacked LR (features + one-sided
# + two-sided), gate (two-sided if far from the human centre, else feature LR). Human ids split by fold.
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1 if 'prepare' not in __file__ else 2]))
from cmargin.paths import DATA, RESULTS
(RESULTS / "a06_gate_stack").mkdir(parents=True, exist_ok=True)
import glob
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from cmargin.blindspot import build, FEATS9, ENT

N_BOOT, FPR = 1000, 0.05
FEAT = FEATS9 + ENT


def lr(tr, te, cols):
    sc = StandardScaler().fit(tr[cols]); m = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(tr[cols]), tr.y)
    return m.decision_function(sc.transform(te[cols]))


def two_sided(h_fdg, x):
    med, scl = h_fdg.median(), (h_fdg.quantile(.75) - h_fdg.quantile(.25)) / 1.349
    return np.abs(x - med) / scl


def evaluate(d, rng):
    d = d.copy()
    ids = np.sort(d.id.unique()); folds = list(GroupKFold(5).split(ids, groups=ids))
    temps = sorted(d["T"].unique())
    methods = ["one_sided", "two_sided", "feature_classifier", "stack", "gate"]
    scores = {(T, m): np.zeros((d["T"] == T).sum()) for T in temps for m in methods}
    ys = {T: d[d["T"] == T].y.values for T in temps}
    for T in temps:
        tgt = d[d["T"] == T].reset_index(drop=True)
        rest = d[d["T"] != T]
        for tr_i, te_i in folds:
            tr = rest[rest.id.isin(ids[tr_i])].copy()
            te_mask = tgt.id.isin(ids[te_i]).values; te = tgt[te_mask].copy()
            h = tr[tr.y == 0].fdg
            tr["two"], te["two"] = two_sided(h, tr.fdg), two_sided(h, te.fdg)
            scores[(T, "one_sided")][te_mask] = te.fdg
            scores[(T, "two_sided")][te_mask] = te.two
            feat = lr(tr, te, FEAT)
            scores[(T, "feature_classifier")][te_mask] = feat
            scores[(T, "stack")][te_mask] = lr(tr, te, FEAT + ["fdg", "two"])
            # gate: far from the human centre -> trust the two-sided score; otherwise the feature classifier
            thr = np.quantile(tr[tr.y == 0].two, 0.9)
            far = te.two.values > thr
            feat_z = (feat - feat.mean()) / (feat.std() + 1e-9)
            scores[(T, "gate")][te_mask] = np.where(far, 10 + te.two.values, feat_z)   # far texts rank above the rest
    rows = []
    for m in methods:
        r = {"method": m}
        for T in temps:
            y, s = ys[T], scores[(T, m)]
            r[f"T{T}"] = roc_auc_score(y, s)
        r["min"] = min(r[f"T{T}"] for T in temps)
        rows.append(r)
    R = pd.DataFrame(rows)
    # TPR at 5% human FPR, threshold from the test temperature's own humans (pilot-scale approximation)
    rows = []
    for m in methods:
        r = {"method": m}
        for T in temps:
            y, s = ys[T], scores[(T, m)]
            thr = np.quantile(s[y == 0], 1 - FPR)
            r[f"T{T}"] = (s[y == 1] > thr).mean()
        rows.append(r)
    P = pd.DataFrame(rows)
    # bootstrap CI for "min" of the two combinations vs the best single method
    best_single = {T: max(roc_auc_score(ys[T], scores[(T, m)]) for m in methods[:3]) for T in temps}
    ci = {}
    for m in ("stack", "gate"):
        boot = []
        for _ in range(N_BOOT):
            vals = []
            for T in temps:
                i = rng.integers(0, len(ys[T]), len(ys[T]))
                vals.append(roc_auc_score(ys[T][i], scores[(T, m)][i]) - max(roc_auc_score(ys[T][i], scores[(T, k)][i]) for k in methods[:3]))
            boot.append(min(vals))
        ci[m] = (np.percentile(boot, 2.5), np.percentile(boot, 97.5))
    return R, P, ci


def main():
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 250)
    for sc in ("gpt2-xl", "gpt2", "Qwen2.5-3B"):
        d = build(f"{DATA}/03_pilot_temperature/scores_{sc}.pkl")
        R, P, ci = evaluate(d, rng)
        print(f"\n== Scorer {sc}: leave-one-temperature-out, AUROC ==")
        print(R.round(3).to_string(index=False))
        print(f"== TPR at 5% human FPR ==")
        print(P.round(3).to_string(index=False))
        for m, (lo, hi) in ci.items():
            print(f"{m} minus the best single method at each temperature, 95% CI of the minimum difference over six temperatures: [{lo:.3f}, {hi:.3f}]")
        R.to_csv(f"{RESULTS}/a06_gate_stack/auroc_{sc}.csv", index=False); P.to_csv(f"{RESULTS}/a06_gate_stack/tpr_{sc}.csv", index=False)


if __name__ == "__main__":
    main()
