# Appendix A: phi-2 and SmolLM2-1.7B pilots; Qwen-fitted gate/stack applied unchanged.
# Pilot 10 analysis (design doc §8.8.9): two new generator families (phi-2, SmolLM2-1.7B).
# Part A: margin vs AUROC per temperature, predicted (margin = 0) vs actual (AUROC = 0.5) crossing,
#         forward AUROC prediction with the probit scale fitted on Qwen pilots (sigma = 0.638).
# Part B: gate / stack trained ONLY on the Qwen2.5-3B run (03_pilot_temperature, GPT-2 XL scoring, all six temperatures),
#         thresholds set on Qwen-run training humans, applied unchanged to each new generator; ids split by fold
#         so the same human text never appears in both training and test.
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1 if 'prepare' not in __file__ else 2]))
from cmargin.paths import DATA, RESULTS
(RESULTS / "a11_new_generators").mkdir(parents=True, exist_ok=True)
import os, tempfile
import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from cmargin.crossing import cross
from cmargin.blindspot import build, FEATS9, ENT

SIGMA = 0.638
FEAT = FEATS9 + ENT


def raw(df):
    df = df.copy()
    df["surprisal"] = [a[0].mean() for a in df.arr]
    df["fdg"] = [(a[1].sum() - a[0].sum()) / np.sqrt(a[2].sum()) for a in df.arr]
    df["y"] = (df.who == "ai").astype(int); df["T"] = df.decoding.str[1:].astype(float)
    return df


def part_a():
    frames = []
    for f in ("scores_gpt2-xl", "scores_phi-2", "scores_SmolLM2-1.7B"):
        frames.append(raw(pd.read_pickle(f"{DATA}/04_pilot_new_generators/{f}.pkl")))
    d = pd.concat(frames, ignore_index=True)
    d["model"] = d.model.str.split("/").str[-1]; d["scorer"] = d.scorer.str.split("/").str[-1]
    rows = []
    for (m, sc, T), g in d.groupby(["model", "scorer", "T"]):
        h, ai = g[g.y == 0], g[g.y == 1]
        rows.append(dict(generator=m, scorer=sc, temperature=T, surprisal_margin=h.surprisal.mean() - ai.surprisal.mean(), AUROC=roc_auc_score(g.y, g.fdg)))
    R = pd.DataFrame(rows)
    R["AUROC_predicted"] = norm.cdf(R.surprisal_margin / SIGMA)
    X = []
    for (m, sc), g in R.groupby(["generator", "scorer"]):
        g = g.sort_values("temperature")
        X.append(dict(generator=m, scorer=sc, fail_temp_predicted=cross(g.temperature, g.surprisal_margin), fail_temp_observed=cross(g.temperature, g.AUROC, 0.5),
                      monotone_decreasing="yes" if (np.diff(g.AUROC) < 0).all() else "no"))
    X = pd.DataFrame(X)
    gx = R[R.scorer == "gpt2-xl"]
    return R, X, (gx.AUROC_predicted - gx.AUROC).abs().mean(), (gx.AUROC_predicted - gx.AUROC).abs().max()


def lr(tr, te, cols):
    sc = StandardScaler().fit(tr[cols]); m = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(tr[cols]), tr.y)
    return m.decision_function(sc.transform(te[cols]))


def two_sided(h, x):
    med, scl = h.median(), (h.quantile(.75) - h.quantile(.25)) / 1.349
    return np.abs(x - med) / scl


def part_b():
    qwen = build(f"{DATA}/03_pilot_temperature/scores_gpt2-xl.pkl")                       # training source: Qwen run, GPT-2 XL scoring
    new = pd.read_pickle(f"{DATA}/04_pilot_new_generators/scores_gpt2-xl.pkl")
    out = []
    for m, sub in new.groupby("model"):
        tmp = os.path.join(tempfile.gettempdir(), "a11_subset.pkl"); sub.to_pickle(tmp); d = build(tmp)  # build() takes a path; keep data/ read-only
        ids = np.sort(d.id.unique()); folds = list(GroupKFold(5).split(ids, groups=ids))
        temps = sorted(d["T"].unique())
        methods = ["one_sided", "two_sided", "feature_classifier", "stack", "gate"]
        scores = {(T, k): np.zeros((d["T"] == T).sum()) for T in temps for k in methods}
        ys = {T: d[d["T"] == T].y.values for T in temps}
        for T in temps:
            tgt = d[d["T"] == T].reset_index(drop=True)
            for tr_i, te_i in folds:
                tr = qwen[qwen.id.isin(ids[tr_i])].copy()              # all six Qwen temperatures, training ids only
                te_mask = tgt.id.isin(ids[te_i]).values; te = tgt[te_mask].copy()
                h = tr[tr.y == 0].fdg
                tr["two"], te["two"] = two_sided(h, tr.fdg), two_sided(h, te.fdg)
                scores[(T, "one_sided")][te_mask] = te.fdg; scores[(T, "two_sided")][te_mask] = te.two
                feat = lr(tr, te, FEAT); scores[(T, "feature_classifier")][te_mask] = feat
                scores[(T, "stack")][te_mask] = lr(tr, te, FEAT + ["fdg", "two"])
                thr = np.quantile(tr[tr.y == 0].two, 0.9)              # fixed on Qwen-run training humans
                far = te.two.values > thr
                fz = (feat - feat.mean()) / (feat.std() + 1e-9)
                scores[(T, "gate")][te_mask] = np.where(far, 10 + te.two.values, fz)
        for k in methods:
            r = {"generator": m.split("/")[-1], "method": k}
            for T in temps:
                r[f"T{T}"] = roc_auc_score(ys[T], scores[(T, k)])
            r["min"] = min(r[f"T{T}"] for T in temps)
            out.append(r)
    return pd.DataFrame(out)


if __name__ == "__main__":
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 100)
    R, X, mae, mx = part_a()
    print("== A. Margin and AUROC ==")
    print(R.pivot_table(index=["generator", "scorer"], columns="temperature", values=["surprisal_margin", "AUROC"]).round(3).to_string())
    print("\n== A. Failure temperature: predicted (margin = 0) vs observed (AUROC = 0.5) ==")
    print(X.round(3).to_string(index=False))
    print(f"\nA. Forward prediction (sigma={SIGMA}, 12 GPT-2 XL points): mean absolute error {mae:.3f}, max {mx:.3f}")
    B = part_b()
    print("\n== B. Trained and thresholded on Qwen, applied unchanged to the new generators (GPT-2 XL scorer) ==")
    print(B.round(3).to_string(index=False))
    R.to_csv(f"{RESULTS}/a11_new_generators/metrics.csv", index=False); X.to_csv(f"{RESULTS}/a11_new_generators/crossing.csv", index=False)
    B.to_csv(f"{RESULTS}/a11_new_generators/transfer.csv", index=False)
