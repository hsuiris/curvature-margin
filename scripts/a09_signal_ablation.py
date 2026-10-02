# Appendix A (signals on the blind spot): ablation by token signal and statistic group.
# Pilot 8 (design doc §8.8.7): which token-level signal and which statistic type carry the
# signal on the blind spot? Data: 03_pilot_temperature arrays (rows 0 surprisal, 1 entropy, 2 var, 3 top-10).
# Signals: s (surprisal), e (entropy), ds (first difference of s), d (e - s, per-position Fast-DetectGPT numerator).
# Groups: mean-location, dispersion, tail, sequence. Two evaluations as in a05_blindspot_features.py: within T=1.0 (5-fold
# grouped CV) and leave-one-temperature-out. Baseline: a SurpMark-style state-transition score (own implementation).
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1 if 'prepare' not in __file__ else 2]))
from cmargin.paths import DATA, RESULTS
(RESULTS / "a09_signal_ablation").mkdir(parents=True, exist_ok=True)
import glob, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

TARGET, K = 1.0, 4
GROUPS = {"location": ["mean", "median"], "dispersion": ["std", "iqr", "mad"], "tails": ["max", "p90", "p95"],
          "sequential": ["dvar", "acf1", "madiff"]}


def stats(x):
    x = np.asarray(x, dtype=np.float64)
    d = np.diff(x)
    med = np.median(x)
    acf = np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 2 and x.std() > 0 else 0.0
    return dict(mean=x.mean(), median=med, std=x.std(), iqr=np.percentile(x, 75) - np.percentile(x, 25),
                mad=np.median(np.abs(x - med)), max=x.max(), p90=np.percentile(x, 90), p95=np.percentile(x, 95),
                dvar=d.var() if len(d) > 1 else 0.0, acf1=acf, madiff=np.abs(d).mean() if len(d) else 0.0)


def build(path):
    s = pd.read_pickle(path)
    rows, seqs = [], []
    for r in s.itertuples():
        a = r.arr.astype(np.float64)
        sig = {"s": a[0], "e": a[1], "ds": np.diff(a[0]), "d": a[1] - a[0]}
        f = {f"{k}_{n}": v for k, x in sig.items() for n, v in stats(x).items()}
        f.update(id=r.id, y=int(r.who == "ai"), T=float(r.decoding.replace("pure", "T1.0")[1:]))
        rows.append(f); seqs.append(a[0])
    df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["s_seq"] = seqs   # raw surprisal sequences for the state-transition baseline (object column, kept apart)
    return df


def lr(tr, te, cols):
    sc = StandardScaler().fit(tr[cols]); m = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(tr[cols]), tr.y)
    return m.decision_function(sc.transform(te[cols]))


# ---- SurpMark-style baseline: surprisal states -> transition distribution -> GJS gap to references ----
def transitions(seq, edges):
    st = np.digitize(seq, edges)                     # 0..K-1
    P = np.zeros((K, K))
    for a, b in zip(st[:-1], st[1:]):
        P[a, b] += 1
    return (P / max(P.sum(), 1)).ravel()              # joint distribution over (state_t, state_t+1)


def gjs(p, q, w=0.5):
    def H(x):
        x = x[x > 0]; return -(x * np.log(x)).sum()
    return H(w * p + (1 - w) * q) - w * H(p) - (1 - w) * H(q)


def surpmark_scores(tr, te):
    edges = np.percentile(np.concatenate(tr[tr.y == 0].s_seq.values), [25, 50, 75])
    Ttr = np.stack([transitions(x, edges) for x in tr.s_seq]); Tte = np.stack([transitions(x, edges) for x in te.s_seq])
    ref_h, ref_m = Ttr[tr.y.values == 0].mean(0), Ttr[tr.y.values == 1].mean(0)
    score = np.array([gjs(t, ref_h) - gjs(t, ref_m) for t in Tte])     # closer to the machine reference = higher
    cols = [f"t{i}" for i in range(K * K)]
    trf = pd.DataFrame(Ttr, columns=cols).assign(y=tr.y.values); tef = pd.DataFrame(Tte, columns=cols)
    feat = lr(trf, tef, cols)
    return score, feat


def evaluate(d, rng):
    tgt = d[d["T"] == TARGET].reset_index(drop=True)
    ids = np.sort(tgt.id.unique()); folds = list(GroupKFold(5).split(ids, groups=ids))
    temps = sorted(d["T"].unique())
    sets = {}
    for sig in ("s", "e", "ds", "d"):
        for g, st in GROUPS.items():
            sets[f"{sig} {g}"] = [f"{sig}_{n}" for n in st]
    for g, st in GROUPS.items():
        sets[f"four-signal {g}"] = [f"{sig}_{n}" for sig in ("s", "e", "ds", "d") for n in st]
    sets["s dispersion+tails"] = sets["s dispersion"] + sets["s tails"]
    sets["s dispersion+tails+sequential"] = sets["s dispersion+tails"] + sets["s sequential"]
    sets["all"] = [c for c in d.columns if c.split("_")[0] in ("s", "e", "ds", "d") and c not in ("s_seq",)]
    out = {}   # (setting, method) -> scores on target
    settings = {"within_blind_spot": None, "cross_temperature": None}
    for setting in settings:
        for m in list(sets) + ["state-transition GJS", "state-transition features"]:
            out[(setting, m)] = np.zeros(len(tgt))
    for tr_i, te_i in folds:
        te_mask = tgt.id.isin(ids[te_i]).values; te = tgt[te_mask]
        train = {"within_blind_spot": tgt[tgt.id.isin(ids[tr_i])], "cross_temperature": d[(d["T"] != TARGET) & d.id.isin(ids[tr_i])]}
        for setting, tr in train.items():
            for m, cols in sets.items():
                out[(setting, m)][te_mask] = lr(tr, te, cols)
            g, f = surpmark_scores(tr, te)
            out[(setting, "state-transition GJS")][te_mask] = g; out[(setting, "state-transition features")][te_mask] = f
    y = tgt.y.values
    rows = []
    for (setting, m), sc in out.items():
        rows.append(dict(evaluation=setting, method=m, AUROC=roc_auc_score(y, sc)))
    R = pd.DataFrame(rows).pivot(index="method", columns="evaluation", values="AUROC").reindex(list(sets) + ["state-transition GJS", "state-transition features"])[list(settings)]
    # worst case over temperatures for a few key sets, leave-one-temperature-out
    key = ["s location", "s dispersion", "s tails", "s sequential", "s dispersion+tails+sequential", "all", "state-transition GJS", "state-transition features"]
    worst = {}
    for m in key:
        vals = []
        for T in temps:
            t2 = d[d["T"] == T].reset_index(drop=True); rest = d[d["T"] != T]
            sc = np.zeros(len(t2))
            for tr_i, te_i in folds:
                te_mask = t2.id.isin(ids[te_i]).values; te = t2[te_mask]; tr = rest[rest.id.isin(ids[tr_i])]
                if m.startswith("state-transition"):
                    g, f = surpmark_scores(tr, te); sc[te_mask] = g if m.endswith("GJS") else f
                else:
                    sc[te_mask] = lr(tr, te, sets[m])
            vals.append(roc_auc_score(t2.y, sc))
        worst[m] = dict(zip([f"T{T}" for T in temps], vals)); worst[m]["min"] = min(vals)
    W = pd.DataFrame(worst).T
    return R, W


def main():
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    for sc in ("gpt2-xl", "Qwen2.5-3B"):
        d = build(f"{DATA}/03_pilot_temperature/scores_{sc}.pkl")
        R, W = evaluate(d, rng)
        print(f"\n== Scorer {sc}: AUROC at T = 1.0 (cross-validation within the blind spot vs trained on the other temperatures) ==")
        print(R.round(3).to_string())
        print(f"\n== Scorer {sc}: leave-one-temperature-out, AUROC per temperature and the minimum over six ==")
        print(W.round(3).to_string())
        R.to_csv(f"{RESULTS}/a09_signal_ablation/ablation_{sc}.csv"); W.to_csv(f"{RESULTS}/a09_signal_ablation/worst_{sc}.csv")


if __name__ == "__main__":
    main()
