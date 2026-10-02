# Appendix A (decoding and alignment): pure vs top-p, base vs instruct, 100 paired texts per cell.
# Decoding analysis (design doc §7.4 / 7.5): does the decoding setting explain the reversed
# direction seen in an earlier pilot? Scores come from notebooks/01_pilot_decoding.ipynb (GPT-2 XL, continuation part only).
# Rows per position: 0 surprisal, 1 entropy, 2 variance of log-prob, 3 top-10 mass.
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1 if 'prepare' not in __file__ else 2]))
from cmargin.paths import DATA, RESULTS
(RESULTS / "a02_decoding_alignment").mkdir(parents=True, exist_ok=True)
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

N_BOOT = 1000
MAGE_REF = {"facebook/opt-2.7b": 0.235}   # same generator in MAGE, GPT-2 XL scoring, pilot 1


def load(path=f"{DATA}/01_pilot_decoding/scores.pkl"):
    s = pd.read_pickle(path)
    a = s.arr
    s["fdg"] = [(x[1].sum() - x[0].sum()) / np.sqrt(x[2].sum()) for x in a]   # higher = AI
    s["surprisal"] = [x[0].mean() for x in a]
    s["entropy"] = [x[1].mean() for x in a]
    s["top10"] = [x[3].mean() for x in a]
    s["n"] = [x.shape[1] for x in a]
    s["y"] = (s.who == "ai").astype(int)
    return s.drop(columns=["arr"])


def auroc_ci(g, col="fdg", rng=None):
    ids = g.id.unique()
    point = roc_auc_score(g.y, g[col])
    boot = []
    for _ in range(N_BOOT):
        take = rng.choice(ids, len(ids))
        d = pd.concat([g[g.id == i] for i in take])
        boot.append(roc_auc_score(d.y, d[col]))
    return point, np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def main():
    rng = np.random.default_rng(0)
    s = load()
    rows = []
    for (model, dec), g in s.groupby(["model", "decoding"]):
        p, lo, hi = auroc_ci(g, "fdg", rng)
        h, ai = g[g.y == 0], g[g.y == 1]
        rows.append(dict(generator=model, decoding=dec, n_texts=len(ai), AUROC=p, CI=f"[{lo:.3f}, {hi:.3f}]",
                         MAGE_reference=MAGE_REF.get(model, np.nan),
                         surprisal_human=h.surprisal.mean(), surprisal_ai=ai.surprisal.mean(),
                         fdg_human=h.fdg.mean(), fdg_ai=ai.fdg.mean(),
                         median_tokens_human=h.n.median(), median_tokens_ai=ai.n.median()))
    R = pd.DataFrame(rows)

    # contrasts: top-p minus pure within a model; instruct minus base within a decoding
    con = []
    for model, g in s.groupby("model"):
        ids = g.id.unique()
        d = {k: v for k, v in g.groupby("decoding")}
        boot = []
        for _ in range(N_BOOT):
            take = rng.choice(ids, len(ids))
            au = {k: roc_auc_score(pd.concat([v[v.id == i] for i in take]).y,
                                   pd.concat([v[v.id == i] for i in take]).fdg) for k, v in d.items()}
            boot.append(au["topp"] - au["pure"])
        lo, md, hi = np.percentile(boot, [2.5, 50, 97.5])
        con.append(dict(comparison=f"{model}: top-p minus pure", diff=md, CI_low=lo, CI_high=hi, CI_excludes_0="yes" if lo > 0 or hi < 0 else "no"))
    base, inst = "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-3B-Instruct"
    for dec in ("pure", "topp"):
        b = s[(s.model == base) & (s.decoding == dec)]
        i = s[(s.model == inst) & (s.decoding == dec)]
        ids = b.id.unique()
        boot = []
        for _ in range(N_BOOT):
            take = rng.choice(ids, len(ids))
            bb, ii = pd.concat([b[b.id == k] for k in take]), pd.concat([i[i.id == k] for k in take])
            boot.append(roc_auc_score(ii.y, ii.fdg) - roc_auc_score(bb.y, bb.fdg))
        lo, md, hi = np.percentile(boot, [2.5, 50, 97.5])
        con.append(dict(comparison=f"Qwen instruct minus base ({dec})", diff=md, CI_low=lo, CI_high=hi, CI_excludes_0="yes" if lo > 0 or hi < 0 else "no"))
    C = pd.DataFrame(con)

    # Post hoc: two-sided score (distance from the human median in either direction); thresholds use human texts only
    two = []
    for (model, dec), g in s.groupby(["model", "decoding"]):
        h = g[g.y == 0].fdg
        med, scl = h.median(), (h.quantile(.75) - h.quantile(.25)) / 1.349
        gg = g.assign(two=lambda x: (x.fdg - med).abs() / scl)
        p, lo, hi = auroc_ci(gg, "two", rng)
        two.append(dict(generator=model, decoding=dec, AUROC_two_sided=p, CI=f"[{lo:.3f}, {hi:.3f}]"))
    T = pd.DataFrame(two)

    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 30)
    print("== AUROC per condition (human vs AI continuations, GPT-2 XL scorer) ==")
    print(R.round(3).to_string(index=False))
    print("\n== Contrasts ==")
    print(C.round(3).to_string(index=False))
    print("\n== Post hoc: two-sided score ==")
    print(T.round(3).to_string(index=False))
    R.to_csv(f"{RESULTS}/a02_decoding_alignment/metrics.csv", index=False)
    C.to_csv(f"{RESULTS}/a02_decoding_alignment/contrasts.csv", index=False)
    T.to_csv(f"{RESULTS}/a02_decoding_alignment/two_sided.csv", index=False)


if __name__ == "__main__":
    main()
