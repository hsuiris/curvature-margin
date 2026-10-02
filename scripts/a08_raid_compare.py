# Section 5.5 / RAID: two scorers side by side, margin sign agreement, two-sided scores.
# Pilot 7c follow-up (post hoc): RAID cells under two scorers side by side.
# (1) same texts, GPT-2 XL vs Qwen2.5-3B: AUROC and margin; (2) two-sided score calibrated on the
# domain's humans (in-sample at pilot scale); (3) cells where the sign of the margin disagrees with
# which side of 0.5 the AUROC falls; (4) forward AUROC prediction with the probit scale from pilots 3/4.
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1 if 'prepare' not in __file__ else 2]))
from cmargin.paths import DATA, RESULTS
(RESULTS / "a08_raid_compare").mkdir(parents=True, exist_ok=True)
import numpy as np, pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.metrics import roc_auc_score

SIGMA = 0.638   # fitted on the six GPT-2 XL pilot conditions of 01/02 (a04_temperature_sweep.py)


def load(scorer):
    d = pd.read_pickle(f"{DATA}/raid_scores/scores_{scorer}.pkl")
    d["surprisal"] = [a[0].mean() for a in d.arr]
    d["fdg"] = [(a[1].sum() - a[0].sum()) / np.sqrt(a[2].sum()) for a in d.arr]
    return d.drop(columns=["arr"])


def matched(g, hum, col):
    """Domain-matched AUROC weighted by #AI; also returns the mean margin."""
    num = den = 0.0
    for dom, gg in g.groupby("domain"):
        h = hum[hum.domain == dom]
        y = np.r_[np.ones(len(gg)), np.zeros(len(h))]
        num += len(gg) * roc_auc_score(y, np.r_[gg[col].values, h[col].values]); den += len(gg)
    return num / den


def main():
    rows = []
    for scorer in ("gpt2-xl", "Qwen2.5-3B"):
        d = load(scorer)
        hum = d[d.model == "human"].copy()
        # two-sided score, calibrated per domain on that domain's humans
        cal = {dom: (h.fdg.median(), (h.fdg.quantile(.75) - h.fdg.quantile(.25)) / 1.349) for dom, h in hum.groupby("domain")}
        d["two"] = [abs(f - cal[dom][0]) / cal[dom][1] for f, dom in zip(d.fdg, d.domain)]
        hum = d[d.model == "human"]
        for (m, dec, rp), g in d[d.model != "human"].groupby(["model", "decoding", "repetition_penalty"]):
            margin = np.mean([hum[hum.domain == dom].surprisal.mean() - gg.surprisal.mean() for dom, gg in g.groupby("domain")])
            rows.append(dict(generator=m, decoding=dec, repetition_penalty=rp, scorer=scorer, surprisal_margin=margin,
                             AUROC_one_sided=matched(g, hum, "fdg"), AUROC_two_sided=matched(g, hum, "two")))
    R = pd.DataFrame(rows)
    R["AUROC_predicted"] = norm.cdf(R.surprisal_margin / SIGMA)
    R["sign_agrees"] = np.where((R.surprisal_margin > 0) == (R.AUROC_one_sided > 0.5), "yes", "no")
    W = R.pivot_table(index=["generator", "decoding", "repetition_penalty"], columns="scorer", values=["AUROC_one_sided", "AUROC_two_sided", "surprisal_margin"])
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 100)
    print("== Same texts, two scorers side by side ==")
    print(W.round(3).to_string())
    rho = spearmanr(R.surprisal_margin, R.AUROC_one_sided)
    print(f"\nSpearman correlation over 68 (cell x scorer) pairs, margin vs one-sided AUROC: {rho.statistic:.3f} (p={rho.pvalue:.2e})")
    print(f"Cells where the sign of the margin disagrees with the side of 0.5: {(R.sign_agrees == 'no').sum()} / {len(R)}")
    print(R[R.sign_agrees == "no"][["generator", "decoding", "repetition_penalty", "scorer", "surprisal_margin", "AUROC_one_sided"]].round(3).to_string(index=False))
    inv = R[R.AUROC_one_sided < 0.5]
    print(f"\n{len(inv)} inverted cells (one-sided < 0.5), after the two-sided score:")
    print(inv[["generator", "decoding", "repetition_penalty", "scorer", "AUROC_one_sided", "AUROC_two_sided"]].round(3).to_string(index=False))
    print(f"\nMean cost of two-sided vs one-sided on cells with one-sided > 0.9: {(R[R.AUROC_one_sided > 0.9].AUROC_two_sided - R[R.AUROC_one_sided > 0.9].AUROC_one_sided).mean():.3f}")
    g = R[R.scorer == "gpt2-xl"]
    print(f"Forward prediction (sigma={SIGMA}, the 34 GPT-2 XL cells only): mean absolute error {(g.AUROC_predicted - g.AUROC_one_sided).abs().mean():.3f}, max {(g.AUROC_predicted - g.AUROC_one_sided).abs().max():.3f}")
    R.to_csv(f"{RESULTS}/a08_raid_compare/compare.csv", index=False)


if __name__ == "__main__":
    main()
