# Section 5.5 / Figure M4GT: M4GT-Bench natural text, per source and generator, margin rule.
# External validation B (external_validation_protocol.md): M4GT-Bench Subtask B English test split, Falcon-7B-Instruct scorer +
# Falcon-7B observer, scored in notebooks/08_m4gt_natural_text.ipynb. Pre-specified: overall and per-domain AUROC, same-data
# TPR at 1% FPR, hash-split actual FPR (all recomputed here from the per-text scores and checked against the notebook's
# summary), plus 1,000-sample bootstrap CIs. Descriptive: per (domain, generator) cells with the curvature margin (the
# protocol's fixed predictor) and the surprisal margin, sign agreement with the AUROC side, Spearman across cells.
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1 if 'prepare' not in __file__ else 2]))
from cmargin.paths import DATA, RESULTS
(RESULTS / "e4_m4gt_natural").mkdir(parents=True, exist_ok=True)
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

N_BOOT, FPR = 1000, 0.01
DET = {"fast_detect_gpt": "FDG", "binoculars": "Bino"}


def auroc(g, h):
    return roc_auc_score(np.r_[np.ones(len(g)), np.zeros(len(h))], np.r_[g, h])


def block(frame, scope, rng):
    r = dict(scope=scope, n_human=(frame.who == "human").sum(), n_ai=(frame.who == "ai").sum())
    hum = frame[frame.who == "human"]; ai = frame[frame.who == "ai"]
    even = hum.text_sha256.str.slice(0, 8).map(lambda x: int(x, 16) % 2 == 0)
    for col, name in DET.items():
        h, g = hum[col].values, ai[col].values
        b = [auroc(rng.choice(g, len(g)), rng.choice(h, len(h))) for _ in range(N_BOOT)]
        thr = np.quantile(h, 1 - FPR, method="higher")
        tb = [(rng.choice(g, len(g)) >= thr).mean() for _ in range(N_BOOT)]
        thr_s = np.quantile(hum[even][col].values, 1 - FPR, method="higher")
        r.update({f"AUROC_{name}": auroc(g, h), f"{name}_CI_low": np.percentile(b, 2.5), f"{name}_CI_high": np.percentile(b, 97.5),
                  f"TPR_1pct_same_data_{name}": (g >= thr).mean(), f"{name}_TPR_CI_low": np.percentile(tb, 2.5), f"{name}_TPR_CI_high": np.percentile(tb, 97.5),
                  f"FPR_realized_split_{name}": (hum[~even][col].values >= thr_s).mean(), f"TPR_split_{name}": (g >= thr_s).mean()})
    r["n_human_calibration"] = int(even.sum()); r["n_human_evaluation"] = int((~even).sum())
    return r


def main():
    rng = np.random.default_rng(0)
    d = pd.read_csv(f"{DATA}/08_m4gt_natural_text/m4gt_falcon_scores.csv")
    nb = pd.read_csv(f"{DATA}/08_m4gt_natural_text/m4gt_falcon_summary.csv")
    pd.set_option("display.width", 300); pd.set_option("display.max_rows", 200); pd.set_option("display.max_columns", 40)
    S = pd.DataFrame([block(d, "overall", rng)] + [block(g, dom, rng) for dom, g in d.groupby("source")])
    # check against the notebook's own summary
    chk = nb[nb.detector == "fast_detect_gpt"].set_index(nb[nb.detector == "fast_detect_gpt"].scope.str.replace("domain:", ""))
    assert np.allclose(S.set_index("scope").loc[chk.index, "AUROC_FDG"], chk.auroc) and np.allclose(S.set_index("scope").loc[chk.index, "TPR_1pct_same_data_FDG"], chk.same_data_tpr_at_1pct_fpr)
    print("== Overall and per domain (humans vs all AI texts of the domain; CIs from 1,000 bootstrap samples; split = 1% threshold set on even-hash humans, realized FPR on the rest) ==")
    print(S.round(3).to_string(index=False))

    # cells: (domain, generator) vs that domain's humans; margins from the per-text means
    rows = []
    for (dom, m), g in d[d.who == "ai"].groupby(["source", "model"]):
        h = d[(d.who == "human") & (d.source == dom)]
        rows.append(dict(domain=dom, generator=m, n_ai=len(g), AUROC_FDG=auroc(g.fast_detect_gpt.values, h.fast_detect_gpt.values),
                         AUROC_Bino=auroc(g.binoculars.values, h.binoculars.values),
                         curvature_margin=g.curvature.mean() - h.curvature.mean(), surprisal_margin=h.surprisal.mean() - g.surprisal.mean(),
                         mean_tokens_ai=g.token_count.mean(), mean_tokens_human=h.token_count.mean()))
    C = pd.DataFrame(rows)
    C["sign_agrees_curvature"] = np.where((C.curvature_margin > 0) == (C.AUROC_FDG > 0.5), "yes", "no")
    C["sign_agrees_surprisal"] = np.where((C.surprisal_margin > 0) == (C.AUROC_FDG > 0.5), "yes", "no")
    print("\n== Per (domain x generator): domain-matched AUROC and the two margins ==")
    print(C.sort_values(["domain", "AUROC_FDG"]).round(3).to_string(index=False))
    rc, rs = spearmanr(C.curvature_margin, C.AUROC_FDG), spearmanr(C.surprisal_margin, C.AUROC_FDG)
    print(f"\n{len(C)} cells: Spearman, curvature margin vs AUROC {rc.statistic:.3f} (p={rc.pvalue:.1e}); surprisal margin {rs.statistic:.3f} (p={rs.pvalue:.1e})")
    print(f"Sign of the curvature margin disagrees with the side of 0.5: {(C.sign_agrees_curvature == 'no').sum()} / {len(C)}; surprisal margin: {(C.sign_agrees_surprisal == 'no').sum()} / {len(C)}")
    print(C[(C.sign_agrees_curvature == "no") | (C.AUROC_FDG < 0.6)][["domain", "generator", "AUROC_FDG", "curvature_margin", "surprisal_margin"]].round(3).to_string(index=False))
    # per generator pooled across its domains (domain-matched, weighted by #AI, as in a08_raid_compare.py)
    Mrows = []
    for m, x in C.groupby("generator"):
        w = x.n_ai.values
        Mrows.append(dict(generator=m, n_domains=len(x), n_ai=int(w.sum()), AUROC_domain_matched_FDG=np.average(x.AUROC_FDG, weights=w),
                          AUROC_domain_matched_Bino=np.average(x.AUROC_Bino, weights=w), curvature_margin=np.average(x.curvature_margin, weights=w),
                          AUROC_lowest_domain=x.AUROC_FDG.min(), lowest_domain=x.domain.iloc[int(x.AUROC_FDG.values.argmin())]))
    M = pd.DataFrame(Mrows)
    print("\n== Per generator (across domains; domain-matched AUROC weighted by the number of AI texts) ==")
    print(M.sort_values("AUROC_domain_matched_FDG").round(3).to_string(index=False))
    S.to_csv(f"{RESULTS}/e4_m4gt_natural/summary.csv", index=False); C.to_csv(f"{RESULTS}/e4_m4gt_natural/cells.csv", index=False); M.to_csv(f"{RESULTS}/e4_m4gt_natural/models.csv", index=False)


if __name__ == "__main__":
    main()
