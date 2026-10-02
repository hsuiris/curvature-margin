# Section 5.2: every zero-shot statistic fails where its own population-mean margin vanishes; K is the curvature case.
# Post hoc analysis (2026-09-29), added after an audit showed that K had only been compared with the surprisal margin M.
# Part A, ten Fast-DetectGPT settings: zeros of K, of the mean and of the median Fast-DetectGPT score difference versus the
#   AUROC = 0.5 crossing, on the same data and on 200 random half splits by source document (margins from half A, crossing
#   from half B; the direct baseline takes the AUROC crossing of half A).
# Part B, four Falcon settings, eight statistics: zero of each statistic's own mean score difference versus its own crossing.
# Part C, extrapolation from low temperatures only (T <= 0.90 and T <= 0.94): linear fits of K, of the mean score
#   difference and of probit(AUROC), each extended to its zero.
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1 if 'prepare' not in __file__ else 2]))
from cmargin.paths import DATA, RESULTS
(RESULTS / "e8_margin_generality").mkdir(parents=True, exist_ok=True)
import glob
import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from cmargin.crossing import cross
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from e6_zero_shot_benchmark import feats as stat_feats, METHODS

OUT = RESULTS / "e8_margin_generality"
N_SPLIT = 200


def z(v):
    return np.nan if v is None else v


def auroc(g, h):
    return roc_auc_score(np.r_[np.ones(len(g)), np.zeros(len(h))], np.r_[g, h])


def load_settings():
    """Per-text curvature c (mean of R - s) and Fast-DetectGPT score for the ten (generator, scorer) settings."""
    sets = {}
    for f in sorted(glob.glob(f"{DATA}/06_scaled_sweep/scores_*.pkl")):
        s = pd.read_pickle(f); A = s.arr
        s["c"] = [(a[1] - a[0]).mean() for a in A]
        s["fdg"] = [(a[1].sum() - a[0].sum()) / np.sqrt(a[2].sum()) for a in A]
        sets[f"{s.model.iloc[0].split('/')[-1]} / {'GPT-2 XL' if 'gpt2-xl' in f else 'self'}"] = s.drop(columns=["arr"])
    for f in sorted(glob.glob(f"{DATA}/07_falcon_dual_scoring/scores_falcon_*.pkl")) + sorted(glob.glob(f"{DATA}/09_heldout_pythia/scores_falcon_*.pkl")):
        s = pd.read_pickle(f); A = s.arr
        s["c"] = [(-a[4] - a[0]).mean() for a in A]
        s["fdg"] = [(-a[0].sum() - a[4].sum()) / np.sqrt(a[5].sum()) for a in A]
        sets[f"{s.model.iloc[0].split('/')[-1]} / Falcon"] = s.drop(columns=["arr"])
    for s in sets.values():
        s["T"] = [float(d[1:]) if d != "human" else np.nan for d in s.decoding]
    return sets


def zeros(d):
    h, ai = d[d.who == "human"], d[d.who == "ai"]
    Ts = sorted(ai["T"].unique()); G = [ai[np.isclose(ai["T"], T)] for T in Ts]
    curves = dict(K=[g.c.mean() - h.c.mean() for g in G], mean_score=[g.fdg.mean() - h.fdg.mean() for g in G],
                  median_score=[g.fdg.median() - h.fdg.median() for g in G], AUROC=[auroc(g.fdg.values, h.fdg.values) for g in G])
    return {k: z(cross(Ts, v, 0.5 if k == "AUROC" else 0.0)) for k, v in curves.items()}, Ts, curves


def part_a(sets):
    rows = []
    for name, d in sets.items():
        zz, Ts, cv = zeros(d)
        side = np.sign(np.array(cv["AUROC"]) - 0.5)
        rows.append(dict(setting=name, fail_temp_observed=zz["AUROC"], K_zero=zz["K"], mean_score_zero=zz["mean_score"], median_score_zero=zz["median_score"],
                         K_error=zz["AUROC"] - zz["K"], mean_score_error=zz["AUROC"] - zz["mean_score"], median_score_error=zz["AUROC"] - zz["median_score"],
                         K_sign_disagreements=int((np.sign(cv["K"]) != side).sum()), mean_score_sign_disagreements=int((np.sign(cv["mean_score"]) != side).sum())))
    same = pd.DataFrame(rows); same.to_csv(OUT / "same_data.csv", index=False)
    rng = np.random.default_rng(0); reps = []
    for _ in range(N_SPLIT):
        err = {k: [] for k in ("K", "mean_score", "median_score", "AUROC_half")}
        for d in sets.values():
            ids = d.id.unique(); a = rng.choice(ids, len(ids) // 2, replace=False)
            za, _, _ = zeros(d[d.id.isin(a)]); zb, _, _ = zeros(d[~d.id.isin(a)])
            for k, src in (("K", "K"), ("mean_score", "mean_score"), ("median_score", "median_score"), ("AUROC_half", "AUROC")):
                err[k].append(abs(zb["AUROC"] - za[src]))
        reps.append({k: np.nanmean(v) for k, v in err.items()})
    reps = pd.DataFrame(reps); reps.to_csv(OUT / "half_split.csv", index=False)
    summ = reps.describe().loc[["mean", "25%", "50%", "75%"]].T
    summ["share_K_better"] = [np.nan if k == "K" else float(np.mean(reps.K < reps[k])) for k in summ.index]
    summ.rename_axis("predictor").to_csv(OUT / "half_split_summary.csv")
    return same, summ


def part_b():
    rows = []
    for f in sorted(glob.glob(f"{DATA}/07_falcon_dual_scoring/scores_falcon_*.pkl")) + sorted(glob.glob(f"{DATA}/09_heldout_pythia/scores_falcon_*.pkl")):
        raw = pd.read_pickle(f); s = stat_feats(raw); s["c"] = [(-a[4] - a[0]).mean() for a in raw.arr]
        h, ai = s[s.who == "human"], s[s.who == "ai"]
        Ts = sorted(ai["T"].unique()); G = [ai[np.isclose(ai["T"], T)] for T in Ts]
        k0 = z(cross(Ts, [g.c.mean() - h.c.mean() for g in G]))
        for m in METHODS:
            obs = z(cross(Ts, [auroc(g[m].values, h[m].values) for g in G], 0.5))
            own = z(cross(Ts, [g[m].mean() - h[m].mean() for g in G]))
            rows.append(dict(generator=s.model.iloc[0], statistic=m, fail_temp_observed=obs, own_margin_zero=own, own_margin_error=obs - own,
                             K_zero=k0, K_error=obs - k0))
    own = pd.DataFrame(rows); own.to_csv(OUT / "own_margin.csv", index=False)
    return own


def part_c(sets):
    rows = []
    for name, d in sets.items():
        _, Ts, cv = zeros(d); Ts = np.array(Ts)
        for cut in (0.90, 0.94):
            m = Ts <= cut + 1e-9
            if m.sum() < 2: continue
            line_zero = lambda y: (lambda b: -b[1] / b[0])(np.polyfit(Ts[m], np.asarray(y)[m], 1))
            obs = cross(Ts, cv["AUROC"], 0.5)
            rows.append(dict(setting=name, cutoff=cut, n_temperatures=int(m.sum()), fail_temp_observed=obs,
                             K_error=obs - line_zero(cv["K"]), mean_score_error=obs - line_zero(cv["mean_score"]),
                             AUROC_probit_error=obs - line_zero(norm.ppf(np.clip(cv["AUROC"], 1e-4, 1 - 1e-4)))))
    ext = pd.DataFrame(rows); ext.to_csv(OUT / "extrapolation.csv", index=False)
    return ext


def main():
    pd.set_option("display.width", 220)
    sets = load_settings()
    same, summ = part_a(sets)
    print("== A. Same data: zero of each margin vs the Fast-DetectGPT AUROC crossing ==")
    print(same.round(4).to_string(index=False))
    print("MAE:", {k: round(same[f"{k}_error"].abs().mean(), 4) for k in ("K", "mean_score", "median_score")})
    print(f"\n== A. {N_SPLIT} half splits by source document: MAE over the ten settings ==")
    print(summ.round(4).to_string())
    own = part_b()
    print("\n== B. Each statistic's own mean margin vs its own crossing (Falcon scorer) ==")
    print(own.round(4).to_string(index=False))
    ok = own.dropna(subset=["own_margin_error"])
    print(f"own-margin MAE {ok.own_margin_error.abs().mean():.4f} over {len(ok)} (generator, statistic) pairs; K's zero, MAE by statistic:")
    print(own.dropna(subset=["K_error"]).groupby("statistic").K_error.apply(lambda e: e.abs().mean()).round(4).to_string())
    ext = part_c(sets)
    print("\n== C. Extrapolation from low temperatures only ==")
    print(ext.round(3).to_string(index=False))
    print(ext.groupby("cutoff")[["K_error", "mean_score_error", "AUROC_probit_error"]].apply(lambda g: g.abs().mean()).round(3).to_string())


if __name__ == "__main__":
    main()
