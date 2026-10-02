# Appendix (evaluation exceptions): failure-temperature intervals of the first study, recomputed with the pairing kept.
# Post hoc sensitivity analysis (2026-10-02, not pre-registered). e6 and e9 resampled the humans and each temperature's machine
# texts separately. Here one bootstrap round draws documents, as in the replication (e12_replication_test.py, round_weights):
# counts are drawn within domain x role strata (role = prompt document or reference-only human); a drawn document brings its
# human text and, if it is a prompt, its continuations at every temperature. 2,000 rounds, percentile interval over the rounds
# that cross. Point estimates, statistics and the interpolation (cmargin.crossing.cross) are those of e6 / e9, unchanged.
# The unstratified columns draw the 1,000 documents without strata, to separate the effect of pairing from that of the strata.
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1])); sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cmargin.paths import DATA, RESULTS
import glob
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from cmargin.crossing import cross
from e6_zero_shot_benchmark import feats, METHODS, SETS
from e9_theory_prediction import per_text

OUT = RESULTS / "e13_paired_crossing_ci"; OUT.mkdir(parents=True, exist_ok=True)
N_BOOT, SEED = 2000, 20261002


def draw(rng, strata, n):
    w = np.zeros((N_BOOT, n))
    for s in strata:
        pick = s[rng.integers(0, len(s), (N_BOOT, len(s)))]
        np.add.at(w, (np.arange(N_BOOT)[:, None], pick), 1)
    return w


def paired_ci(Ts, G, hv, W, prompt_pos):
    """G: temperatures x prompts, columns aligned with prompt_pos; W: rounds x documents. Returns low, high, crossing share."""
    o = np.argsort(hv, kind="stable"); hs = hv[o]
    lo, hi = np.searchsorted(hs, G, "left"), np.searchsorted(hs, G, "right")
    auc = lambda wh, wg: (wg * (np.r_[0.0, np.cumsum(wh[o])][lo] * 0.5 + np.r_[0.0, np.cumsum(wh[o])][hi] * 0.5)).sum(-1) / (wg.sum() * wh.sum())
    one = np.ones(len(hv))
    assert np.allclose(auc(one, one[prompt_pos]), [roc_auc_score(np.r_[np.ones(len(g)), np.zeros(len(hv))], np.r_[g, hv]) for g in G])
    boots = [c for c in (cross(Ts, auc(w, w[prompt_pos]), 0.5) for w in W) if c is not None]
    return (np.percentile(boots, 2.5), np.percentile(boots, 97.5)) if boots else (np.nan, np.nan), len(boots) / N_BOOT


def evaluate(d, methods, rng):
    h = d[d.who == "human"].reset_index(drop=True); ai = d[d.who == "ai"]
    assert h.id.is_unique
    pos = pd.Series(h.index, index=h.id); Ts = sorted(ai["T"].unique())
    by_T = [ai[np.isclose(ai["T"], T)].set_index("id") for T in Ts]
    ids = by_T[0].index
    assert all(set(g.index) == set(ids) and g.index.is_unique for g in by_T), "every temperature must share the same prompts"
    assert (by_T[0].domain.values == h.domain.values[pos[ids].values]).all()
    prompt_pos = pos[ids].values; role = np.zeros(len(h), bool); role[prompt_pos] = True
    strata = [np.flatnonzero((h.domain.values == dom) & (role == r)) for dom in sorted(h.domain.unique()) for r in (True, False)]
    W = {"paired": draw(rng, [s for s in strata if len(s)], len(h)), "paired_unstratified": draw(rng, [np.arange(len(h))], len(h))}
    for meth in methods:
        G = np.array([g.loc[ids, meth].values for g in by_T]); hv = h[meth].values
        r = dict(method=meth, fail_temp=cross(Ts, [roc_auc_score(np.r_[np.ones(len(g)), np.zeros(len(hv))], np.r_[g, hv]) for g in G], 0.5),
                 n_prompts=len(ids), n_humans=len(h))
        for k, w in W.items():
            (r[f"{k}_CI_low"], r[f"{k}_CI_high"]), r[f"{k}_crossing_share"] = paired_ci(Ts, G, hv, w, prompt_pos)
            r[f"{k}_CI_half_width"] = (r[f"{k}_CI_high"] - r[f"{k}_CI_low"]) / 2
        yield r


def main():
    rng = np.random.default_rng(SEED); pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    rows = []
    for study, pat in (("development", "06_scaled_sweep"), ("first_test", "13_theory_prediction_test")):   # e9, self-scoring
        for f in sorted(glob.glob(f"{DATA}/{pat}/scores_self_*.pkl")):
            rows += [dict(source="e9", dataset=study, generator=f.split("scores_self_")[1][:-4], **r) for r in evaluate(per_text(pd.read_pickle(f)), ["fdg"], rng)]
            print("done", f, flush=True)
    for setname, pat in SETS.items():                                                                      # e6, Falcon scorer
        for f in sorted(glob.glob(pat)):
            d = feats(pd.read_pickle(f))
            rows += [dict(source="e6", dataset=setname, generator=d.model.iloc[0], **r) for r in evaluate(d, METHODS, rng)]
            print("done", f, flush=True)
    R = pd.DataFrame(rows)
    old9 = pd.concat([pd.read_csv(RESULTS / "e9_theory_prediction" / f"{n}.csv") for n in ("development", "test")]).assign(source="e9", method="fdg")
    old9 = old9.rename(columns=dict(fail_temp_observed="fail_temp", fail_temp_CI_low="CI_low", fail_temp_CI_high="CI_high"))
    old = pd.concat([old9, pd.read_csv(RESULTS / "e6_zero_shot_benchmark" / "crossing.csv").assign(source="e6")])[["source", "generator", "method", "fail_temp", "CI_low", "CI_high"]]
    R = R.merge(old.rename(columns=dict(fail_temp="fail_temp_stored", CI_low="separate_CI_low", CI_high="separate_CI_high")), on=["source", "generator", "method"], validate="1:1")
    ok = R.fail_temp.notna()
    assert np.allclose(R.fail_temp[ok], R.fail_temp_stored[ok], atol=1e-9) and R.fail_temp_stored[~ok].isna().all(), "point estimates must equal the stored e6 / e9 results"
    R["separate_CI_half_width"] = (R.separate_CI_high - R.separate_CI_low) / 2
    R["half_width_ratio"] = R.paired_CI_half_width / R.separate_CI_half_width
    R.drop(columns=["fail_temp_stored"]).to_csv(OUT / "crossing_ci.csv", index=False)
    show = ["source", "generator", "method", "fail_temp", "separate_CI_low", "separate_CI_high", "paired_CI_low", "paired_CI_high",
            "separate_CI_half_width", "paired_CI_half_width", "paired_unstratified_CI_half_width", "half_width_ratio", "paired_crossing_share"]
    print("\n== Failure-temperature 95% intervals: separate resampling (stored) vs. document-level, pairing kept ==")
    print(R[show].round(4).to_string(index=False))
    S = R[R.method != "DMAP_chi2"]
    print("\n== Half-width by source (DMAP_chi2 excluded: no stable crossing) ==")
    print(S.groupby("source")[["separate_CI_half_width", "paired_CI_half_width", "paired_unstratified_CI_half_width", "half_width_ratio"]].agg(["mean", "min", "max"]).round(4).to_string())
    fd = S[S.method.isin(["fdg", "FastDetectGPT"])]
    print("\nFast-DetectGPT largest half-width:", fd.groupby("source")[["separate_CI_half_width", "paired_CI_half_width"]].max().round(4).to_dict("index"))
    print("largest half-width overall:", S.loc[S.paired_CI_half_width.idxmax(), ["generator", "method", "separate_CI_half_width", "paired_CI_half_width"]].to_dict())
    shift = np.maximum((S.paired_CI_low - S.separate_CI_low).abs(), (S.paired_CI_high - S.separate_CI_high).abs())
    print("largest endpoint shift:", shift.groupby(S.source).max().round(4).to_dict())


if __name__ == "__main__":
    main()
