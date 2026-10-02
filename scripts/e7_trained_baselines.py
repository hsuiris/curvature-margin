# Section 5.4 / Table 6 / Figure 4: MAGE Longformer, RoBERTa, DivEye, AdaDetectGPT.
# Trained baselines (design doc §8.8.16): per-text scores from notebooks/12_trained_baselines_run.ipynb (11_trained_baselines/scores_trained.csv).
# Columns: set (main1 = 06_scaled_sweep, extA = 09_heldout_pythia, m4gt = 08_m4gt_natural_text), id, domain, model, decoding, who + detector columns. Leave-one-generator-out variants are stored
# per held-out generator (diveye_logo_<gen>, ada_logo_<gen>); humans carry all of them. Protocol as e3_falcon_dual.py: AUROC +
# 1,000-bootstrap CI per (set, generator, temperature); 1% threshold from even-id humans, actual FPR on odd-id humans; crossing by
# linear interpolation + bootstrap CI; per-domain AUROC; M4GT overall / per-domain / per-generator.
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1 if 'prepare' not in __file__ else 2]))
from cmargin.paths import DATA, RESULTS
(RESULTS / "e7_trained_baselines").mkdir(parents=True, exist_ok=True)
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from cmargin.crossing import cross

N_BOOT, FPR = 1000, 0.01
DET = ["mage_longformer", "roberta_openai", "diveye_all3", "diveye_logo", "ada_identity", "ada_all3", "ada_logo"]


def col(det, m):
    return f"{det}_{m}" if det.endswith("_logo") else det


def auroc(g, h):
    return roc_auc_score(np.r_[np.ones(len(g)), np.zeros(len(h))], np.r_[g, h])


def main():
    rng = np.random.default_rng(0)
    d = pd.read_csv(f"{DATA}/11_trained_baselines/scores_trained.csv")
    d["T"] = [float(x[1:]) if str(x).startswith("T") else np.nan for x in d.decoding]
    pd.set_option("display.width", 300); pd.set_option("display.max_rows", 500); pd.set_option("display.max_columns", 40)
    M, D, X, R = [], [], [], []
    for setname in ("main1", "extA"):
        s = d[d.set == setname]; h_all = s[s.who == "human"]
        for m, ai in s[s.who == "ai"].groupby("model"):
            Ts = sorted(ai["T"].unique()); G = {T: ai[np.isclose(ai["T"], T)] for T in Ts}
            for det in DET:
                c = col(det, m)
                if c not in d.columns or ai[c].isna().all(): continue
                h = h_all[h_all[c].notna()]; hv = h[c].values
                h_cal, h_test = h[h.id % 2 == 0][c].values, h[h.id % 2 == 1][c].values
                thr = np.quantile(h_cal, 1 - FPR); curve = []
                for T in Ts:
                    g = G[T][c].dropna().values
                    if len(g) == 0: continue
                    au = auroc(g, hv); curve.append((T, au))
                    b = [auroc(rng.choice(g, len(g)), rng.choice(hv, len(hv))) for _ in range(N_BOOT)]
                    tb = [(rng.choice(g, len(g)) > thr).mean() for _ in range(N_BOOT)]
                    M.append(dict(dataset=setname, generator=m, detector=det, temperature=T, n_ai=len(g), AUROC=au, CI_low=np.percentile(b, 2.5), CI_high=np.percentile(b, 97.5),
                                  TPR_1pct=(g > thr).mean(), TPR_CI_low=np.percentile(tb, 2.5), TPR_CI_high=np.percentile(tb, 97.5), FPR_realized=(h_test > thr).mean()))
                    for dom, gg in G[T].groupby("domain"):
                        D.append(dict(dataset=setname, generator=m, detector=det, domain=dom, temperature=T, AUROC=auroc(gg[c].dropna().values, h[h.domain == dom][c].values)))
                if len(curve) < 2: continue
                Ts2, au2 = [x[0] for x in curve], [x[1] for x in curve]; ac = cross(Ts2, au2, 0.5); boots = []
                for _ in range(N_BOOT):
                    hb = rng.choice(hv, len(hv)); cc = cross(Ts2, [auroc(rng.choice(G[T][c].dropna().values, len(G[T])), hb) for T in Ts2], 0.5)
                    if cc is not None: boots.append(cc)
                X.append(dict(dataset=setname, generator=m, detector=det, fail_temp=ac, CI_low=np.percentile(boots, 2.5) if boots else np.nan, CI_high=np.percentile(boots, 97.5) if boots else np.nan,
                              share_bootstrap_crossing=len(boots) / N_BOOT, monotone_decreasing="yes" if (np.diff(au2) < 0).all() else "no", AUROC_lowest_temp=au2[0], AUROC_at_T1_0=dict(curve).get(1.0, np.nan)))
    s = d[d.set == "m4gt"]
    for det in DET:
        if det.endswith("_logo") or det not in d.columns or s[det].isna().all(): continue
        for scope, frame in [("overall", s)] + [(dom, g) for dom, g in s.groupby("domain")]:
            hid = frame[frame.who == "human"]; h = hid[det].dropna().values; g = frame[frame.who == "ai"][det].dropna().values
            thr = np.quantile(hid[hid.id % 2 == 0][det].dropna().values, 1 - FPR, method="higher")
            b = [auroc(rng.choice(g, len(g)), rng.choice(h, len(h))) for _ in range(N_BOOT)]
            R.append(dict(detector=det, scope=scope, n_human=len(h), n_ai=len(g), AUROC=auroc(g, h), CI_low=np.percentile(b, 2.5), CI_high=np.percentile(b, 97.5),
                          TPR_1pct_same_data=(g >= np.quantile(h, 1 - FPR, method="higher")).mean(), TPR_split=(g >= thr).mean(), FPR_realized_split=(hid[hid.id % 2 == 1][det].dropna().values >= thr).mean()))
        for (dom, m), g in s[s.who == "ai"].groupby(["domain", "model"]):
            h = s[(s.who == "human") & (s.domain == dom)][det].dropna().values
            R.append(dict(detector=det, scope=f"{dom}:{m}", n_human=len(h), n_ai=len(g), AUROC=auroc(g[det].dropna().values, h)))
    M, D, X, R = pd.DataFrame(M), pd.DataFrame(D), pd.DataFrame(X), pd.DataFrame(R)
    print("== Failure temperature (AUROC = 0.5), monotonicity, AUROC at T = 1.0 ==")
    print(X.round(3).to_string(index=False))
    print("\n== AUROC and TPR at 1% FPR, T = 0.8 ==")
    print(M[np.isclose(M.temperature, 0.8)].pivot_table(index=["dataset", "generator"], columns="detector", values=["AUROC", "TPR_1pct"]).round(3).to_string())
    print("\n== AUROC at T = 1.0 ==")
    print(M[np.isclose(M.temperature, 1.0)].pivot_table(index=["dataset", "generator"], columns="detector", values="AUROC").round(3).to_string())
    if len(D):
        rows = []
        for (a, m, det, dom), g in D.groupby(["dataset", "generator", "detector", "domain"]):
            g = g.sort_values("temperature"); rows.append(dict(dataset=a, generator=m, detector=det, domain=dom, fail_temp=cross(g.temperature, g.AUROC, 0.5)))
        DX = pd.DataFrame(rows); print("\n== Largest cross-domain spread of the failure temperature ==")
        print(DX.groupby(["generator", "detector"]).fail_temp.agg(lambda x: x.max() - x.min() if x.notna().all() else np.nan).unstack().round(3).to_string())
        DX.to_csv(f"{RESULTS}/e7_trained_baselines/domain_crossing.csv", index=False)
    print("\n== M4GT B (overall and per domain) ==")
    print(R[~R.scope.str.contains(":")].round(3).to_string(index=False))
    M.to_csv(f"{RESULTS}/e7_trained_baselines/metrics.csv", index=False); D.to_csv(f"{RESULTS}/e7_trained_baselines/domains.csv", index=False)
    X.to_csv(f"{RESULTS}/e7_trained_baselines/crossing.csv", index=False); R.to_csv(f"{RESULTS}/e7_trained_baselines/m4gt.csv", index=False)


if __name__ == "__main__":
    main()
