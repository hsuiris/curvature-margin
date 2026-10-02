# Section 5.3: predict the self-scoring failure temperature from human text only (design doc §8.8.18, pre-registered 2026-09-29).
# First-order theory: sampling from p at temperature T gives E[-log p] = H + (T - 1) Var_p[log p] per position, so machine text
# scored by its own generator has E_g[mean(H - s)] ~ -(T - 1) V and the curvature margin vanishes at T* = 1 - E_h[c] / V,
# with c = mean(H - s) and V = mean Var_p[log p] measured on human continuations only. No parameter is fitted.
# Development (06_scaled_sweep, self scores; exploratory, outcome known): prediction vs the observed AUROC = 0.5 crossing.
# Test (13_theory_prediction_test, four new base models): the stored predictions.json is recomputed from the human scores and
# checked; criteria: each |error| <= 0.02 and MAE <= 0.01; advantage only if the MAE is below both baselines (T* = 1 and the
# development mean 1.011); a crossing outside the grid is undetermined. Crossing CIs: 1,000 bootstrap resamples of humans and
# of each temperature's machine texts.
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1 if 'prepare' not in __file__ else 2]))
from cmargin.paths import DATA, RESULTS
(RESULTS / "e9_theory_prediction").mkdir(parents=True, exist_ok=True)
import glob, json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from cmargin.crossing import cross

OUT = RESULTS / "e9_theory_prediction"
DEV_MEAN, TOL_EACH, TOL_MAE, N_BOOT = 1.011, 0.02, 0.01, 1000


def per_text(s):
    s = s.copy(); A = s.arr
    s["c"] = [(a[1] - a[0]).mean() for a in A]; s["V"] = [a[2].mean() for a in A]
    s["fdg"] = [(a[1] - a[0]).sum() / np.sqrt(a[2].sum()) for a in A]
    s["T"] = [float(d[1:]) if d != "human" else np.nan for d in s.decoding]
    return s.drop(columns=["arr"])


def auroc(g, h):
    return roc_auc_score(np.r_[np.ones(len(g)), np.zeros(len(h))], np.r_[g, h])


def evaluate(s, rng):
    h, ai = s[s.who == "human"], s[s.who == "ai"]
    Ts = sorted(ai["T"].unique()); G = [ai[np.isclose(ai["T"], T)].fdg.values for T in Ts]
    pred = 1 - h.c.mean() / h.V.mean()
    obs = cross(Ts, [auroc(g, h.fdg.values) for g in G], 0.5)
    boots = []
    for _ in range(N_BOOT):
        hb = rng.choice(h.fdg.values, len(h))
        c = cross(Ts, [auroc(rng.choice(g, len(g)), hb) for g in G], 0.5)
        if c is not None: boots.append(c)
    lo, hi = (np.percentile(boots, 2.5), np.percentile(boots, 97.5)) if boots else (np.nan, np.nan)
    return dict(E_h_c=h.c.mean(), V_human=h.V.mean(), fail_temp_predicted=pred, fail_temp_observed=np.nan if obs is None else obs, fail_temp_CI_low=lo, fail_temp_CI_high=hi,
                crossing_share=len(boots) / N_BOOT, n_humans=len(h), temperatures=len(Ts))


def with_errors(R):
    R["fail_temp_error"] = R.fail_temp_observed - R.fail_temp_predicted; R["error_guess_T1"] = R.fail_temp_observed - 1.0
    R["error_dev_mean"] = R.fail_temp_observed - DEV_MEAN
    return R


def main():
    rng = np.random.default_rng(0); pd.set_option("display.width", 220)
    dev = with_errors(pd.DataFrame([dict(generator=f.split("scores_self_")[1][:-4], **evaluate(per_text(pd.read_pickle(f)), rng))
                                    for f in sorted(glob.glob(f"{DATA}/06_scaled_sweep/scores_self_*.pkl"))]))
    dev.to_csv(OUT / "development.csv", index=False)
    print("== Development (exploratory; the observed crossings were known) ==")
    print(dev.round(4).to_string(index=False))
    print(f"MAE: theory {dev.fail_temp_error.abs().mean():.4f} | guess T* = 1 {dev.error_guess_T1.abs().mean():.4f}")
    test_dir = pathlib.Path(DATA) / "13_theory_prediction_test"
    if not (test_dir / "predictions.json").exists():
        print("\nTest data not found yet (13_theory_prediction_test/predictions.json)."); return
    stored = json.load(open(test_dir / "predictions.json")); rows = []
    for tag, rec in stored.items():
        f = test_dir / f"scores_self_{tag}.pkl"
        if not f.exists():
            print("missing", f); continue
        s = per_text(pd.read_pickle(f)); h = per_text(pd.read_pickle(test_dir / f"humans_self_{tag}.pkl"))
        assert abs((1 - h.c.mean() / h.V.mean()) - rec["T_pred"]) < 1e-9, f"{tag}: stored prediction differs from the human scores"
        rows.append(dict(generator=tag, prediction_written_at_utc=rec["written_at_utc"], **evaluate(s, rng)))
    test = with_errors(pd.DataFrame(rows)); test.to_csv(OUT / "test.csv", index=False)
    print("\n== Pre-registered test (four new base models, self-scoring) ==")
    print(test.drop(columns=["prediction_written_at_utc"]).round(4).to_string(index=False))
    ok = test.dropna(subset=["fail_temp_error"])
    if len(ok) < len(test): print("undetermined (crossing outside the grid):", test[test.fail_temp_error.isna()].generator.tolist())
    mae, mae1, maed = ok.fail_temp_error.abs().mean(), ok.error_guess_T1.abs().mean(), ok.error_dev_mean.abs().mean()
    c1 = bool(len(ok) and (ok.fail_temp_error.abs() <= TOL_EACH).all() and mae <= TOL_MAE); c2 = bool(len(ok) and mae < mae1 and mae < maed)
    summary = dict(n_models=len(test), n_determined=len(ok), MAE_theory=mae, MAE_guess_T1=mae1, MAE_dev_mean=maed,
                   criterion1_accuracy=c1, criterion2_advantage_over_baselines=c2)
    pd.DataFrame([summary]).to_csv(OUT / "test_summary.csv", index=False)
    print(f"MAE: theory {mae:.4f} | guess T* = 1 {mae1:.4f} | development mean {maed:.4f}")
    print("criterion 1 (each |error| <= 0.02 and MAE <= 0.01):", "met" if c1 else "not met")
    print("criterion 2 (MAE below both baselines):", "met" if c2 else "not met")
    corr_file = test_dir / "predictions_corrected.json"
    if corr_file.exists():   # secondary predictor, design doc 8.8.20: banned-token correction, human text only
        corr = json.load(open(corr_file))
        t2 = test.set_index("generator")
        t2["fail_temp_predicted_corrected"] = [corr[g]["T_pred_corrected"] for g in t2.index]
        t2["delta"] = [corr[g]["delta"] for g in t2.index]
        t2["corrected_written_at_utc"] = [corr[g]["written_at_utc"] for g in t2.index]
        t2["fail_temp_error_corrected"] = t2.fail_temp_observed - t2.fail_temp_predicted_corrected
        t2 = t2.reset_index()[["generator", "delta", "fail_temp_predicted", "fail_temp_predicted_corrected", "fail_temp_observed",
                                "fail_temp_error", "fail_temp_error_corrected", "corrected_written_at_utc"]]
        t2.to_csv(OUT / "test_corrected.csv", index=False)
        okc = t2.dropna(subset=["fail_temp_error_corrected"]); maec = okc.fail_temp_error_corrected.abs().mean()
        c1c = bool(len(okc) and (okc.fail_temp_error_corrected.abs() <= TOL_EACH).all() and maec <= TOL_MAE); c2c = bool(len(okc) and maec < mae1 and maec < maed)
        print("\n== Secondary: corrected prediction (8.8.20; prospective for gpt-neo and bloom) ==")
        print(t2.drop(columns=["corrected_written_at_utc"]).round(4).to_string(index=False))
        print(f"MAE corrected {maec:.4f} vs original {mae:.4f}; criterion 1:", "met" if c1c else "not met", "| criterion 2:", "met" if c2c else "not met")
        summary.update(MAE_corrected=maec, corrected_criterion1=c1c, corrected_criterion2=c2c)
        pd.DataFrame([summary]).to_csv(OUT / "test_summary.csv", index=False)
    diag = []   # diagnostic, not pre-registered: the theory assumes E[mean(H - s)] = 0 for machine text at T = 1
    for tag in stored:
        s = per_text(pd.read_pickle(test_dir / f"scores_self_{tag}.pkl"))
        g1, h = s[np.isclose(s["T"], 1.0)], s[s.who == "human"]
        diag.append(dict(generator=tag, machine_c_at_T1=g1.c.mean(), machine_V_at_T1=g1.V.mean(), human_c=h.c.mean(), human_V=h.V.mean()))
    diag = pd.DataFrame(diag); diag.to_csv(OUT / "test_diagnostics.csv", index=False)
    print("\n== Diagnostic (not pre-registered): machine mean(H - s) at T = 1, zero in theory ==")
    print(diag.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
