# Power and error-rate simulations behind the plan's "檢定力" section (design doc 8.8.24, plan 5.1; implementation list item 4).
# Step 1 (document noise): real document resampling of notebook 13's per-text scores, emulating the registered design:
#   P and E drawn independently from the same 1,000 human texts (two disjoint groups), P by source, E by source x role
#   (A-half prompts, B-half prompts, reference); machine texts carry their prompt's weight. Per round and model:
#   T_hat, slope prediction b, T* (all prompts vs E), T_direct (A half vs P, rules 1-5), T*_B (B half vs E minus A half).
#   2,000 rounds, seed 20260930 (the procedure of the review's v5_noise.py; no number is copied by hand).
# Step 2: split that noise into a part shared by all models and a model-specific part (v5_decomp.py).
# Step 3: simulations, seed 20260930: table 1 (criteria 1, 2), table 2 (both chains, plan 5.1), table 3 (form R under
#   five nulls), validity of the TOST (19 fixed configurations) and of the 5.1 "better than" test (the same 19
#   configurations plus exact nulls), and the reviewers' two counterexamples run through e12's own functions.
#   The analysis bootstrap is emulated parametrically: replicate = observed + fresh document noise, crossed with a family
#   resample (11 of 11 with replacement, Danube kept as one family), 2,000 replicates as registered.
# Usage: python scripts/e12_power.py [all|noise|table1|table2|table3|tost|better|examples] [--quick]
#   Reads DATA/13_theory_prediction_test (set CMARGIN_DATA to the Drive data folder); writes RESULTS/e12_power/.
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1])); sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import argparse, json, math, time
import numpy as np, pandas as pd
from cmargin.paths import DATA, RESULTS
from cmargin.crossing import cross_down, direct_estimate
import e12_replication_test as e12

SEED = 20260930
OUT = RESULTS / "e12_power"
NB13 = pathlib.Path(DATA) / "13_theory_prediction_test"
E9 = pathlib.Path(__file__).resolve().parents[1] / "results" / "e9_theory_prediction"
TAGS = ("OLMo-2-0425-1B", "granite-3.3-2b-base", "gpt-neo-1.3B", "bloom-1b7")
QUANT = ("T_hat", "b", "T*", "T_direct", "T*_B")
F = 11; FAM = np.r_[0, np.arange(F)]; NM = len(FAM); MOV = np.arange(1, F); CNT = np.bincount(FAM, minlength=F)
FIXED = CNT > 1
LINE_RATIO = e12.LINE_K / e12.SLOPE_K       # line and slope share E_P[c], so their noise differs only by this factor
N_BOOT = e12.N_BOOT


# ------------------------------------------------------------------ known models (e9 results, not typed in by hand)
def known_models():
    dev = pd.read_csv(E9 / "development.csv"); test = pd.read_csv(E9 / "test.csv")
    k = pd.concat([dev, test], ignore_index=True)
    Ec = k.E_h_c.to_numpy()
    pool = np.column_stack([k.fail_temp_predicted, k.fail_temp_observed, 1 - e12.SLOPE_K * Ec, e12.LINE_A - e12.LINE_K * Ec])
    names = k.generator.tolist()
    assert names == ["Qwen2.5-3B", "SmolLM2-1.7B", "phi-2", "OLMo-2-0425-1B", "granite-3.3-2b-base", "gpt-neo-1.3B", "bloom-1b7"]
    return pool, names          # columns: T_hat, T*, slope prediction, line prediction


POOL7, NAMES7 = known_models()
POOLS = {"known7": POOL7, "test4": POOL7[3:], "test3_noBLOOM": POOL7[3:6]}


# ------------------------------------------------------------------ step 1: document noise from notebook 13
def load_nb13():
    Z = {}
    for tag in TAGS:
        s = pd.read_pickle(NB13 / f"scores_self_{tag}.pkl")
        st = np.array([e12.text_stats(a) for a in s.arr])
        T = np.array([float(d[1:]) if d != "human" else np.nan for d in s.decoding])
        h = (s.who == "human").to_numpy()
        order = np.argsort(s.id.to_numpy()[h]); assert (s.id.to_numpy()[h][order] == np.arange(1000)).all()
        Z[tag] = dict(c=st[h][order, 0], V=st[h][order, 1], fdg=st[h][order, 2], dom=s.domain.to_numpy()[h][order].astype(str))
        Ts = sorted(set(T[~h])); M = np.full((len(Ts), 300), np.nan)
        for k, t in enumerate(Ts):
            m = (~h) & np.isclose(T, t); M[k, s.id.to_numpy()[m]] = st[m, 2]
        assert np.isfinite(M).all()
        Z[tag].update(M=M, Ts=np.array(Ts))
    return Z


def noise_rounds(n=2000, seed=SEED):
    Z = load_nb13(); dom = Z[TAGS[0]]["dom"]; ids = np.arange(1000); Ts = Z[TAGS[0]]["Ts"]
    prompt = {s: np.sort(ids[(dom == s) & (ids < 300)]) for s in e12.SOURCES}
    A = {s: prompt[s][:25] for s in e12.SOURCES}; B = {s: prompt[s][25:] for s in e12.SOURCES}
    REF = {s: ids[(dom == s) & (ids >= 300)] for s in e12.SOURCES}
    isA = np.isin(np.arange(300), np.concatenate(list(A.values()))).astype(float); isB = 1 - isA
    sets = {t: e12.AucSet(Z[t]["M"], Z[t]["fdg"]) for t in TAGS}

    def mult(rng, k, pool):
        w = np.zeros(1000); np.add.at(w, rng.choice(pool, k), 1.0); return w

    def stats(wP, wA, wB, wR):
        wE, wEB, wM = wA + wB + wR, wB + wR, (wA + wB)[:300]
        out = []
        for t in TAGS:
            z = Z[t]; Ec = (wP * z["c"]).sum() / wP.sum()
            th = 1 - (wP * z["c"]).sum() / (wP * z["V"]).sum(); b = 1 - e12.SLOPE_K * Ec
            ts = cross_down(Ts, sets[t].auc(wM, wE)); tsB = cross_down(Ts, sets[t].auc(wM * isB, wEB))
            td, _ = direct_estimate(Ts, sets[t].auc(wM * isA, wP))
            out.append([th, b] + [np.nan if x is None else x for x in (ts, td, tsB)])
        return np.array(out)

    one = np.ones(1000)
    obs = stats(one, np.isin(ids, np.concatenate(list(A.values()))).astype(float),
                np.isin(ids, np.concatenate(list(B.values()))).astype(float), (ids >= 300).astype(float))
    rng = np.random.default_rng(seed); R = []
    for _ in range(n):
        wP = sum(mult(rng, int((dom == s).sum()), ids[dom == s]) for s in e12.SOURCES)
        wA = sum(mult(rng, 25, A[s]) for s in e12.SOURCES); wB = sum(mult(rng, 25, B[s]) for s in e12.SOURCES)
        wR = sum(mult(rng, len(REF[s]), REF[s]) for s in e12.SOURCES)
        R.append(stats(wP, wA, wB, wR))
    return obs, np.array(R)


def decompose(R):
    """Common (shared by all models) and model-specific covariance of the five quantities."""
    D = R - np.nanmean(R, 0); D = np.nan_to_num(D)
    Bn, M, K = D.shape
    tot = sum(D[:, m].T @ D[:, m] for m in range(M)) / (Bn - 1) / M
    com = sum(D[:, m].T @ D[:, m2] for m in range(M) for m2 in range(M) if m != m2) / (Bn - 1) / (M * (M - 1))
    com = (com + com.T) / 2; spec = tot - com

    def psd(C):
        w, V = np.linalg.eigh(C); return (V * np.clip(w, 0, None)) @ V.T
    return psd(com), psd(spec)


def get_noise(quick=False):
    """Recomputed on every run (about two seconds), so no stale cache can stand in for it."""
    t0 = time.time(); obs, R = noise_rounds(200 if quick else 2000); com, spec = decompose(R)
    print(f"noise: {len(R)} document-resampling rounds of notebook 13 in {time.time() - t0:.0f}s; non-finite share",
          np.round(np.isnan(R).mean(axis=(0, 1)), 4))
    return com, spec, obs, R


# ------------------------------------------------------------------ simulation machinery
class Sim:
    def __init__(self, com, spec):
        self.Lc = np.linalg.cholesky(com + 1e-14 * np.eye(5)); self.Ls = np.linalg.cholesky(spec + 1e-14 * np.eye(5))

    def noise(self, rng, size):
        """Document noise of (T_hat, b, T*, T_direct, T*_B) for the 12 models, shape size + (12, 5)."""
        return rng.standard_normal(size + (1, 5)) @ self.Lc.T + rng.standard_normal(size + (NM, 5)) @ self.Ls.T


def fmean(x):
    out = np.zeros(x.shape[:-1] + (F,))
    for f in range(F):
        out[..., f] = x[..., FAM == f].mean(-1)
    return out


def dvals(q, line):
    th, b, ts, td, tb = (q[..., i] for i in range(5))
    return dict(slope=np.abs(ts - b) - np.abs(ts - th), direct=np.abs(tb - td) - np.abs(tb - th),
                line=np.abs(ts - line) - np.abs(ts - th), known_mean=np.abs(ts - e12.KNOWN_MEAN) - np.abs(ts - th),
                guess1=np.abs(ts - 1) - np.abs(ts - th))


def draw_truth(rng, pool):
    j = rng.integers(0, len(pool), NM); p = pool[j]
    return np.column_stack([p[:, 0], p[:, 2], p[:, 1], p[:, 1], p[:, 1]]), p[:, 3]


def observe(sim, rng, tq, tl):
    e = sim.noise(rng, ()); return tq + e, tl + LINE_RATIO * e[:, 1]


def crossed_boot(sim, rng, q, line, keys, nboot=N_BOOT):
    """Emulated e12 bootstrap: observed values plus fresh document noise, crossed with an 11-of-11 family resample."""
    eb = sim.noise(rng, (nboot,)); db = dvals(q + eb, line + LINE_RATIO * eb[..., 1]); idx = rng.integers(0, F, (nboot, F))
    return {k: np.take_along_axis(fmean(db[k]), idx, 1).mean(1) for k in keys}


def form_r_sim(fth, fts, prng, nperm):
    x, y = fth - fth.mean(), fts - fts.mean(); r0 = (x * y).sum()
    P = np.tile(np.arange(F), (nperm, 1)); P[:, MOV] = MOV[np.argsort(prng.random((nperm, F - 1)), axis=1)]
    p = (1 + ((x[P] * y).sum(1) >= r0 - 1e-15).sum()) / (1 + nperm)
    return p <= e12.ALPHA_C2 and r0 > 0


def better(boot):
    return e12.quantile_inf(boot, e12.ALPHA_CHAIN) > 0


def equivalent(boot):
    return e12.quantile_inf(boot, e12.ALPHA_CHAIN) > -e12.EQ_MARGIN and e12.quantile_inf(boot, 1 - e12.ALPHA_CHAIN) < e12.EQ_MARGIN


# ------------------------------------------------------------------ tables 1 and 2
def scenario(sim, pool, nsim, nperm=2000, seed=SEED):
    rng = np.random.default_rng(seed); prng = np.random.default_rng(seed + 1)
    g = {k: np.zeros(nsim, bool) for k in ("c1", "R", "A_slope", "A_direct", "A_line", "B_slope", "B_direct",
                                           "K_known_mean", "K_guess1", "better_known_mean", "better_guess1")}
    for i in range(nsim):
        q, line = observe(sim, rng, *draw_truth(rng, pool))
        th, ts = q[:, 0], q[:, 2]; err = np.abs(ts - th)
        g["c1"][i] = (err <= e12.TOL_EACH).all() and fmean(err).mean() <= e12.TOL_MAE
        g["R"][i] = form_r_sim(fmean(th), fmean(ts), prng, nperm)
        d = {k: fmean(v) for k, v in dvals(q, line).items()}
        g["K_known_mean"][i] = (d["known_mean"] > 0).sum() >= 10; g["K_guess1"][i] = (d["guess1"] > 0).sum() >= 10
        bt = crossed_boot(sim, rng, q, line, ("slope", "direct", "line", "known_mean", "guess1"))
        for k in ("slope", "direct", "line"):
            g["A_" + k][i] = better(bt[k])
        for k in ("slope", "direct"):
            g["B_" + k][i] = equivalent(bt[k])
        g["better_known_mean"][i] = better(bt["known_mean"]); g["better_guess1"][i] = better(bt["guess1"])
    gate = g["R"]; a = gate.copy(); b = gate.copy(); out = dict(c1=g["c1"].mean(), R=gate.mean(), c1_and_R=(g["c1"] & gate).mean())
    for k in ("slope", "direct", "line"):
        a = a & g["A_" + k]; out["A_" + k] = a.mean()
    for k in ("slope", "direct"):
        b = b & g["B_" + k]; out["B_" + k] = b.mean()
    out.update(ref_K10_known_mean=g["K_known_mean"].mean(), ref_K10_guess1=g["K_guess1"].mean(),
               ref_better_known_mean_first=(gate & g["better_known_mean"]).mean(), ref_better_guess1=(gate & g["better_guess1"]).mean(),
               alone_A_slope=g["A_slope"].mean(), alone_A_direct=g["A_direct"].mean())
    return {k: float(v) for k, v in out.items()}


# ------------------------------------------------------------------ table 3: form R under five nulls
def table3(sim_noise, nsim, nperm=999, seed=SEED):
    """Registered form R (Danube fixed, 10 families permuted) when T_hat and T* are unrelated. Noise: P-type common +
    specific for T_hat, E-type common + specific (machine included) for T*, as in the review's perm_big.py."""
    Pc, Ps, Ec, Es = sim_noise
    known = POOL7[:, :2]
    gens = {"B 的情境：T̂、T* 各自從已知 7 個模型獨立抽": lambda r: (known[r.integers(0, 7, NM), 0], known[r.integers(0, 7, NM), 1]),
            "獨立的常態邊際分布": lambda r: (1.0125 + 0.0085 * r.standard_normal(NM), 1.0089 + 0.0065 * r.standard_normal(NM)),
            "A 的反例（T̂ 全部 1.0045；T* 90% 1.0025、10% 1.0）": lambda r: (np.full(NM, 1.0045), np.where(r.random(NM) < 0.9, 1.0025, 1.0)),
            "獨立的偏斜（對數常態）邊際分布": lambda r: (1 + r.lognormal(-5.3, 0.9, NM), 1 + r.lognormal(-5.5, 0.9, NM)),
            "家族之間沒有差異，只有文件雜訊": lambda r: (np.full(NM, 1.0), np.full(NM, 1.005))}
    out = {}
    for lab, gen in gens.items():
        rng = np.random.default_rng(seed); rej = 0
        for _ in range(nsim):
            th0, ts0 = gen(rng)
            th = th0 + Pc * rng.standard_normal() + Ps * rng.standard_normal(NM)
            ts = ts0 + Ec * rng.standard_normal() + Es * rng.standard_normal(NM)
            x, y = np.bincount(FAM, th, F) / CNT, np.bincount(FAM, ts, F) / CNT
            x = x - x.mean(); y = y - y.mean(); r0 = (x * y).sum()
            P = np.tile(np.arange(F), (nperm, 1)); P[:, MOV] = MOV[np.argsort(rng.random((nperm, F - 1)), axis=1)]
            rej += (1 + ((x[P] * y).sum(1) >= r0 - 1e-15).sum()) / (1 + nperm) <= e12.ALPHA_C2 and r0 > 0
        out[lab] = rej / nsim
    return out


# ------------------------------------------------------------------ validity of the TOST and of the 5.1 "better than" test
def configurations(rng):
    """19 fixed configurations: 7 homogeneous (all 12 models share one known model's values) + 6 + 6 drawn ones."""
    confs = [(f"homogeneous {n}", np.tile([POOL7[i, 0], POOL7[i, 2], POOL7[i, 1], POOL7[i, 1], POOL7[i, 1]], (NM, 1)),
              np.full(NM, POOL7[i, 3])) for i, n in enumerate(NAMES7)]
    for pool in ("known7", "test4"):
        for c in range(6):
            tq, tl = draw_truth(rng, POOLS[pool]); confs.append((f"{pool} draw {c}", tq, tl))
    return confs


def true_delta(sim, rng, tq, tl, keys, ntrue=20000):
    e = sim.noise(rng, (ntrue,)); dv = dvals(tq + e, tl + LINE_RATIO * e[..., 1])
    return {k: float(fmean(dv[k]).mean(-1).mean()) for k in keys}


def validity(sim, nsim, seed=SEED + 11):
    """Share of experiments whose interval lies wholly above (or below) the true Delta: the error rate of a one-sided
    test whose boundary sits at Delta. TOST: 1.25% per side. 'Better than' (5.1): lower 1.25% bound above Delta."""
    rng = np.random.default_rng(seed); rows = []
    keys = ("slope", "direct", "line")
    for name, tq, tl in configurations(rng):
        D = true_delta(sim, rng, tq, tl, keys)
        cnt = {(k, s): 0 for k in keys for s in ("above", "below")}
        for _ in range(nsim):
            q, line = observe(sim, rng, tq, tl)
            bt = crossed_boot(sim, rng, q, line, keys)
            for k in keys:
                lo, hi = e12.quantile_inf(bt[k], e12.ALPHA_CHAIN), e12.quantile_inf(bt[k], 1 - e12.ALPHA_CHAIN)
                cnt[(k, "above")] += lo > D[k]; cnt[(k, "below")] += hi < D[k]
        rows.append(dict(config=name, **{f"Delta_{k}": D[k] for k in keys}, **{f"{k}_{s}": v / nsim for (k, s), v in cnt.items()}))
    return rows


def exact_nulls(sim, nsim, seed=SEED + 21):
    """Configurations whose true Delta is 0 (or below) by construction; share of experiments that pass the 5.1 test.
    Includes the two cases where the version-5 sign test failed badly (all T* above both predictions: 21.7%)."""
    rng = np.random.default_rng(seed); out = {}

    def run(tq, tl, key):
        hits = 0
        for _ in range(nsim):
            q, line = observe(sim, rng, tq, tl)
            hits += better(crossed_boot(sim, rng, q, line, (key,))[key])
        return hits / nsim

    def truth(i_models, margin=None, key="slope", shift=0.0):
        p = POOL7[i_models]; b, ln = p[:, 2], p[:, 3]
        base = b if key == "slope" else ln
        th = base + shift; ts = p[:, 1] if margin is None else np.maximum(th, base) + margin
        return np.column_stack([th, b, ts, ts, ts]), ln

    idx = rng.integers(0, 7, NM)
    for key in ("slope", "line"):
        for lab, margin in (("T* 像已知 7 個那樣分散", None), ("所有 T* 都高於兩個預測 0.005", 0.005), ("所有 T* 都高於兩個預測 0.02", 0.02)):
            tq, tl = truth(idx, margin, key)
            D = true_delta(sim, rng, tq, tl, (key,))[key]
            out[f"{key}：{lab}"] = dict(Delta=D, false_pass=run(tq, tl, key))
    # direct estimate: the formula is less noisy than the direct estimate, so an unbiased formula has Delta > 0.
    # Shift every T_hat away from T* until Delta = 0 (bisection on common random numbers), then run the test.
    for lab, pool in (("T* 像已知 7 個", POOL7), ("T* 像第一次檢驗 4 個", POOLS["test4"])):
        p = pool[rng.integers(0, len(pool), NM)]
        e = sim.noise(np.random.default_rng(seed + 99), (20000,))

        def delta(s):
            tq = np.column_stack([p[:, 1] + s, p[:, 2], p[:, 1], p[:, 1], p[:, 1]])
            return float(fmean(dvals(tq + e, p[:, 3] + LINE_RATIO * e[..., 1])["direct"]).mean(-1).mean()), tq
        lo, hi = 0.0, 0.02
        for _ in range(40):
            mid = (lo + hi) / 2; lo, hi = (mid, hi) if delta(mid)[0] > 0 else (lo, mid)
        D, tq = delta(hi)
        out[f"direct：{lab}，T̂ 平移 {hi:.5f} 使 Δ = 0"] = dict(Delta=D, false_pass=run(tq, p[:, 3], "direct"))
    return out


# ------------------------------------------------------------------ reviewers' counterexamples through e12's own code
def reviewer_examples():
    x = np.r_[1.005, 1 + 0.0001 * np.arange(10)]; y = np.r_[0.995, 1 + 0.0001 * np.arange(10)]
    fixed = np.r_[True, np.zeros(10, bool)]
    r, p = e12.form_r(x, y, fixed, e12.N_PERM, np.random.default_rng(e12.SEED_PERM))
    ex1 = dict(r=r, p=p, criterion2=e12.c2_pass(r, p))
    # A7 (Danube): model 1 d = +0.002; model 2 direct estimate missing and T*_B missing (case 4); 9 wins, 1 loss elsewhere
    line = lambda c: 0.5 - 3.0 * (np.array(e12.TEMPS) - c)
    t = np.array(e12.TEMPS)
    m1 = e12.direct_case(1.010, e12.direct_estimate(t, line(1.012))[0], *e12.endpoint_state(t, line(1.010))[:2])
    td2 = e12.direct_estimate(t, np.full(11, 0.5))[0]
    m2_with = e12.direct_case(1.010, td2, *e12.endpoint_state(t, line(1.014))[:2])
    m2_without = e12.direct_case(1.010, td2, *e12.endpoint_state(t, np.full(11, 0.6))[:2])
    fam_of = np.r_[0, 0, np.arange(1, 11)]
    pairs = [m1, m2_without] + [(1, 0.001)] * 9 + [(1, -0.001)]
    K = e12.describe_families({"direct": pairs}, fam_of, 11)["direct"]["K"]
    pairs_with = [m1, m2_with] + [(1, 0.001)] * 9 + [(1, -0.001)]
    K_with = e12.describe_families({"direct": pairs_with}, fam_of, 11)["direct"]["K"]
    return dict(A3_negative_r=ex1, A7_danube=dict(model1=m1, model2_with_endpoint=m2_with, model2_without_endpoint=m2_without,
                                                   K_case4=K, K_with_endpoint=K_with))


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", default="all")
    ap.add_argument("--quick", action="store_true", help="small run counts, for a smoke test only")
    a = ap.parse_args(argv); q = a.quick
    OUT.mkdir(parents=True, exist_ok=True)
    com, spec, obs, R = get_noise(q)
    sim = Sim(com, spec)
    res = {"seed": SEED, "noise_sd": {"common": dict(zip(QUANT, np.sqrt(np.diag(com)).round(6).tolist())),
                                      "specific": dict(zip(QUANT, np.sqrt(np.diag(spec)).round(6).tolist()))}}
    todo = ["table1", "table2", "table3", "tost", "better", "examples"] if a.what == "all" else [a.what]
    t0 = time.time()
    if "table1" in todo or "table2" in todo:
        res["tables_1_2"] = {n: scenario(sim, pool, 300 if q else 10000) for n, pool in POOLS.items()}
    if "table3" in todo:
        sd = lambda i: math.sqrt(com[i, i]), lambda i: math.sqrt(spec[i, i])
        res["table3"] = table3((sd[0](0), sd[1](0), sd[0](2), sd[1](2)), 2000 if q else 100000)
    if "tost" in todo or "better" in todo:
        res["validity_19"] = validity(sim, 200 if q else 2000)
    if "better" in todo:
        res["exact_nulls"] = exact_nulls(sim, 400 if q else 10000)
    if "examples" in todo:
        res["examples"] = reviewer_examples()
    res["seconds"] = round(time.time() - t0)
    name = "power_quick.json" if q else f"power_{a.what}.json"
    (OUT / name).write_text(json.dumps(e12.clean(res), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(e12.clean(res), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
