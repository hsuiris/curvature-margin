"""Acceptance tests for scripts/e12_replication_test.py (plan 8.8.24 version 5.1, implementation list item 3).
Synthetic data only; the bootstrap and permutation counts are reduced where only the decision logic is tested."""
import copy, json, math, pathlib, shutil, sys

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[0] / "scripts"))
import e12_replication_test as e12   # noqa: E402
import synth12 as S                  # noqa: E402

FAST = dict(n_boot=200, n_perm=5000, n_perm_boot=200)
T = np.array(e12.TEMPS)


def run(specs, **kw):
    return e12.analyze(S.build(specs), **{**FAST, **kw})


@pytest.fixture(scope="module")
def good():
    return run(S.standard())


# ---------------------------------------------------------------- registered constants
def test_registered_constants():
    assert e12.TEMPS == (0.94, 0.97, 0.99, 1.00, 1.01, 1.02, 1.03, 1.04, 1.06, 1.09, 1.14)
    assert (e12.N_PERM, e12.SEED_PERM, e12.N_BOOT, e12.SEED_DOC, e12.SEED_FAM, e12.N_PERM_BOOT, e12.SEED_PERM_BOOT) == \
        (100_000, 20260930, 2000, 20260932, 20260931, 10_000, 20260933)
    assert (e12.ALPHA_C2, e12.ALPHA_CHAIN, e12.EQ_MARGIN, e12.TOL_EACH, e12.TOL_MAE) == (0.025, 0.0125, 0.002, 0.02, 0.005)
    assert (e12.KNOWN_MEAN, e12.SLOPE_K, e12.LINE_A, e12.LINE_K) == (1.0089, 0.1451, 1.0004, 0.1409)
    assert e12.CHAIN_A == ("slope", "direct", "line") and e12.CHAIN_B == ("slope", "direct")   # plan 5.1: known mean left chain A
    assert sum(v[1] for v in e12.QUOTA.values()) == 1000 and e12.REF_SIZES == {"main": 1000, "B": 850, "direct": 1000}
    assert (e12.STAGE2_DAYS, e12.MAX_ATTEMPTS) == (14, 4)        # plan: stage 2 within 14 days; first run plus three reruns
    assert e12.VARIANT_B_RANGE == (0.80, 1.30) and e12.RETOK_SHIFT == 0.0010


# ---------------------------------------------------------------- helpers
def test_quantile_inf():
    x = np.random.default_rng(1).normal(size=2000)
    for qq in (0.0125, 0.025, 0.5, 0.9875):
        assert e12.quantile_inf(x, qq) == float(np.quantile(x, qq))
    y = x.copy(); y[:20] = -np.inf          # 1% of the rounds missing: the 1.25% bound stays finite
    assert math.isfinite(e12.quantile_inf(y, 0.0125))
    y[:30] = -np.inf                        # 1.5% missing: the lower bound becomes -inf
    assert e12.quantile_inf(y, 0.0125) == -math.inf
    z = x.copy(); z[:30] = np.inf
    assert e12.quantile_inf(z, 0.9875) == math.inf
    with pytest.raises(AssertionError):
        e12.quantile_inf(np.r_[x, np.nan], 0.5)


def test_weighted_auroc_equals_sklearn():
    rng = np.random.default_rng(2)
    g = np.round(rng.normal(0.3, 1, 300), 1); h = np.round(rng.normal(0, 1, 1000), 1)    # rounding makes ties
    auc = e12.wauc(g, np.ones(300), h, np.ones(1000))
    assert abs(auc - roc_auc_score(np.r_[np.ones(300), np.zeros(1000)], np.r_[g, h])) < 1e-12
    s = e12.AucSet(np.vstack([g, g + 0.5]), h)
    wg, wh = rng.integers(0, 3, 300).astype(float), rng.integers(0, 3, 1000).astype(float)
    a = s.auc(wg, wh)
    assert abs(a[0] - e12.wauc(g, wg, h, wh)) < 1e-12 and abs(a[1] - e12.wauc(g + 0.5, wg, h, wh)) < 1e-12


def test_improvement_and_worst_case():
    assert e12.improvement(1.01, 1, 1.008, 1.0) == (pytest.approx(0.01 - 0.002), "實測")
    assert e12.improvement(None, 2, 1.008, 1.0) == (pytest.approx(-0.008), "最壞情境界限")
    assert e12.improvement(None, 3, 1.008, 1.0) == (pytest.approx(-0.008), "最壞情境界限")
    assert e12.improvement(1.01, 1, None, 1.0) == (None, "預測未寫出")
    # the worst-case bound is never above the value any crossing would give (triangle inequality)
    rng = np.random.default_rng(3)
    for Ts, Th, b in rng.uniform(0.9, 1.2, (2000, 3)):
        assert e12.improvement(None, 2, Th, b)[0] <= abs(Ts - b) - abs(Ts - Th) + 1e-15


def test_direct_estimate_four_cases():
    assert e12.direct_case(1.010, 1.012, 1, 1.010) == (1, pytest.approx(0.002))
    assert e12.direct_case(1.010, 1.014, 2, None) == (2, pytest.approx(-0.004))
    assert e12.direct_case(1.010, None, 1, 1.014) == (3, pytest.approx(-0.004))
    assert e12.direct_case(1.010, None, 3, None) == (4, 0.0)
    assert e12.direct_case(None, 1.01, 1, 1.01) == (0, None)


def test_reviewer_A7_danube_counterexample():
    """Model 1 d = +0.002; model 2 has neither the direct estimate nor T*_B (case 4); the other ten families win nine."""
    line = lambda c: 0.5 - 3.0 * (T - c)
    m1 = e12.direct_case(1.010, e12.direct_estimate(T, line(1.012))[0], *e12.endpoint_state(T, line(1.010))[:2])
    td = e12.direct_estimate(T, np.full(11, 0.5))
    assert td == (None, 5)
    m2 = e12.direct_case(1.010, td[0], *e12.endpoint_state(T, np.full(11, 0.6))[:2])
    assert m1 == (1, pytest.approx(0.002)) and m2 == (4, 0.0)
    fam_of = np.r_[0, 0, np.arange(1, 11)]
    d = e12.describe_families({"direct": [m1, m2] + [(1, 0.001)] * 9 + [(1, -0.001)]}, fam_of, 11)["direct"]
    assert d["K"] == 9 and d["d_f"][0] == 0.0 and 0 in d["excluded"]      # old rule: +0.001 and K = 10
    m2b = e12.direct_case(1.010, td[0], *e12.endpoint_state(T, line(1.014))[:2])
    d = e12.describe_families({"direct": [m1, m2b] + [(1, 0.001)] * 9 + [(1, -0.001)]}, fam_of, 11)["direct"]
    assert m2b == (3, pytest.approx(-0.004)) and d["K"] == 9 and d["d_f"][0] == pytest.approx(-0.001)


def _obs(n=12, case4=None):
    """Observed per-model values with every main endpoint in state 1; model `case4` lacks both direct values."""
    v = []
    for i in range(n):
        x = dict(T_hat=1.01, b=e12.baseline_values(-0.05), main=(1, 1.012, "向下交會"), direct=(1.02, 2), B=(1, 1.012, "向下交會"))
        if i == case4:
            x.update(direct=(None, 5), B=(2, None, "網格內未失效"))
        v.append(x)
    return v


def test_chain_A_stops_on_incomplete_direct_data():
    obs = _obs(case4=1)
    fam_desc = {k: dict(dbar=0.001, K=9) for k in e12.BASELINES + ("direct",)}
    boot = {k: np.full(200, 0.004) for k in ("slope", "direct", "line", "known_mean", "guess1")}
    ch = e12.run_chains(True, obs, boot, fam_desc, 11)
    assert [s["status"] for s in ch["A"]] == ["通過", "資料不完整（未通過）", "未檢定"]
    assert [s["status"] for s in ch["B"]] == ["未通過", "未檢定"]            # 0.004 lies outside +-0.002, so B stops at once
    boot_eq = dict(boot, slope=np.full(200, 0.0005))                       # inside +-0.002: chain B reaches the direct step
    chB = e12.run_chains(True, obs, boot_eq, fam_desc, 11)["B"]
    assert [s["status"] for s in chB] == ["通過", "未檢定"] and chB[1]["reason"] == "資料不完整"   # plan 3.4: incomplete -> untested
    assert [s["status"] for s in e12.run_chains(False, obs, boot, fam_desc, 11)["A"]] == ["未檢定"] * 3


def test_chain_stops_after_a_failed_step_and_missing_rounds_count_against():
    obs = _obs()
    fam_desc = {k: dict(dbar=0.0, K=5) for k in e12.BASELINES + ("direct",)}
    x = np.linspace(-0.0015, 0.0015, 200)
    boot = {"slope": x + 0.0005, "direct": x, "line": x, "known_mean": x, "guess1": x}
    ch = e12.run_chains(True, obs, boot, fam_desc, 11)
    assert [s["status"] for s in ch["A"]] == ["未通過", "未檢定", "未檢定"]
    assert ch["B"][0]["status"] == "通過" and ch["B"][0]["lower"] > -0.002 and ch["B"][0]["upper"] < 0.002
    y = x.copy(); y[:5] = np.nan                                  # 2.5% of rounds missing: -inf / +inf in the bounds
    boot2 = dict(boot, slope=y + 0.0005)
    ch = e12.run_chains(True, obs, boot2, fam_desc, 11)
    assert ch["B"][0]["lower"] == -math.inf and ch["B"][0]["upper"] == math.inf and ch["B"][0]["status"] == "未通過"
    pos = np.full(200, 0.003); pos[:2] = np.nan                   # 1% missing: the 1.25% lower bound is still finite and > 0
    assert e12.run_chains(True, obs, dict(boot, slope=pos), fam_desc, 11)["A"][0]["status"] == "通過"


def test_reviewer_A3_negative_all_family_r():
    x = np.r_[1.005, 1 + 0.0001 * np.arange(10)]; y = np.r_[0.995, 1 + 0.0001 * np.arange(10)]
    fixed = np.r_[True, np.zeros(10, bool)]
    r, p = e12.form_r(x, y, fixed, e12.N_PERM, np.random.default_rng(e12.SEED_PERM))
    assert round(r, 3) == -0.929 and p == pytest.approx(1 / 100_001) and not e12.c2_pass(r, p)
    r10, p10 = e12.form_r(x[1:], y[1:], fixed[1:], e12.N_PERM, np.random.default_rng(e12.SEED_PERM))
    assert p10 == p and r10 > 0            # the fixed family never changes the p value, only the all-family r


def test_form_r_constant_values_give_p_one():
    assert e12.form_r(np.full(11, 1.01), np.linspace(1, 1.02, 11), np.zeros(11, bool), 1000, np.random.default_rng(0))[1] == 1.0


# ---------------------------------------------------------------- whole analysis on synthetic data
def test_success_path(good):
    assert good["criterion1"]["passed"] and good["criterion2"]["passed"] and good["success"]
    assert good["F"] == 11 and good["fixed_families"] == ["Danube"]
    A = {s["comparison"]: s for s in good["chain_A"]}
    assert A["slope"]["status"] == "通過" and A["slope"]["lower"] > 0
    assert A["direct"]["status"] in ("未通過", "通過")
    rows = [r["model"] for r in good["per_model"]]
    assert len(rows) == 12 and all(r["state"] == 1 for r in good["per_model"])
    assert good["described"]["guess1"]["F"] == 11 and "dbar_interval" in good["described"]["known_mean"]


def test_state_two_three_missing_and_worst_case_bounds():
    specs = S.standard(m0=dict(pattern="above"), m2=dict(pattern="below"), m3=dict(pattern="up", T_star=1.02),
                       m4=dict(missing="text"))
    res = run(specs)
    pm = {r["model"]: r for r in res["per_model"]}
    assert (pm["m1"]["state"], pm["m1"]["reason"]) == (2, "網格內未失效")
    assert (pm["m3"]["state"], pm["m3"]["reason"]) == (2, "網格起點已失效")
    assert (pm["m4"]["state"], pm["m4"]["reason"]) == (3, "只有向上穿越")
    assert pm["m5"]["state"] == 3 and "文字路徑" in pm["m5"]["issues"]
    for name in ("m1", "m3", "m4", "m5"):
        r = pm[name]
        for k in e12.BASELINES:
            assert r[f"d_{k}_kind"] == "最壞情境界限" and r[f"d_{k}"] == pytest.approx(-abs(r["T_hat"] - r[f"b_{k}"]))
    assert pm["m1"]["extrapolated"] is not None and pm["m1"]["extrapolated"] > 1.14 and pm["m4"]["extrapolated"] is None
    r = pm["m1"]                                                   # state 2, every AUROC above 0.5: "T_hat - b" if both <= 1.14
    for k in e12.BASELINES:
        b0 = r[f"b_{k}"]
        assert r[f"d_{k}_if_outside"] == (pytest.approx(r["T_hat"] - b0) if max(b0, r["T_hat"]) <= 1.14 else None)
    r = pm["m3"]                                                   # state 2, every AUROC below 0.5: "b - T_hat" if both >= 0.94
    for k in e12.BASELINES:
        b0 = r[f"b_{k}"]
        assert r[f"d_{k}_if_outside"] == (pytest.approx(b0 - r["T_hat"]) if min(b0, r["T_hat"]) >= 0.94 else None)
    assert not res["criterion1"]["passed"] and not res["criterion2"]["computed"] and not res["success"]
    assert all(s["status"] == "未檢定" for s in res["chain_A"] + res["chain_B"])
    md, rows = e12.claims(res)
    assert "標準 1 不符合" in rows and "網格內沒有交會（狀態②）" in md and "原因未定（狀態③）" in md


def test_prediction_not_written_counts_everywhere():
    specs = S.standard(m6=dict(written=False))           # the second Danube model
    res = run(specs)
    pm = {r["model"]: r for r in res["per_model"]}
    assert not pm["m7"]["prediction_written"] and pm["m7"]["reason"] == "預測未寫出" and pm["m7"]["T_hat"] is None
    assert not res["criterion1"]["passed"] and not res["criterion2"]["passed"] and not res["success"]
    fam = {f["family"]: f for f in res["per_family"]}["Danube"]
    for k in e12.BASELINES + ("direct",):
        assert fam[f"d_f_{k}"] is None or fam[f"d_f_{k}"] <= 0          # never a formula win
    md, rows = e12.claims(res)
    assert "預測未寫出" in md and "標準 1 不符合" in rows


def test_single_model_family_without_prediction_is_not_a_win_and_is_listed():
    res = run(S.standard(m0=dict(written=False)))
    fam = {f["family"]: f for f in res["per_family"]}["Llama"]
    assert all(fam[f"d_f_{k}"] is None for k in e12.BASELINES + ("direct",))


def test_criterion1_only():
    specs = S.standard()
    rng = np.random.default_rng(5)
    for s, t, e in zip(specs, rng.permutation(np.linspace(1.004, 1.008, 12)), rng.permutation(np.linspace(-0.004, 0.004, 12))):
        s["T_star"], s["T_hat"] = t, 1.006 + e   # failure temperatures close together; formula values unrelated to them
    res = run(specs)
    assert res["criterion1"]["passed"] and not res["criterion2"]["passed"]
    assert "只符合標準 1" in e12.claims(res)[1]


def test_reviewer_A3_end_to_end():
    """Ten single-model families with T_hat = T* = 1 + 0.0001 i; both Danube models T_hat = 1.005, T* = 0.995."""
    specs = [S.model(f"s{i}", f"F{i}", 1 + 0.0001 * i, kappa=200.0) for i in range(10)]
    specs += [S.model("d1", "Danube", 0.995, T_hat=1.005, kappa=200.0), S.model("d2", "Danube", 0.995, T_hat=1.005, kappa=200.0)]
    res = run(specs, n_perm=e12.N_PERM)
    c1, c2 = res["criterion1"], res["criterion2"]
    assert c1["passed"] and c1["family_mae"] < 0.0011
    assert c2["r"] < -0.9 and c2["p"] <= 0.025 and not c2["passed"] and c2["p_ok_but_r_not_positive"]
    assert "標準 2：p ≤ 0.025 但全體 r ≤ 0" in e12.claims(res)[1]


def test_direct_case_four_in_a_two_model_family_end_to_end():
    specs = S.standard(m6=dict(pattern="flatA_aboveB"))
    res = run(specs)
    pm = {r["model"]: r for r in res["per_model"]}
    assert pm["m7"]["direct_rule"] == 5 and pm["m7"]["state_B"] == 2 and pm["m7"]["direct_case"] == 4
    fam = {f["family"]: f for f in res["per_family"]}["Danube"]
    assert fam["d_f_direct"] <= 0 and "Danube" in res["direct_excluded_families"]


def test_leave_one_family_out_and_groups(good):
    loo = {x["dropped"]: x for x in good["leave_one_family_out"]}
    assert set(loo) == set(good["families"]) and all("dbar_slope" in x for x in loo.values())
    assert loo["Danube"]["p"] == good["criterion2"]["p"]       # dropping the fixed family leaves p unchanged
    assert good["tokenizer_groups"]["n_groups"] == 11
    assert set(good["groups"]) == {"lang_group=other", "strip_group=keep"}


def test_claims_every_row_is_reachable(good):
    """Every row of the plan's claims table (version 5.1) is selected by at least one result."""
    seen = set(e12.claims(good)[1])

    def variant(**edit):
        r = copy.deepcopy(good)
        for path, val in edit.items():
            obj = r; keys = path.split(".")
            for k in keys[:-1]:
                obj = obj[int(k)] if k.isdigit() else obj[k]
            obj[int(keys[-1]) if keys[-1].isdigit() else keys[-1]] = val
        return set(e12.claims(r)[1])

    both = {"chain_A.0.status": "通過", "chain_A.0.passed": True, "chain_A.1.status": "通過", "chain_A.1.passed": True}
    seen |= variant(**both)
    seen |= variant(**{"chain_A.0.status": "未通過", "chain_A.0.passed": False, "chain_B.0.status": "通過", "chain_B.0.passed": True,
                       "chain_B.0.upper": 0.001})
    seen |= variant(**{"chain_A.0.status": "通過", "chain_A.0.passed": True, "chain_A.1.status": "未通過", "chain_A.1.passed": False})
    seen |= variant(**{"criterion2.passed": False, "success": False})
    seen |= variant(**{"criterion1.passed": False, "success": False})
    seen |= variant(**{**both, "chain_A.0.dbar": -0.0001})
    seen |= variant(**{"leave_one_family_out.0.passed": False})
    seen |= variant(**{"criterion2.passed": False, "success": False, "criterion2.p_ok_but_r_not_positive": True})
    table = ["標準 1、2 都符合，「優於」順序也通過斜率與直接估計", "標準 1、2 符合，「優於」順序的斜率未通過",
             "標準 1、2 與斜率通過，直接估計未通過", "只符合標準 1", "標準 1 不符合", "「優於」順序任何一步未通過或未輪到",
             "「優於」順序某一步通過：", "「優於」順序某一步通過，但該比較的家族平均改善量 d̄ ≤ 0", "「相當」順序某一步通過",
             "「相當」順序未通過、未輪到或資料不完整", "猜 T* = 1、直線、已知平均", "對猜 T* = 1 與已知平均（任何結果）：猜 T* = 1",
             "對猜 T* = 1 與已知平均（任何結果）：已知平均 1.0089", "逐家族移除：拿掉某一家族後標準 2 不再通過",
             "標準 2：p ≤ 0.025 但全體 r ≤ 0", "公式對變體 A（只報告，不檢定）", "任何結果"]
    missing = [row for row in table if not any(s.startswith(row) for s in seen)]
    assert not missing, missing
    assert any(s.startswith("「優於」順序任何一步未通過或未輪到：直線（未檢定）") for s in seen)


def test_prediction_word_needs_both_gates(good):
    r = copy.deepcopy(good)
    r["chain_A"][0].update(status="通過", passed=True); r["chain_A"][1].update(status="未通過", passed=False)
    r["prediction_claim_allowed"] = bool(r["success"] and r["chain_A"][0]["passed"] and r["chain_A"][1]["passed"])
    assert not r["prediction_claim_allowed"] and "不可使用" in e12.claims(r)[0]


def test_outputs_are_byte_identical(tmp_path):
    for k in (1, 2):
        e12.write_outputs(run(S.standard()), tmp_path / f"o{k}")
    for n in ("per_model.csv", "per_family.csv", "criteria.json", "claims.md"):
        assert (tmp_path / "o1" / n).read_bytes() == (tmp_path / "o2" / n).read_bytes(), n


# ---------------------------------------------------------------- reading notebook-15 files
def test_files_round_trip_and_hash_checks(tmp_path):
    inp = S.build(S.standard(m9=dict(written=False)))
    reg = S.write_files(inp, tmp_path)
    loaded = e12.load_inputs(tmp_path, reg)
    assert [m["name"] for m in loaded["models"]] == [m["name"] for m in inp["models"]]
    assert loaded["ref_sizes"] == e12.REF_SIZES
    res = e12.analyze(loaded, **FAST)
    assert {r["model"]: r["prediction_written"] for r in res["per_model"]}["m10"] is False
    out = tmp_path / e12.OUT_NAME
    # a damaged score file makes that model state 3 with the reason, it never stops the analysis
    (out / "scores_text_m2_T1.02.pkl").write_bytes(b"damaged")
    res = e12.analyze(e12.load_inputs(tmp_path, reg), **FAST)
    m2 = {r["model"]: r for r in res["per_model"]}["m2"]
    assert m2["state"] == 3 and "讀不出來" in m2["issues"] and not res["criterion1"]["passed"]
    # any change to a registered file stops everything
    p = out / "predictions.json"; good = p.read_bytes(); p.write_bytes(good.replace(b'"r1_commit"', b'"r1_commit" ', 1))
    with pytest.raises(e12.InputError, match="R2"):
        e12.load_inputs(tmp_path, reg)
    p.write_bytes(good)
    man = tmp_path / e12.MANIFEST; m0 = man.read_bytes(); man.write_bytes(m0 + b"\n")
    with pytest.raises(e12.InputError, match="R0"):
        e12.load_inputs(tmp_path, reg)
    man.write_bytes(m0)
    cfg = json.load(open(out / "run_config.json")); cfg["E_ids"] = [i for i in cfg["E_ids"] if i >= 300]   # the 700 reference texts only
    json.dump(cfg, open(out / "run_config.json", "w"))
    with pytest.raises(e12.InputError):
        e12.load_inputs(tmp_path, reg)


# ---------------------------------------------------------------- review B6 / A8: the bootstrap's sharing rules and the checks
def test_verify_prediction_stops_on_any_change():
    inp = S.build(S.standard())
    for path, delta in (("T_hat", 1e-9), ("T_hat_E", 1e-9), ("T_A", 1e-9), ("baselines.slope", 1e-9), ("baselines.line", 1e-9)):
        bad = copy.deepcopy(inp); p = bad["models"][3]["pred"]
        if "." in path:
            a, b = path.split("."); p[a][b] += delta
        else:
            p[path] += delta
        with pytest.raises(e12.InputError):
            e12.analyze(bad, n_boot=2, n_perm=10, n_perm_boot=10)
    bad = copy.deepcopy(inp); bad["models"][3]["pred"]["variant_B"]["value"] += 1e-9
    with pytest.raises(e12.InputError):
        e12.analyze(bad, n_boot=2, n_perm=10, n_perm_boot=10)


def test_every_model_shares_the_same_document_draws():
    """Two models with identical scores must get identical bootstrap results: the counts are drawn once per round."""
    inp = S.build(S.standard())
    twin = copy.deepcopy(inp["models"][0]); twin.update(name="twin", family="TwinFamily", no=13)
    inp["models"].append(twin)
    pm = {r["model"]: r for r in e12.analyze(inp, **FAST)["per_model"]}
    for key in ("T_star_p2_5", "T_star_p97_5", "T_hat_p2_5", "T_hat_p97_5", "share_state1"):
        assert pm["m1"][key] == pm["twin"][key], key


def test_round_weights_strata_and_prompt_weights():
    inp = S.build(S.standard()); pre = e12.prepare(inp)
    sizes = sorted(len(s) for s in pre["strataE"])
    assert len(pre["strataE"]) == 18 and len(pre["strataP"]) == 6        # E: source x (A half, B half, reference)
    assert sizes.count(25) == 12 and sorted(set(sizes)) == [25, 116, 117]
    rng = np.random.default_rng(0)
    for _ in range(20):
        wP, wE, wM = e12.round_weights(pre, rng)
        assert all(wE[s].sum() == len(s) for s in pre["strataE"]) and all(wP[s].sum() == len(s) for s in pre["strataP"])
        assert np.array_equal(wM, wE[pre["prompt_pos"]])               # a prompt's machine texts carry its E count
    m = inp["models"][0]; wE = np.ones(pre["nE"]); wE[pre["prompt_pos"][7]] = 3.0; wM = wE[pre["prompt_pos"]]
    v = e12.round_values([m], pre, np.ones(pre["nP"]), wE, wM)[0]
    for ti in range(len(T)):
        assert abs(v["auc_main"][ti] - e12.wauc(m["m_text"][ti], wM, m["hE"]["fdg"], wE)) < 1e-12


def test_direct_estimate_mae_on_case_one_models():
    res = run(S.standard(m6=dict(pattern="flatA_aboveB")))
    pm = [r for r in res["per_model"] if r["direct_case"] == 1]
    d = res["direct_mae"]
    assert d["n_models"] == len(pm) == 11 and d["n_excluded"] == 1
    assert d["formula"] == pytest.approx(np.mean([abs(r["T_star_B"] - r["T_hat"]) for r in pm]))
    assert d["direct"] == pytest.approx(np.mean([abs(r["T_star_B"] - r["T_direct"]) for r in pm]))


def test_stage2_deadline_and_rerun_limit_make_state_three(tmp_path):
    inp = S.build(S.standard())
    late = "2026-11-01T00:00:00+00:00"                               # 17 days after STARTED
    att = {"stage2": {"m2": [dict(start="2026-10-16T00:00:00+00:00", ok=False, error="x")] * 4 + [dict(start="2026-10-19T00:00:00+00:00", ok=True)]}}
    reg = S.write_files(inp, tmp_path / "a", attempts=att)
    pm = {r["model"]: r for r in e12.analyze(e12.load_inputs(tmp_path / "a", reg), **FAST)["per_model"]}
    assert pm["m2"]["state"] == 3 and "超過重跑上限" in pm["m2"]["issues"] and pm["m1"]["state"] == 1
    reg = S.write_files(inp, tmp_path / "b", written=late)
    res = e12.analyze(e12.load_inputs(tmp_path / "b", reg), **FAST)
    assert all(r["state"] == 3 and "超過 14 天期限" in r["issues"] for r in res["per_model"])
    reg = S.write_files(inp, tmp_path / "c", started="2026-10-13T00:00:00+00:00")   # stage 2 began before R2's block time
    with pytest.raises(e12.InputError, match="區塊時間"):
        e12.load_inputs(tmp_path / "c", reg)


def test_every_round_passes_the_prompt_weights_to_the_models(monkeypatch):
    """Review B6 (mutation M2b): the call site must hand each round's prompt weights (wM = wE at the prompts) to the models."""
    seen, real = [], e12.round_values

    def spy(models, pre, wP, wE, wM):
        seen.append(bool(np.array_equal(wM, wE[pre["prompt_pos"]])))
        return real(models, pre, wP, wE, wM)
    monkeypatch.setattr(e12, "round_values", spy)
    e12.analyze(S.build(S.standard()), n_boot=5, n_perm=10, n_perm_boot=10)
    assert seen == [True] * 6                                  # the observed data plus five bootstrap rounds



def test_fourteen_day_limit_boundary(tmp_path):
    """Review B6: one stage-2 file an hour inside the 14 days keeps state 1; an hour beyond makes state 3."""
    inp = S.build(S.standard())
    for label, offset, state in (("in", "2026-10-28T23:00:00+00:00", "1"), ("out", "2026-10-29T01:00:00+00:00", "3")):
        reg = S.write_files(inp, tmp_path / label, written=offset)            # STARTED is 2026-10-15T00:00:00+00:00
        pm = {r["model"]: r for r in e12.analyze(e12.load_inputs(tmp_path / label, reg), **FAST)["per_model"]}
        assert {str(r["state"]) for r in pm.values()} == {state}, label


def test_mixed_environments_in_stage_two_make_state_three(tmp_path):
    """Plan 5.4: one model's stage-2 files must come from one Colab environment; models are listed with their environment."""
    inp = S.build(S.standard())
    new = dict(S.ENV, torch="2.9.0+cu126")
    reg = S.write_files(inp, tmp_path, env_of=lambda name, t: new if (name == "m4" and t > 1.05) else S.ENV)
    pm = {r["model"]: r for r in e12.analyze(e12.load_inputs(tmp_path, reg), **FAST)["per_model"]}
    assert pm["m4"]["state"] == 3 and "執行環境不一致" in pm["m4"]["issues"]
    assert pm["m1"]["state"] == 1 and "torch 2.8.0+cu126" in pm["m1"]["environment_stage2"] and pm["m1"]["environment_stage1"]
