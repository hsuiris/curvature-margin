# Pre-registered analysis of the second replication test (design doc 8.8.24, plan version 5.1; notebook 15).
# Question: does the unfitted formula T_hat = 1 - E_P[c] / V_P (human text only, P group) predict the temperature at which
# each model's self-scored Fast-DetectGPT AUROC falls to 0.5, on 12 new base models from 11 families?
#   Criterion 1 (accuracy): every model in state 1, each |T* - T_hat| <= 0.02, family-weighted MAE <= 0.005.
#   Criterion 2 (form R): permutation test of the family-level Pearson r (single-model families permuted, multi-model
#     families fixed, 100,000 permutations, seed 20260930); passes if p <= 0.025 and the all-family r > 0.
#   Chain A ("better than", 1.25%): slope -> direct estimate -> line. Each step passes if the 1.25th percentile of the
#     family-mean improvement d-bar over the crossed family x document bootstrap is above 0 (plan 5.1).
#   Chain B ("equivalent", 1.25% per side): slope -> direct estimate. TOST with the same bootstrap: the 97.5% percentile
#     interval must lie inside (-0.002, +0.002).
#   Guess T* = 1 and the known mean 1.0089 are described only (plan 5.1): K, d-bar and its interval, no pass/fail.
# Missing values never help the formula: state 2/3 endpoints take the worst-case bound in descriptive improvements; the
# confirmatory bootstrap steps need complete original data and score a bootstrap round with any missing value as -inf
# (lower bound) / +inf (upper bound).
# Usage: python scripts/e12_replication_test.py [--data DIR] [--registration FILE] [--out DIR] [--dryrun]
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import argparse, hashlib, json, math, pickle
from datetime import datetime, timedelta
import numpy as np, pandas as pd
from scipy.optimize import brentq
from scipy.special import ndtr
from cmargin.crossing import crossings_down, endpoint_state, direct_estimate, extrapolate

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCHEMA = "nb15/v1"
OUT_NAME = "15_replication_test"
MANIFEST = "manifests/m4gt_humans_PE_manifest.csv"

# ---- registered constants (plan 8.8.24) ----
TEMPS = (0.94, 0.97, 0.99, 1.00, 1.01, 1.02, 1.03, 1.04, 1.06, 1.09, 1.14)
SOURCES = ("arxiv", "outfox", "peerread", "reddit", "wikihow", "wikipedia")
QUOTA = {"arxiv": (167, 167, 25, 25), "outfox": (167, 167, 25, 25), "peerread": (167, 167, 25, 25),
         "reddit": (167, 167, 25, 25), "wikihow": (166, 166, 25, 25), "wikipedia": (166, 166, 25, 25)}   # (P, E, A half, B half)
# Human references (plan "直接估計" and "終點"): the main endpoint uses the WHOLE E group (1,000 texts, prompts included;
# not role == "reference", which is only 700), T*_B uses E minus the A half (850), the direct estimate uses P (1,000).
REF_SIZES = {"main": 1000, "B": 850, "direct": 1000}
MAX_ATTEMPTS, STAGE2_DAYS = 4, 14          # first run plus three reruns per model and stage; stage 2 within 14 days of its start
TOL_EACH, TOL_MAE = 0.02, 0.005
ALPHA_C2, ALPHA_CHAIN, EQ_MARGIN = 0.025, 0.0125, 0.002
N_PERM, SEED_PERM = 100_000, 20260930
N_BOOT, SEED_DOC, SEED_FAM = 2000, 20260932, 20260931
N_PERM_BOOT, SEED_PERM_BOOT = 10_000, 20260933
GUESS1, KNOWN_MEAN, SLOPE_K, LINE_A, LINE_K = 1.0, 1.0089, 0.1451, 1.0004, 0.1409
RETOK_SHIFT = 0.0010                                             # notebook 14; byte-level BPE models only
BYTE_BPE = ("togethercomputer/RedPajama-INCITE-Base-3B-v1", "stabilityai/stablelm-3b-4e1t", "state-spaces/mamba-1.4b-hf",
            "PleIAs/Pleias-1.2b-Preview", "meta-llama/Llama-3.2-1B")
VARIANT_B_RANGE = (0.80, 1.30)
BASELINES = ("guess1", "known_mean", "slope", "line")
CHAIN_A = ("slope", "direct", "line")          # plan 5.1: the known mean left the chain
CHAIN_B = ("slope", "direct")
LABEL = {"guess1": "猜 T* = 1", "known_mean": "已知平均 1.0089", "slope": "過原點斜率", "line": "直線",
         "direct": "直接估計"}


# ---------------------------------------------------------------- shared formulas (same code as notebook 15 cell 5)
def sha256_file(path):
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            d.update(block)
    return d.hexdigest()


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def text_stats(arr):
    """Per-text curvature c = mean(H - s), V = mean Var, Fast-DetectGPT score and sqrt(sum Var), in float64."""
    a = np.asarray(arr, dtype=np.float64)
    g = a[1] - a[0]
    sv = math.sqrt(a[2].sum())
    return g.mean(), a[2].mean(), g.sum() / sv, sv


def formula(c, V):
    return float(1 - np.mean(c) / np.mean(V))


def baseline_values(Ec):
    return {"guess1": GUESS1, "known_mean": KNOWN_MEAN, "slope": 1 - SLOPE_K * Ec, "line": LINE_A - LINE_K * Ec}


def variant_a(fdg, sv):
    return float(1 - np.mean(fdg) / np.mean(sv))


def variant_b(fdg, sv):
    """AUROC(T) = mean_ij Phi(-(T - 1) sv_i - fdg_j) = 0.5 on P humans; brentq on [0.80, 1.30], xtol 1e-6."""
    fdg, sv = np.asarray(fdg, float), np.asarray(sv, float)
    if not (np.isfinite(fdg).all() and np.isfinite(sv).all()):
        return None, "無法計算"
    f = lambda T: float(ndtr(-(T - 1) * sv[:, None] - fdg[None, :]).mean() - 0.5)
    lo, hi = VARIANT_B_RANGE
    flo, fhi = f(lo), f(hi)
    if flo < 0 and fhi < 0:
        return None, "無解（< 0.80）"
    if flo > 0 and fhi > 0:
        return None, "無解（> 1.30）"
    return float(brentq(f, lo, hi, xtol=1e-6)), "有解"


# ---------------------------------------------------------------- statistics helpers
def quantile_inf(x, q):
    """numpy.quantile's default (linear) rule, extended to +-inf: an infinite neighbour makes the result infinite."""
    x = np.sort(np.asarray(x, dtype=float))
    assert not np.isnan(x).any(), "missing rounds must be +-inf, not NaN"
    if np.isfinite(x).all():
        return float(np.quantile(x, q))
    h = (len(x) - 1) * q; i = int(math.floor(h)); f = h - i
    a, b = x[i], x[min(i + 1, len(x) - 1)]
    if f == 0 or a == b:
        return float(a)
    if a == -np.inf:
        return -math.inf
    if b == np.inf:
        return math.inf
    return float(a + (b - a) * f)


def wauc(g, wg, h, wh):
    """Weighted P(machine > human) + 0.5 P(tie); equals sklearn's roc_auc_score when every weight is 1."""
    g, wg, h, wh = (np.asarray(v, float) for v in (g, wg, h, wh))
    o = np.argsort(h, kind="stable"); hs, cw = h[o], np.r_[0.0, np.cumsum(wh[o])]
    lo, hi = np.searchsorted(hs, g, "left"), np.searchsorted(hs, g, "right")
    return float((wg * (cw[lo] + 0.5 * (cw[hi] - cw[lo]))).sum() / (wg.sum() * cw[-1]))


class AucSet:
    """Machine scores (temperatures x texts) against one human reference; only the weights change between rounds."""

    def __init__(self, g, h):
        self.o = np.argsort(h, kind="stable"); hs = np.asarray(h, float)[self.o]
        self.lo = np.searchsorted(hs, g, "left"); self.hi = np.searchsorted(hs, g, "right")

    def auc(self, wg, wh):
        cw = np.r_[0.0, np.cumsum(wh[self.o])]
        num = (wg * (cw[self.lo] + 0.5 * (cw[self.hi] - cw[self.lo]))).sum(-1)
        return num / (wg.sum() * cw[-1])


def form_r(x, y, fixed, n_perm, rng):
    """Family-level Pearson r and its permutation p value (single-model families permuted, the rest fixed)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    F = len(x)
    if F < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return math.nan, 1.0
    xc, yc = x - x.mean(), y - y.mean()
    r = float((xc * yc).sum() / math.sqrt((xc ** 2).sum() * (yc ** 2).sum()))
    mov = np.flatnonzero(~np.asarray(fixed, bool))
    P = np.tile(np.arange(F), (n_perm, 1))
    if len(mov) > 1:
        P[:, mov] = mov[np.argsort(rng.random((n_perm, len(mov))), axis=1)]
    s = (xc[P] * yc).sum(1); s0 = (xc * yc).sum()
    tol = 1e-12 * float(np.abs(xc).sum() * np.abs(yc).max())   # identical orderings must count as ties, not rounding noise
    return r, float((1 + (s >= s0 - tol).sum()) / (n_perm + 1))


def c2_pass(r, p):
    return bool(p <= ALPHA_C2 and not math.isnan(r) and r > 0)


# ---------------------------------------------------------------- per-model quantities for one set of weights
def model_values(m, temps, wP, wE, wM, halves):
    """Every per-model quantity the analysis needs, for one set of document weights (all ones = observed data)."""
    isA, isB, notA = halves
    none = (3, None, "資料不足")
    v = {"T_hat": None, "b": {}, "T_A": None, "main": none, "ids": none, "B": none,
         "direct": (None, None), "auc_main": None, "auc_ids": None, "auc_B": None, "auc_A": None}
    hP = m.get("hP")
    if hP is not None:
        sw = wP.sum()
        Ec, V = (wP * hP["c"]).sum() / sw, (wP * hP["V"]).sum() / sw
        if m.get("pred") is not None:
            v["T_hat"] = float(1 - Ec / V)
        v["b"] = baseline_values(float(Ec))
        v["T_A"] = float(1 - (wP * hP["fdg"]).sum() / (wP * hP["sv"]).sum())
    if m.get("sets") is None:
        return v
    S = m["sets"]
    if "text" in S:
        a = S["text"].auc(wM, wE); v["auc_main"] = a; v["main"] = endpoint_state(temps, a)
        a = S["text"].auc(wM * isB, wE * notA); v["auc_B"] = a; v["B"] = endpoint_state(temps, a)
        a = S["direct"].auc(wM * isA, wP); v["auc_A"] = a; v["direct"] = direct_estimate(temps, a)
    if "ids" in S:
        a = S["ids"].auc(wM, wE); v["auc_ids"] = a; v["ids"] = endpoint_state(temps, a)
    return v


def improvement(T, state, T_hat, b):
    """d(b) = |T* - b| - |T* - T_hat|; states 2 and 3 take the worst case -|T_hat - b|; None if there is no T_hat."""
    if T_hat is None:
        return None, "預測未寫出"
    if state == 1:
        return abs(T - b) - abs(T - T_hat), "實測"
    return -abs(T_hat - b), "最壞情境界限"


def direct_case(T_hat, T_d, B_state, T_B):
    """The four cases of the direct-estimate comparison; returns (case, d). case 0 = no prediction."""
    if T_hat is None:
        return 0, None
    if T_d is not None and B_state == 1:
        return 1, abs(T_B - T_d) - abs(T_B - T_hat)
    if T_d is not None:
        return 2, -abs(T_hat - T_d)
    if B_state == 1:
        return 3, -abs(T_B - T_hat)
    return 4, 0.0


def family_value(ds, flags):
    """Family improvement from its models' d (None = no prediction). Any model without a prediction, or any direct
    comparison in case 4, caps the family at min(mean, 0): it never counts as a formula win. None if nothing is left."""
    have = [d for d in ds if d is not None]
    if not have:
        return None
    f = float(np.mean(have))
    return min(f, 0.0) if (len(have) < len(ds) or any(flags)) else f


# ---------------------------------------------------------------- loading
class InputError(Exception):
    pass


def need(cond, msg):
    if not cond:
        raise InputError(msg)


def read_score_file(path, kind, path_kind, entry, temp, reg_man_sha, r1_commit, gen_rows=None):
    """Validate one notebook-15 score file and reduce it to per-text statistics; raises InputError on any problem."""
    need(path.exists(), f"{path.name}: 檔案不存在")
    try:
        obj = pickle.load(open(path, "rb"))
    except Exception as e:                                  # noqa: BLE001 - any unreadable file is a data problem
        raise InputError(f"{path.name}: 讀不出來（{type(e).__name__}）")
    cfg = obj.get("config", {})
    need(obj.get("schema") == SCHEMA and obj.get("kind") == kind and obj.get("path") == path_kind, f"{path.name}: 格式不符")
    need(obj.get("config_hash") == config_hash(cfg), f"{path.name}: 設定雜湊和內容不符")
    for key, want in (("model", entry["name"]), ("revision", entry["revision"]), ("start", entry["start"]),
                      ("banned_sha256", entry["banned_sha256"]), ("pad", entry["pad"]), ("temperature", temp),
                      ("manifest_sha256", reg_man_sha), ("r1_commit", r1_commit)):
        need(cfg.get(key) == want, f"{path.name}: 設定的 {key} 和 R1 登錄不符")
    rows = obj["rows"]; ids = [r["id"] for r in rows]
    need(len(ids) == len(set(ids)), f"{path.name}: 有重複的編號")
    out = {}
    for r in rows:
        arr = np.asarray(r["arr"]); x = np.asarray(r["ids"])
        need(arr.shape == (4, len(x) - r["k"]) and len(x) > r["k"], f"{path.name}: 第 {r['id']} 篇的長度和遮罩不符")
        need(np.isfinite(arr).all(), f"{path.name}: 第 {r['id']} 篇有非有限值")
        if gen_rows is not None:
            g = gen_rows.get(r["id"])
            need(g is not None and x.tolist() == entry["start"] + g["prompt"] + g["ids"]
                 and r["k"] == len(entry["start"]) + len(g["prompt"]), f"{path.name}: 第 {r['id']} 篇的編號路徑輸入和生成檔不符")
        c, V, fdg, sv = text_stats(arr)
        out[r["id"]] = dict(c=c, V=V, fdg=fdg, sv=sv, j=r["j"], len_p=r["len_p"], cut=r["cut"],
                            same_prefix=r["same_prefix"], same_cont=r["same_cont"], n=arr.shape[1])
    if kind == "machine":
        need("written_utc" in obj, f"{path.name}: 沒有寫入時間")
        out["__written_utc__"] = obj["written_utc"]
    out["__environment__"] = env_text(obj.get("environment"))
    return out


def read_gen_file(path, entry, temp, reg_man_sha, r1_commit):
    need(path.exists(), f"{path.name}: 檔案不存在")
    try:
        obj = json.load(open(path))
    except Exception as e:                                  # noqa: BLE001
        raise InputError(f"{path.name}: 讀不出來（{type(e).__name__}）")
    cfg = obj.get("config", {})
    need(obj.get("schema") == SCHEMA and obj.get("kind") == "gen", f"{path.name}: 格式不符")
    need(obj.get("config_hash") == config_hash(cfg), f"{path.name}: 設定雜湊和內容不符")
    for key, want in (("model", entry["name"]), ("revision", entry["revision"]), ("temperature", temp),
                      ("manifest_sha256", reg_man_sha), ("r1_commit", r1_commit)):
        need(cfg.get(key) == want, f"{path.name}: 設定的 {key} 和 R1 登錄不符")
    rows = {r["id"]: r for r in obj["rows"]}
    need(len(rows) == len(obj["rows"]), f"{path.name}: 有重複的編號")
    banned = set(entry["banned"])
    for r in rows.values():
        need(len(r["ids"]) == r["target"], f"{path.name}: 第 {r['id']} 篇的生成長度不等於目標")
        need(not (set(r["ids"]) & banned), f"{path.name}: 第 {r['id']} 篇含禁止清單裡的編號")
    need("written_utc" in obj, f"{path.name}: 沒有寫入時間")
    return rows, obj["written_utc"], env_text(obj.get("environment"))


def env_text(env):
    """The Colab environment a file was made in (plan 5.4: models made before and after a change are listed separately)."""
    return "; ".join(f"{k} {env[k]}" for k in sorted(env)) if isinstance(env, dict) else "未記錄"


def when(text):
    t = datetime.fromisoformat(str(text))
    need(t.tzinfo is not None, f"{text}: 時間要帶時區")
    return t


def stage2_rule(inp, name, written):
    """Plan ("GPU 預算與執行方式"): stage 2 has to finish within 14 days of its start, and a model gets the first run plus at
    most three reruns. A model whose last stage-2 file was written later, or that failed four times before finishing, is
    state 3. Returns the reason, or None."""
    if not written:
        return None
    done = max(when(t) for t in written)
    if done > inp["stage2_started"] + timedelta(days=STAGE2_DAYS):
        return f"超過 14 天期限：第二階段 {inp['stage2_started'].isoformat()} 開始，這個模型的最後一個檔案 {done.isoformat()} 才寫入"
    failed = [a for a in inp["attempts"].get("stage2", {}).get(name, []) if not a.get("ok") and when(a["start"]) < done]
    if len(failed) >= MAX_ATTEMPTS:
        return f"超過重跑上限：完成前已失敗 {len(failed)} 次"
    return None


def roundtrip_counts(rows):
    """Re-encoding changes (prefix and continuation separately), split points moved back, and 512-token truncations."""
    return dict(n=len(rows), prefix_changed=int(sum(not r["same_prefix"] for r in rows)),
                continuation_changed=int(sum(not r["same_cont"] for r in rows)),
                j_back=int(sum(r["j"] < r["len_p"] for r in rows)), truncated=int(sum(r["cut"] > 0 for r in rows)))


def tag_of(name):
    return name.split("/")[-1]


def fmt_t(t):
    return f"{t:.2f}"


def load_inputs(data_dir, registration, dryrun=False):
    """Read and verify every input. Problems with one model's stage-2 files become state 3 ('資料不足'); problems with the
    registered files (registration, stage2_go, predictions, r1_models, manifest) stop the analysis."""
    data_dir = pathlib.Path(data_dir); out = data_dir / OUT_NAME
    reg = json.load(open(registration))
    for sec in ("R0", "R1", "R2"):
        need(sec in reg, f"registration.json 缺少 {sec} 段落")
    go = json.load(open(out / "stage2_go.json"))
    pred_sha = sha256_file(out / "predictions.json")
    need(pred_sha == reg["R2"]["predictions_sha256"], "predictions.json 的 SHA-256 和 R2 登錄值不符")
    need(pred_sha == go["predictions_sha256"], "predictions.json 的 SHA-256 和 stage2_go.json 不符")
    man_path = data_dir / MANIFEST
    man_sha = sha256_file(man_path)
    need(man_sha == reg["R0"]["pe_manifest_sha256"], "P／E manifest 的 SHA-256 和 R0 登錄值不符")
    r1 = json.load(open(out / "r1_models.json"))
    need(r1["models"] == reg["R1"]["models"], "r1_models.json 的模型表和 R1 登錄不符")
    preds = json.load(open(out / "predictions.json"))
    run = json.load(open(out / "run_config.json"))
    man = pd.read_csv(man_path)
    need(man.id.is_unique, "manifest 的 id 有重複")
    temps = tuple(run["temps"])
    if not dryrun:
        need(temps == TEMPS and not run["dryrun"], "run_config.json 的溫度網格或模式和登錄不符")
        full = {g: sorted(man.id[man.group == g].tolist()) for g in ("P", "E")}
        need(sorted(run["P_ids"]) == full["P"] and sorted(run["E_ids"]) == full["E"], "run_config.json 的 P／E 和 manifest 不符")
        need(sorted(run["A_ids"]) == sorted(man.id[man.role == "prompt_A"].tolist())
             and sorted(run["B_ids"]) == sorted(man.id[man.role == "prompt_B"].tolist()), "提示的 A／B 半和 manifest 不符")
    mi = man.set_index("id")
    P_ids, E_ids = list(run["P_ids"]), list(run["E_ids"])
    nq = [sum(v[i] for v in run["quota"].values()) for i in range(4)]
    need([len(P_ids), len(E_ids), len(run["A_ids"]), len(run["B_ids"])] == nq, "P、E、A 半、B 半的篇數和配額不符")
    need(set(man.group[man.id.isin(E_ids)]) == {"E"} and set(man.group[man.id.isin(P_ids)]) == {"P"}, "P／E 的組別和 manifest 不符")
    ref = {"main": len(E_ids), "B": len(set(E_ids) - set(run["A_ids"])), "direct": len(P_ids)}
    if not dryrun:
        need(run["quota"] == {k: list(v) for k, v in QUOTA.items()}, "run_config.json 的配額和登錄不符")
        need(ref == REF_SIZES, f"人類參考篇數 {ref} 和計畫不符（主要終點要用整個 E 組）")
    prompts = sorted(run["A_ids"] + run["B_ids"])
    need(set(prompts) <= set(E_ids), "提示必須屬於 E 組")
    need((out / "stage2_started.json").exists(), "缺少 stage2_started.json（第二階段的開始時間）")
    started = when(json.load(open(out / "stage2_started.json"))["utc"])
    if not dryrun:
        need(started > when(go["ots_block_time"]), "第二階段在 R2 時間證明的區塊時間之前就開始了")
    attempts = json.load(open(out / "attempts.json")) if (out / "attempts.json").exists() else {}
    inp = dict(temps=np.array(temps), P_ids=np.array(P_ids), E_ids=np.array(E_ids), prompts=np.array(prompts),
               P_src=mi.loc[P_ids, "source"].to_numpy(), E_src=mi.loc[E_ids, "source"].to_numpy(),
               E_role=mi.loc[E_ids, "role"].to_numpy(), A_ids=set(run["A_ids"]), ref_sizes=ref, models=[],
               r1_commit=r1["r1_commit"], dryrun=bool(run["dryrun"]), stage2_started=started, attempts=attempts)
    need(set(preds["models"]) == {e["name"] for e in r1["models"]}, "predictions.json 的模型和 R1 名單不符")
    for e in r1["models"]:
        inp["models"].append(load_model(out, e, preds["models"][e["name"]], inp, man_sha))
    return inp


def load_model(out, e, pred, inp, man_sha):
    tag = tag_of(e["name"])
    m = dict(e, tag=tag, pred=None, issues=[], hP=None, hE=None, m_text=None, m_ids=None, meta={})
    if pred.get("status") != "written":
        m["issues"].append(f"預測未寫出：{pred.get('reason', '')}")
    else:
        m["pred"] = pred
    try:
        h = read_score_file(out / f"humans_{tag}.pkl", "human", "text", e, None, man_sha, inp["r1_commit"])
        m["meta"]["environment_stage1"] = h.pop("__environment__")
        need(set(h) == set(inp["P_ids"]) | set(inp["E_ids"]), f"humans_{tag}.pkl: 篇數和 P／E 不符")
        for g, ids in (("hP", inp["P_ids"]), ("hE", inp["E_ids"])):
            m[g] = {k: np.array([h[i][k] for i in ids], float) for k in ("c", "V", "fdg", "sv")}
        m["meta"]["human_roundtrip"] = {g: roundtrip_counts([h[i] for i in ids])
                                        for g, ids in (("P", inp["P_ids"]), ("E", inp["E_ids"]))}
    except InputError as err:
        if m["pred"] is not None:     # a written prediction must still have its human scores: the registration is broken
            raise InputError(f"{e['name']}: 第一階段的人類分數讀不到或不完整（{err}）")
        m["issues"].append(f"人類分數：{err}")
    if m["pred"] is None:
        return m
    nT, pr = len(inp["temps"]), inp["prompts"]
    written, envs = [], set()
    for path_kind in ("text", "ids"):
        M = np.full((nT, len(pr)), np.nan); meta = []
        try:
            for ti, t in enumerate(inp["temps"]):
                gen, gw, genv = read_gen_file(out / f"gen_{tag}_T{fmt_t(t)}.json", e, float(t), man_sha, inp["r1_commit"])
                need(sorted(gen) == pr.tolist(), f"gen_{tag}_T{fmt_t(t)}.json: 提示和預期不符")
                s = read_score_file(out / f"scores_{path_kind}_{tag}_T{fmt_t(t)}.pkl", "machine", path_kind, e, float(t),
                                    man_sha, inp["r1_commit"], gen_rows=gen if path_kind == "ids" else None)
                written += [gw, s.pop("__written_utc__")]; envs |= {genv, s.pop("__environment__")}
                need(sorted(s) == pr.tolist(), f"scores_{path_kind}_{tag}_T{fmt_t(t)}.pkl: 篇數和提示不符")
                M[ti] = [s[i]["fdg"] for i in pr]
                meta.append(dict(T=float(t), c_mean=float(np.mean([s[i]["c"] for i in pr])), **roundtrip_counts([s[i] for i in pr])))
            m["m_" + path_kind] = M
            m["meta"]["machine_" + path_kind] = meta
        except InputError as err:
            m["issues"].append(f"{'文字' if path_kind == 'text' else '編號'}路徑：{err}")
    m["meta"]["environment_stage2"] = " | ".join(sorted(envs))
    rule = stage2_rule(inp, e["name"], written) or (f"第二階段檔案的執行環境不一致：{' | '.join(sorted(envs))}" if len(envs) > 1 else None)
    if rule:                       # state 3 for both endpoints; the files themselves are fine
        m["m_text"] = m["m_ids"] = None
        m["issues"].append(rule)
    return m


# ---------------------------------------------------------------- analysis
def prepare(inp):
    """Index sets, strata and per-model AUROC structures."""
    temps = inp["temps"]; E_ids = inp["E_ids"]
    pos_E = {i: k for k, i in enumerate(E_ids)}
    prompt_pos = np.array([pos_E[i] for i in inp["prompts"]])
    isA = np.array([i in inp["A_ids"] for i in inp["prompts"]], float)
    notA = np.ones(len(E_ids)); notA[[pos_E[i] for i in inp["A_ids"]]] = 0.0
    strata = [np.flatnonzero(inp["P_src"] == s) for s in SOURCES if (inp["P_src"] == s).any()]
    stE = [np.flatnonzero((inp["E_src"] == s) & (inp["E_role"] == r)) for s in SOURCES
           for r in ("prompt_A", "prompt_B", "reference")]
    stE = [s for s in stE if len(s)]
    for m in inp["models"]:
        sets = {}
        if m["hE"] is not None and m["m_text"] is not None and np.isfinite(m["m_text"]).all():
            sets["text"] = AucSet(m["m_text"], m["hE"]["fdg"]); sets["direct"] = AucSet(m["m_text"], m["hP"]["fdg"])
        if m["hE"] is not None and m["m_ids"] is not None and np.isfinite(m["m_ids"]).all():
            sets["ids"] = AucSet(m["m_ids"], m["hE"]["fdg"])
        m["sets"] = sets or None
    fams = sorted({m["family"] for m in inp["models"]})
    fam_of = np.array([fams.index(m["family"]) for m in inp["models"]])
    return dict(temps=temps, prompt_pos=prompt_pos, halves=(isA, 1.0 - isA, notA), strataP=strata, strataE=stE,
                nP=len(inp["P_ids"]), nE=len(E_ids), fams=fams, fam_of=fam_of,
                fixed=np.bincount(fam_of, minlength=len(fams)) > 1)


def draw(rng, strata, n):
    w = np.zeros(n)
    for s in strata:
        w += np.bincount(s[rng.integers(0, len(s), len(s))], minlength=n)
    return w


def round_weights(pre, rng):
    """One bootstrap round: document counts drawn within the P strata (source) and the E strata (source x role). The same
    counts serve every model, temperature, path, prediction and endpoint; a prompt's machine texts carry its E count."""
    wP = draw(rng, pre["strataP"], pre["nP"]); wE = draw(rng, pre["strataE"], pre["nE"])
    return wP, wE, wE[pre["prompt_pos"]]


def round_values(models, pre, wP, wE, wM):
    return [model_values(m, pre["temps"], wP, wE, wM, pre["halves"]) for m in models]


def family_mean(x, fam_of, F):
    return np.bincount(fam_of, np.asarray(x, float), F) / np.bincount(fam_of, minlength=F)


def compare_values(vals):
    """Per-model improvement for every comparison, with the plan's missing-value rules (descriptive level)."""
    out = {}
    for k in BASELINES:
        out[k] = [improvement(v["main"][1], v["main"][0], v["T_hat"], v["b"].get(k)) if v["b"] else (None, "無人類分數")
                  for v in vals]
    out["direct"] = [direct_case(v["T_hat"], v["direct"][0], v["B"][0], v["B"][1]) for v in vals]
    return out


def complete(vals, key):
    """Complete data for a confirmatory step: every model has a prediction and a measured value (state 1 / case 1)."""
    if key == "direct":
        return all(direct_case(v["T_hat"], v["direct"][0], v["B"][0], v["B"][1])[0] == 1 for v in vals)
    return all(v["T_hat"] is not None and v["main"][0] == 1 for v in vals)


def dbar_complete(vals, key, fam_of, F, fam_idx):
    """Family-mean improvement on complete data (None if any model lacks a measured value)."""
    if not complete(vals, key):
        return None
    if key == "direct":
        d = [direct_case(v["T_hat"], v["direct"][0], v["B"][0], v["B"][1])[1] for v in vals]
    else:
        d = [abs(v["main"][1] - v["b"][key]) - abs(v["main"][1] - v["T_hat"]) for v in vals]
    fd = family_mean(d, fam_of, F)
    return float(fd[fam_idx].mean())


def describe_families(cmp, fam_of, F):
    """Family d_f with the missing-value rules; K counts d_f > 0; d-bar over families with a defined value."""
    res = {}
    for k, pairs in cmp.items():
        fd, excluded = [], []
        for f in range(F):
            members = [pairs[i] for i in np.flatnonzero(fam_of == f)]
            if k == "direct":
                ds = [d for _, d in members]; flags = [c == 4 for c, _ in members]
            else:
                ds = [d for d, _ in members]; flags = [False] * len(members)
            v = family_value(ds, flags)
            fd.append(v)
            if v is None or (k == "direct" and any(flags)):
                excluded.append(f)
        K = int(sum(1 for v in fd if v is not None and v > 0))
        keep = [v for f, v in enumerate(fd) if f not in excluded]
        res[k] = dict(d_f=fd, K=K, dbar=float(np.mean(keep)) if keep else None, excluded=excluded)
    return res


def run_chains(c2, obs, boot, fam_desc, F):
    """Chain A ("better than": lower 1.25% percentile of d-bar* > 0) and chain B (TOST: the 97.5% interval inside
    +-0.002). boot[k] holds d-bar* per round, NaN where any model lacked a measured value (-inf / +inf in the bounds).
    Both chains start only after criterion 2; a step that fails or lacks complete data stops its chain."""
    chains = {}
    for name, steps in (("A", CHAIN_A), ("B", CHAIN_B)):
        go, rows = c2, []
        for k in steps:
            x = np.asarray(boot[k], float)
            s = dict(comparison=k, dbar=fam_desc[k]["dbar"], K=fam_desc[k]["K"], F=F,
                     lower=quantile_inf(np.where(np.isnan(x), -np.inf, x), ALPHA_CHAIN),
                     complete=complete(obs, k), missing_rounds=int(np.isnan(x).sum()))
            if name == "B":
                s["upper"] = quantile_inf(np.where(np.isnan(x), np.inf, x), 1 - ALPHA_CHAIN)
            if not go:
                s["status"] = "未檢定"
            elif not s["complete"]:
                s["status"], s["reason"] = ("資料不完整（未通過）", "資料不完整") if name == "A" else ("未檢定", "資料不完整")
                go = False
            else:
                ok = s["lower"] > 0 if name == "A" else (s["lower"] > -EQ_MARGIN and s["upper"] < EQ_MARGIN)
                s["status"] = "通過" if ok else "未通過"; go = ok
            s["passed"] = s["status"] == "通過"
            rows.append(s)
        chains[name] = rows
    return chains


def verify_prediction(m, v):
    """Every registered number in predictions.json must be reproducible from the stored stage-1 human scores."""
    p = m["pred"]
    if p is None:
        return
    close = lambda a, b: a is not None and b is not None and abs(a - b) <= 1e-12
    need(close(v["T_hat"], p["T_hat"]), f"{m['name']}: predictions.json 的主要預測和人類分數重算不同")
    need(close(formula(m["hE"]["c"], m["hE"]["V"]), p["T_hat_E"]), f"{m['name']}: 次要預測 (a) 和重算不同")
    need(close(v["T_A"], p["T_A"]), f"{m['name']}: 變體 A 和重算不同")
    b_val, b_status = variant_b(m["hP"]["fdg"], m["hP"]["sv"])
    need(b_status == p["variant_B"]["status"] and (b_val is None or close(b_val, p["variant_B"]["value"])),
         f"{m['name']}: 變體 B 和重算不同")
    for k in BASELINES:
        need(close(v["b"][k], p["baselines"][k]), f"{m['name']}: 基準 {k} 和重算不同")


def analyze(inp, n_boot=N_BOOT, n_perm=N_PERM, n_perm_boot=N_PERM_BOOT):
    pre = prepare(inp)
    temps, F, fam_of, fixed = pre["temps"], len(pre["fams"]), pre["fam_of"], pre["fixed"]
    models = inp["models"]; nM = len(models)
    one_P, one_E = np.ones(pre["nP"]), np.ones(pre["nE"])
    obs = round_values(models, pre, one_P, one_E, one_E[pre["prompt_pos"]])
    for m, v in zip(models, obs):
        verify_prediction(m, v)
    cmp = compare_values(obs)
    fam_desc = describe_families(cmp, fam_of, F)
    res = dict(F=F, families=pre["fams"], n_models=nM, fixed_families=[pre["fams"][f] for f in np.flatnonzero(fixed)])

    # ---- criterion 1
    states = [v["main"][0] for v in obs]
    all1 = all(s == 1 for s in states) and all(v["T_hat"] is not None for v in obs)
    err = [abs(v["main"][1] - v["T_hat"]) if (v["main"][0] == 1 and v["T_hat"] is not None) else None for v in obs]
    fam_err = [float(np.mean([err[i] for i in np.flatnonzero(fam_of == f)])) if all1 else None for f in range(F)]
    mae = float(np.mean(fam_err)) if all1 else None
    c1 = bool(all1 and max(err) <= TOL_EACH and mae <= TOL_MAE)
    res["criterion1"] = dict(passed=c1, all_state1=all1, max_error=max(err) if all1 else None, family_mae=mae,
                             failures=[dict(model=m["name"], state=v["main"][0], reason=v["main"][2] if m["pred"] else "預測未寫出",
                                            issues=m["issues"]) for m, v in zip(models, obs) if v["main"][0] != 1 or v["T_hat"] is None],
                             over_0_02=[m["name"] for m, e in zip(models, err) if e is not None and e > TOL_EACH])

    # ---- criterion 2 (form R); only defined when every family is in state 1
    fth = family_mean([v["T_hat"] for v in obs], fam_of, F) if all1 else None
    fts = family_mean([v["main"][1] for v in obs], fam_of, F) if all1 else None
    if all1:
        r, p = form_r(fth, fts, fixed, n_perm, np.random.default_rng(SEED_PERM))
        c2 = c2_pass(r, p)
    else:
        r, p, c2 = None, None, False
    res["criterion2"] = dict(passed=c2, computed=all1, r=r, p=p, n_perm=n_perm, seed=SEED_PERM,
                             p_ok_but_r_not_positive=bool(all1 and p <= ALPHA_C2 and not (r > 0)))
    res["success"] = bool(c1 and c2)

    # ---- bootstrap (documents x families), shared by chains A and B and the descriptions
    rng_doc, rng_fam, rng_pb = (np.random.default_rng(s) for s in (SEED_DOC, SEED_FAM, SEED_PERM_BOOT))
    keys = CHAIN_A + ("known_mean", "guess1")
    boot = {k: np.full(n_boot, np.nan) for k in keys}
    K_boot = {k: np.zeros(n_boot, int) for k in ("guess1", "known_mean")}
    T_b = np.full((n_boot, nM), np.nan); Th_b = np.full((n_boot, nM), np.nan); st_b = np.zeros((n_boot, nM), int)
    c2_b = np.zeros(n_boot, bool)
    for b in range(n_boot):
        wP, wE, wM = round_weights(pre, rng_doc)
        idx = rng_fam.integers(0, F, F)
        vals = round_values(models, pre, wP, wE, wM)
        for k in keys:
            d = dbar_complete(vals, k, fam_of, F, idx)
            boot[k][b] = np.nan if d is None else d
        cb = compare_values(vals)
        for k in K_boot:
            K_boot[k][b] = describe_families({k: cb[k]}, fam_of, F)[k]["K"]
        for i, v in enumerate(vals):
            st_b[b, i] = v["main"][0]
            if v["main"][0] == 1:
                T_b[b, i] = v["main"][1]
            if v["T_hat"] is not None:
                Th_b[b, i] = v["T_hat"]
        if all(v["main"][0] == 1 and v["T_hat"] is not None for v in vals):
            xb = family_mean([v["T_hat"] for v in vals], fam_of, F); yb = family_mean([v["main"][1] for v in vals], fam_of, F)
            c2_b[b] = c2_pass(*form_r(xb, yb, fixed, n_perm_boot, rng_pb))
    lo_arr = {k: np.where(np.isnan(x), -np.inf, x) for k, x in boot.items()}
    hi_arr = {k: np.where(np.isnan(x), np.inf, x) for k, x in boot.items()}
    chains = run_chains(c2, obs, boot, fam_desc, F)
    res["chain_A"], res["chain_B"] = chains["A"], chains["B"]

    # ---- descriptions: guess 1 and the known mean (no test), and K distributions
    res["described"] = {}
    for k in ("guess1", "known_mean"):
        fd = fam_desc[k]
        res["described"][k] = dict(K=fd["K"], F=F, d_f=dict(zip(pre["fams"], fd["d_f"])), dbar=fd["dbar"],
                                   dbar_interval=[quantile_inf(lo_arr[k], 0.025), quantile_inf(hi_arr[k], 0.975)],
                                   missing_rounds=int(np.isnan(boot[k]).sum()),
                                   K_boot=dict(median=float(np.median(K_boot[k])), p2_5=float(np.percentile(K_boot[k], 2.5)),
                                               p97_5=float(np.percentile(K_boot[k], 97.5))))
    g1 = [x for x in fam_desc["guess1"]["d_f"] if x is not None]
    res["described"]["guess1"]["largest_family_share"] = (float(max(g1) / sum(g1)) if g1 and sum(g1) > 0 else None)
    res["direct_cases"] = {str(c): int(sum(1 for cc, _ in cmp["direct"] if cc == c)) for c in range(5)}
    case1 = [i for i, (c, _) in enumerate(cmp["direct"]) if c == 1]      # plan: both MAEs on T*_B over the same case-1 models
    res["direct_mae"] = dict(n_models=len(case1), n_excluded=nM - len(case1),
                             formula=float(np.mean([abs(obs[i]["B"][1] - obs[i]["T_hat"]) for i in case1])) if case1 else None,
                             direct=float(np.mean([abs(obs[i]["B"][1] - obs[i]["direct"][0]) for i in case1])) if case1 else None)
    res["direct_excluded_families"] = [pre["fams"][f] for f in fam_desc["direct"]["excluded"]]
    res["c2_bootstrap_pass_share"] = float(c2_b.mean())
    res["bootstrap"] = dict(n=n_boot, seed_documents=SEED_DOC, seed_families=SEED_FAM, seed_permutations=SEED_PERM_BOOT,
                            n_perm=n_perm_boot)

    # ---- sensitivity: leave one family out, tokenizer groups
    loo = []
    for f in range(F):
        keep = np.flatnonzero(np.arange(F) != f)
        if all1 and len(keep) >= 2:
            rr, pp = form_r(fth[keep], fts[keep], fixed[keep], n_perm, np.random.default_rng(SEED_PERM))
            row = dict(dropped=pre["fams"][f], r=rr, p=pp, passed=c2_pass(rr, pp))
        else:
            row = dict(dropped=pre["fams"][f], r=None, p=None, passed=False)
        for k in ("slope", "direct", "line"):
            row[f"dbar_{k}"] = dbar_complete(obs, k, fam_of, F, keep) if len(keep) else None
            row[f"K_{k}"] = int(sum(1 for g in keep if (fam_desc[k]["d_f"][g] or 0) > 0))
        loo.append(row)
    res["leave_one_family_out"] = loo
    groups = sorted({m["tok_group"] for m in models})
    g_of = np.array([groups.index(m["tok_group"]) for m in models])
    if all1 and len(groups) >= 2:
        gx, gy = family_mean([v["T_hat"] for v in obs], g_of, len(groups)), family_mean([v["main"][1] for v in obs], g_of, len(groups))
        rr, pp = form_r(gx, gy, np.bincount(g_of, minlength=len(groups)) > 1, n_perm, np.random.default_rng(SEED_PERM))
        res["tokenizer_groups"] = dict(n_groups=len(groups), r=rr, p=pp, passed=c2_pass(rr, pp))
    else:
        res["tokenizer_groups"] = dict(n_groups=len(groups), r=None, p=None, passed=False)

    # ---- per-model and per-family tables
    rows = []
    for i, (m, v) in enumerate(zip(models, obs)):
        pr = m["pred"] or {}
        T = v["main"][1]
        row = dict(no=m.get("no"), model=m["name"], family=m["family"], tok_group=m["tok_group"],
                   lang_group=m.get("lang_group"), strip_group=m.get("strip_group"),
                   prediction_written=m["pred"] is not None, T_hat=v["T_hat"], T_hat_E=pr.get("T_hat_E"),
                   T_A=pr.get("T_A"), T_B=pr.get("variant_B", {}).get("value"), T_B_status=pr.get("variant_B", {}).get("status"),
                   T_hat_corrected=(v["T_hat"] - RETOK_SHIFT) if (v["T_hat"] is not None and m["name"] in BYTE_BPE) else None,
                   state=v["main"][0], reason=v["main"][2] if m["pred"] is not None else "預測未寫出", T_star=T,
                   all_crossings=";".join(f"{c:.6f}" for c in crossings_down(temps, v["auc_main"])) if v["auc_main"] is not None else "",
                   extrapolated=extrapolate(temps, v["auc_main"]) if (v["auc_main"] is not None and v["main"][0] == 2) else None,
                   error=err[i], state_ids=v["ids"][0], T_star_ids=v["ids"][1], state_B=v["B"][0], T_star_B=v["B"][1],
                   T_direct=v["direct"][0], direct_rule=v["direct"][1], direct_case=cmp["direct"][i][0],
                   d_direct=cmp["direct"][i][1], issues=" | ".join(m["issues"]),
                   environment_stage1=m["meta"].get("environment_stage1"), environment_stage2=m["meta"].get("environment_stage2"))
        for k in BASELINES:
            row[f"b_{k}"] = v["b"].get(k)
            row[f"d_{k}"], row[f"d_{k}_kind"] = cmp[k][i]
        if row["state"] == 2 and v["T_hat"] is not None:        # supplementary: "if a crossing exists outside the grid"
            for k in BASELINES:
                b0 = v["b"][k]
                if v["main"][2] == "網格內未失效":
                    row[f"d_{k}_if_outside"] = (v["T_hat"] - b0) if max(b0, v["T_hat"]) <= temps[-1] else None
                else:
                    row[f"d_{k}_if_outside"] = (b0 - v["T_hat"]) if min(b0, v["T_hat"]) >= temps[0] else None
        ok = ~np.isnan(T_b[:, i])
        row.update(T_star_p2_5=float(np.percentile(T_b[ok, i], 2.5)) if ok.any() else None,
                   T_star_p97_5=float(np.percentile(T_b[ok, i], 97.5)) if ok.any() else None,
                   share_state1=float((st_b[:, i] == 1).mean()), share_state2=float((st_b[:, i] == 2).mean()),
                   share_state3=float((st_b[:, i] == 3).mean()))
        okh = ~np.isnan(Th_b[:, i]); oke = ok & okh
        row.update(T_hat_p2_5=float(np.percentile(Th_b[okh, i], 2.5)) if okh.any() else None,
                   T_hat_p97_5=float(np.percentile(Th_b[okh, i], 97.5)) if okh.any() else None,
                   error_p2_5=float(np.percentile(np.abs(T_b[oke, i] - Th_b[oke, i]), 2.5)) if oke.any() else None,
                   error_p97_5=float(np.percentile(np.abs(T_b[oke, i] - Th_b[oke, i]), 97.5)) if oke.any() else None)
        for path_kind in ("text", "ids"):
            for x in m["meta"].get("machine_" + path_kind, []):
                if abs(x["T"] - 1.0) < 1e-9:
                    row[f"c_T1_{path_kind}"] = x["c_mean"]
        hr = m["meta"].get("human_roundtrip")
        if hr:
            for g in ("P", "E"):
                row[f"roundtrip_{g}_prefix_changed"] = hr[g]["prefix_changed"]
                row[f"roundtrip_{g}_continuation_changed"] = hr[g]["continuation_changed"]
                row[f"truncated_{g}"] = hr[g]["truncated"]
        mt = m["meta"].get("machine_text", [])
        if mt:
            row["machine_prefix_changed"] = ";".join(str(x["prefix_changed"]) for x in mt)
            row["machine_continuation_changed"] = ";".join(str(x["continuation_changed"]) for x in mt)
            row["machine_j_back"] = ";".join(str(x["j_back"]) for x in mt)
            row["machine_truncated"] = ";".join(str(x["truncated"]) for x in mt)
        rows.append(row)
    res["per_model"] = rows
    fams = []
    for f, name in enumerate(pre["fams"]):
        members = [models[i]["name"] for i in np.flatnonzero(fam_of == f)]
        r = dict(family=name, models=";".join(members), fixed_in_permutation=bool(fixed[f]),
                 T_hat_f=float(fth[f]) if all1 else None, T_star_f=float(fts[f]) if all1 else None, error_f=fam_err[f])
        for k in BASELINES + ("direct",):
            r[f"d_f_{k}"] = fam_desc[k]["d_f"][f]
        fams.append(r)
    res["per_family"] = fams

    # ---- exploratory descriptions: variant A, corrected prediction, groups
    closer, gaps = 0, []
    for f in range(F):
        ms = [obs[i] for i in np.flatnonzero(fam_of == f)]
        if all(v["main"][0] == 1 and v["T_hat"] is not None for v in ms):
            e_formula = np.mean([abs(v["main"][1] - v["T_hat"]) for v in ms])
            e_a = np.mean([abs(v["main"][1] - v["T_A"]) for v in ms])
            closer += int(e_formula < e_a)
            gaps += [abs(v["T_hat"] - v["T_A"]) for v in ms]
    res["variant_A"] = dict(families_formula_closer=closer, max_abs_difference=float(max(gaps)) if gaps else None)
    corr = [(v["main"][1], v["T_hat"] - RETOK_SHIFT) for m, v in zip(models, obs)
            if m["name"] in BYTE_BPE and v["main"][0] == 1 and v["T_hat"] is not None]
    res["corrected_prediction"] = dict(n=len(corr), mae=float(np.mean([abs(a - b) for a, b in corr])) if corr else None,
                                       label="探索")
    grp = {}
    for key in ("lang_group", "strip_group"):
        for g in sorted({m.get(key) for m in models if m.get(key) is not None}):
            fs = sorted({m["family"] for m in models if m.get(key) == g})
            fi = [pre["fams"].index(x) for x in fs]
            e = [fam_err[f] for f in fi if fam_err[f] is not None]
            d1 = [fam_desc["guess1"]["d_f"][f] for f in fi if fam_desc["guess1"]["d_f"][f] is not None]
            ds = [fam_desc["slope"]["d_f"][f] for f in fi if fam_desc["slope"]["d_f"][f] is not None]
            grp[f"{key}={g}"] = dict(families=len(fs), mean_error=float(np.mean(e)) if e else None,
                                     mean_d_guess1=float(np.mean(d1)) if d1 else None, mean_d_slope=float(np.mean(ds)) if ds else None)
    res["groups"] = grp
    res["prediction_claim_allowed"] = bool(res["success"] and chains["A"][0]["passed"] and chains["A"][1]["passed"])
    return res


# ---------------------------------------------------------------- claims (plan: 結果對應可用說法)
def f4(x):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}"


def claims(res):
    F = res["F"]; c1, c2 = res["criterion1"], res["criterion2"]
    A = {s["comparison"]: s for s in res["chain_A"]}; B = {s["comparison"]: s for s in res["chain_B"]}
    c2_can = (f"在這次的 {F} 個模型家族中，公式值和實際失效溫度正相關（全體 r ＝ {f4(c2['r'])}）：公式說較晚失效的家族，"
              f"實際上也較晚失效；顯著性來自單一模型家族的重排檢定（p ＝ {c2['p']:.5f}）") if c2["passed"] else \
        "未能證明公式分辨得出家族之間失效溫度的高低"
    better = lambda k: (f"在這次的 {F} 個家族中，公式的家族平均絕對誤差比{LABEL[k]}小（家族平均改善量 d̄ ＝ {f4(A[k]['dbar'])}，"
                        f"單尾 98.75% 下限 {f4(A[k]['lower'])} > 0；下限已把文件與家族的抽樣算進來）")
    equiv = lambda k: (f"在這次的 {F} 個家族中，公式和{LABEL[k]}的平均絕對誤差差距在 ±0.002 以內（d̄ ＝ {f4(B[k]['dbar'])}，"
                       f"97.5% 區間 [{f4(B[k]['lower'])}, {f4(B[k]['upper'])}]，區間已把文件與家族的抽樣算進來）")
    rows = []

    def add(row, cond, can, cannot):
        if cond:
            rows.append((row, can, cannot))

    add("標準 1、2 都符合，「優於」順序也通過斜率與直接估計", res["success"] and A["slope"]["passed"] and A["direct"]["passed"],
        "只用人類文章、不擬合參數，就能在生成任何機器文字之前預測這次 " + str(F) + " 個家族的失效溫度；誤差在門檻內，" + c2_can
        + "；" + better("slope") + "；" + better("direct"), "「對模型家族普遍成立」")
    add("標準 1、2 符合，「優於」順序的斜率未通過", res["success"] and not A["slope"]["passed"],
        "重現成功：不擬合參數就能事先算出失效溫度，誤差在門檻內，" + c2_can + "；未能證明優於一參數斜率"
        + ("；和一參數斜率的平均絕對誤差差距在 ±0.002 以內" if B["slope"]["passed"] else ""),
        "「優於斜率」「預測」" + ("" if B["slope"]["passed"] else "「和斜率相當」"))
    add("標準 1、2 與斜率通過，直接估計未通過", res["success"] and A["slope"]["passed"] and not A["direct"]["passed"],
        "重現成功，" + c2_can + "，" + better("slope") + "；未能證明優於直接估計，公式的用途限於還沒有機器文字的時候",
        "「比直接量測準」「預測」")
    add("只符合標準 1", c1["passed"] and not c2["passed"],
        "準確度達到事先登錄的門檻，但未能證明公式分辨得出家族之間失效溫度的高低", "「重現成功」「優於基準」")
    if not c1["passed"]:
        reasons = []
        if c1["over_0_02"] or (c1["family_mae"] is not None and c1["family_mae"] > TOL_MAE):
            reasons.append(f"實測誤差超過門檻（超過 0.02：{', '.join(c1['over_0_02']) or '無'}；家族平均 {f4(c1['family_mae'])}）")
        for x in c1["failures"]:
            tag = "預測未寫出" if x["reason"] == "預測未寫出" else ("網格內沒有交會（狀態②）" if x["state"] == 2 else "原因未定（狀態③）")
            reasons.append(f"{x['model']}：{tag}，{x['reason']}")
        add("標準 1 不符合", True, "未達重現成功，分列原因：" + "；".join(reasons), "「重現成功」")
    for k in CHAIN_A:
        s = A[k]
        add(f"「優於」順序任何一步未通過或未輪到：{LABEL[k]}（{s['status']}）", not s["passed"],
            f"未能證明優於{LABEL[k]}" + ("；未輪到，標「未檢定」" if s["status"] == "未檢定" else f"（{s['status']}）"), f"「優於{LABEL[k]}」")
        add(f"「優於」順序某一步通過：{LABEL[k]}", s["passed"], better(k), "")
        add(f"「優於」順序某一步通過，但該比較的家族平均改善量 d̄ ≤ 0：{LABEL[k]}",
            s["passed"] and s["dbar"] is not None and s["dbar"] <= 0,
            f"檢定通過，但實際的家族平均改善量不為正（d̄ ＝ {f4(s['dbar'])}，下限 {f4(s['lower'])}）；寫出兩者，並列出拉低平均的家族與差距",
            f"「平均而言優於{LABEL[k]}」「整體優於{LABEL[k]}」")
    for k in CHAIN_B:
        s = B[k]
        add(f"「相當」順序某一步通過：{LABEL[k]}", s["passed"], equiv(k),
            "不附界限與區間的「一樣準」「沒有差別」；「每個家族都一樣準」；推到名單以外")
        add(f"「相當」順序未通過、未輪到或資料不完整：{LABEL[k]}（{s['status']}）", not s["passed"],
            f"未能證明和{LABEL[k]}相當（{s['status']}{'：' + s['reason'] if s.get('reason') else ''}；d̄ ＝ {f4(s['dbar'])}，97.5% 區間 [{f4(s['lower'])}, {f4(s['upper'])}]）。"
            "這不表示兩者有差別", "「相當」「差不多準」「兩者有差別」")
    add("猜 T* = 1、直線、已知平均", True, "這三者沒有登錄等效檢定", "「和〔這三者〕相當」")
    for k in ("guess1", "known_mean"):
        d = res["described"][k]
        add(f"對猜 T* = 1 與已知平均（任何結果）：{LABEL[k]}", True,
            f"在這次使用的人類文章下，本次 {F} 個家族中有 {d['K']} 個的公式值比{LABEL[k]}更接近實際值（描述，沒有做檢定）；"
            f"家族平均改善量 d̄ ＝ {f4(d['dbar'])}（重抽 95% 區間 [{f4(d['dbar_interval'][0])}, {f4(d['dbar_interval'][1])}]）",
            f"「優於{LABEL[k]}」「平均而言優於{LABEL[k]}」「多數模型家族的真實勝率超過一半」「對多數模型家族成立」")
    flips = [x for x in res["leave_one_family_out"] if c2["passed"] and not x["passed"]]
    add("逐家族移除：拿掉某一家族後標準 2 不再通過", bool(flips),
        "標準 2 的結論依賴下列家族：" + "、".join(x["dropped"] for x in flips), "不加說明地寫「普遍」「穩健」")
    add("標準 2：p ≤ 0.025 但全體 r ≤ 0", c2["p_ok_but_r_not_positive"],
        f"標準 2 不符合。單一模型家族之間正相關，但把含兩個模型的家族算進來後，全體相關不為正（r ＝ {f4(c2['r'])}）",
        "「家族之間正相關」「重現成功」")
    va = res["variant_A"]
    add("公式對變體 A（只報告，不檢定）", True,
        f"本次 {F} 個家族中公式值較接近的家族數 {va['families_formula_closer']}，兩者數值的最大差距 {f4(va['max_abs_difference'])}",
        "「公式提供了偵測器分數以外的資訊」")
    add("任何結果", True, "結論限於這次名單上的家族", "把結論推到名單以外的模型家族，例如「模型家族普遍如此」")
    lines = ["# e12 可用說法（依計畫第 5.1 版「結果對應可用說法」自動選出）", "",
             f"- 重現成功：{'是' if res['success'] else '否'}；「預測」一詞：{'可以使用' if res['prediction_claim_allowed'] else '不可使用，改寫「事先算出」'}", ""]
    for row, can, cannot in rows:
        lines += [f"## {row}", "", f"- 可以寫：{can}"] + ([f"- 不可以寫：{cannot}"] if cannot else []) + [""]
    return "\n".join(lines), [r[0] for r in rows]


# ---------------------------------------------------------------- output
def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return None if math.isnan(x) else (("inf" if x > 0 else "-inf") if math.isinf(x) else x)
    return x


def write_outputs(res, out_dir):
    out_dir = pathlib.Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(res["per_model"]).to_csv(out_dir / "per_model.csv", index=False)
    pd.DataFrame(res["per_family"]).to_csv(out_dir / "per_family.csv", index=False)
    crit = {k: v for k, v in res.items() if k not in ("per_model", "per_family")}
    with open(out_dir / "criteria.json", "w", encoding="utf-8") as f:
        json.dump(clean(crit), f, ensure_ascii=False, indent=1)
        f.write("\n")
    md, _ = claims(res)
    (out_dir / "claims.md").write_text(md + "\n", encoding="utf-8")


def main(argv=None):
    from cmargin.paths import DATA, RESULTS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(DATA)); ap.add_argument("--registration", default=str(ROOT / "registration" / "registration.json"))
    ap.add_argument("--out", default=str(RESULTS / "e12_replication_test"))
    ap.add_argument("--dryrun", action="store_true", help="accept the reduced grid and subsets of a notebook-15 dry run")
    a = ap.parse_args(argv)
    try:
        inp = load_inputs(a.data, a.registration, dryrun=a.dryrun)
        res = analyze(inp)
    except InputError as err:
        sys.exit(f"停止：{err}")
    write_outputs(res, a.out)
    c1, c2 = res["criterion1"], res["criterion2"]
    print(f"criterion 1: {'met' if c1['passed'] else 'not met'} (family MAE {f4(c1['family_mae'])})")
    print(f"criterion 2: {'met' if c2['passed'] else 'not met'} (r {f4(c2['r'])}, p {c2['p']})")
    for name in ("A", "B"):
        print(f"chain {name}: " + " -> ".join(f"{s['comparison']} {s['status']}" for s in res[f"chain_{name}"]))
    print("outputs in", a.out)


if __name__ == "__main__":
    main()
