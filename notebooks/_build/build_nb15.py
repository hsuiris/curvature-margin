"""Build notebook 15 (replication test, design doc 8.8.24, plan 5.1) from one source: the English version for the repo and
the Chinese version for Drive. Code is identical in both; only these differ:
  ⟦"en"¦"zh"⟧  a message string (both sides must be one Python string literal with the same f-string expressions);
               every pair is written to scripts/nb15_allowed_diffs.json for scripts/check_nb15_pair.py;
  ⟪en¦zh⟫      comment or markdown text (comments are stripped before the comparison);
  ⟨DRIVE_DATA⟩ the Drive data folder in cell 2 (listed separately in the allowed-diffs file).
Usage: python build_nb15.py   (NB15_REPO overrides the repository folder)."""
import ast, json, os, re
from pathlib import Path

REPO = Path(os.environ.get("NB15_REPO", "/private/tmp/cm_work/curvature-margin"))
DRIVE = Path("/Users/xuyunqin/Library/CloudStorage/GoogleDrive-emilyhuang12380@gmail.com/我的雲端硬碟/AI-Text/03_實驗/notebooks")
DST = {"en": REPO / "notebooks" / "15_replication_test.ipynb", "zh": DRIVE / "15_replication_test.ipynb"}
ALLOWED = REPO / "scripts" / "nb15_allowed_diffs.json"
DRIVE_DATA = {"en": '"/content/drive/MyDrive/curvature-margin/data"', "zh": '"/content/drive/MyDrive/AI-Text/03_實驗/data"'}
META = {"accelerator": "GPU", "colab": {"gpuType": "A100", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}

CELLS = []


def md(text):
    CELLS.append(("markdown", text.strip("\n")))


def code(text):
    CELLS.append(("code", text.strip("\n")))


# ================================================================== cell 0
md(r"""
⟪# 15 · Replication test: the failure-temperature formula on 12 new models (design doc 8.8.24, plan 5.1, pre-registered)¦# 15 重現實驗：在 12 個新模型上檢驗失效溫度公式（設計文件 8.8.24，計畫第 5.1 版，事先登錄）⟫

⟪**Publication copy:** Instructions and messages were edited after registration. Numerical settings and executable logic are unchanged; the original registration records are archived separately and are not included in this publication snapshot.¦**公開整理版：**操作說明與提示文字在登錄後修訂。數值設定與執行邏輯維持原樣；原始登錄紀錄另行封存，未收錄於此公開版本。⟫

⟪**How to run**: every session, pick an **A100** Colab server and click "Run All". The notebook works out which stage it is in. Each stage stops with a red error message (`raise`) when it is done: that is expected. Complete the registration step for that stage, then click "Run All" again.¦**怎麼跑**：每次開機選 **A100** 的 Colab 伺服器，按「Run All」。notebook 會自己判斷目前在哪個階段；每個階段做完會用 `raise` 停下來，出現紅色錯誤訊息是正常的。完成該階段的登錄後，再按一次「Run All」。⟫

⟪- Cell 1 kills the other notebook kernels on the same machine and checks for an A100 with at least 20 GB free.¦- 第 1 格會先清掉同一台機器上其他 notebook 的 kernel，並確認是 A100、可用記憶體至少 20 GB。⟫
⟪- Llama 3.2 and Gemma 3 need a Hugging Face token: add `HF_TOKEN` under the key icon (Secrets) on the left and allow this notebook to use it. The token is never printed or written to a file.¦- Llama 3.2 與 Gemma 3 需要 Hugging Face 授權：在左側「鑰匙」圖示（Secrets）新增 `HF_TOKEN`，並允許這份 notebook 存取。token 不會印出，也不會寫進任何檔案。⟫
⟪- If the connection drops, click "Run All" again: finished files that pass the checks are kept; a damaged file is moved to `_invalid/` with the reason and redone.¦- 中途斷線就重新「Run All」：做完而且檢查通過的檔案會沿用；壞掉的檔案會移到 `_invalid/` 並記錄原因，再重做。⟫

⟪| Stage | What it does | Rough time on an A100 | Ends with |¦| 階段 | 做什麼 | A100 粗估時間 | 結束時 |⟫
| --- | --- | --- | --- |
⟪| 0 | tokenizer and GPU pre-checks (12 models and the candidate Zamba2); no scores are kept | about 0.5 h | `stage0_record.json`, stop until R1 |¦| 第零階段 | tokenizer 與 GPU 預檢（12 個模型加候補 Zamba2），不保存任何分數 | 約 0.5 小時 | 寫出 `stage0_record.json`，停下等 R1 |⟫
⟪| 1 | each model scores the P and E human texts; `predictions.json` is written before any text is generated | about 0.2 h plus loading | SHA-256 printed, stop until R2 |¦| 第一階段 | 每個模型評分 P、E 人類文章；在生成任何文字之前寫出 `predictions.json` | 約 0.2 小時加上載入 | 印出 SHA-256，停下等 R2 |⟫
⟪| 2 | gate check, then 300 continuations at each of 11 temperatures, scored on the text path and the id path | about 4.5 h | `run_info.json` |¦| 第二階段 | 閘門核對後，11 個溫度各續寫 300 篇，文字路徑與編號路徑各評分一次 | 約 4.5 小時 | 寫出 `run_info.json` |⟫

⟪Results go to `DATA_DIR/15_replication_test/`. The pre-registered analysis is `scripts/e12_replication_test.py`.¦結果存在 `03_實驗/data/15_replication_test/`。事先登錄的分析程式是公開倉庫的 `scripts/e12_replication_test.py`。⟫
""")

# ================================================================== cell 1
code(r'''
# 1. ⟪Run mode; kill the other notebook kernels on this machine; check the GPU (requirement R1.4).¦執行模式；清掉同一台機器上其他 notebook 的 kernel；檢查 GPU（需求 R1.4）。⟫
#    ⟪DRYRUN is True only in the laptop dry run (scripts/dryrun_15.py changes this one line). It must be False on Colab.¦DRYRUN 只有筆電試跑（scripts/dryrun_15.py 只改這一行）才是 True；在 Colab 上必須是 False。⟫
DRYRUN = False
import os, sys, signal, subprocess, time, math, tempfile, importlib.util
DRY = dict(   # ⟪values used instead of the registered ones when DRYRUN is True; nothing else changes¦DRYRUN 為 True 時取代登錄值的設定；除此之外程式完全相同⟫
    device="cpu", gpu_names=("cpu",), min_free_gb=0, kill=(), data_dir=os.path.join(tempfile.gettempdir(), "nb15_dryrun", "data"),
    mount_drive=False, hf_home=None, hf_token=False, clean_cache=False, install=False,
    quota={s: (10, 10, 1, 1) for s in ("arxiv", "outfox", "peerread", "reddit", "wikihow", "wikipedia")}, exact_manifest=False,
    temps=(1.00, 1.14), dtype="float32", candidate=None, min_families=1, pm_max_min=float("inf"), total_max_h=float("inf"),
    models=[dict(no=1, name="gpt2", family="GPT-2", group="D1", lang="other", strip=False, gated=False),
            dict(no=2, name="h2oai/h2o-danube3-500m-base", family="Danube", group="D2", lang="other", strip=True, gated=False)],
    gpu_check=dict(prompts=4, new_tokens=8, context=48, steps=8, score_batch=2, score_len=64, repeats=3))


def setting(name, value):
    """The registered value, or its dry-run replacement when DRYRUN is True."""
    return DRY[name] if DRYRUN else value


ON_COLAB = importlib.util.find_spec("google") is not None and importlib.util.find_spec("google.colab") is not None
assert not (DRYRUN and ON_COLAB), ⟦"DRYRUN must be False on Colab."¦"在 Colab 上 DRYRUN 必須是 False。"⟧
import torch
DEVICE = setting("device", "cuda")
me = os.getpid(); patterns = setting("kill", ("kernel_launcher",))
procs = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout.splitlines()[1:]
victims = [int(l.split()[0]) for l in procs if any(p in l for p in patterns) and int(l.split()[0]) != me]
for p in victims:
    os.kill(p, signal.SIGKILL)
if victims:
    time.sleep(3)


def device_check():
    """Cell 1 and the start of every stage: the registered device (an A100 with at least 20 GB free)."""
    if DEVICE == "cuda":
        assert torch.cuda.is_available(), ⟦"No GPU: switch the Colab server to an A100."¦"沒有 GPU：請把 Colab 伺服器換成 A100。"⟧
        name, free_gb = torch.cuda.get_device_name(0), torch.cuda.mem_get_info()[0] / 2**30
    else:
        name, free_gb = DEVICE, float("inf")
    assert any(g in name for g in setting("gpu_names", ("A100",))), ⟦f"{name} is not an A100; choose an A100 server."¦f"{name} 不是 A100，請改選 A100 伺服器。"⟧
    assert free_gb >= setting("min_free_gb", 20), ⟦f"Only {free_gb:.0f} GB free; restart the session and run again."¦f"可用記憶體只有 {free_gb:.0f} GB，請重新啟動執行階段再跑。"⟧
    return name, free_gb


GPU_NAME, free_gb = device_check()
print(⟦f"Killed: {victims} | {GPU_NAME}, {free_gb:.0f} GB free | DRYRUN = {DRYRUN}"¦f"砍掉：{victims}｜{GPU_NAME} 可用 {free_gb:.0f} GB｜DRYRUN = {DRYRUN}"⟧)
''')

# ================================================================== cell 2
code(r'''
# 2. ⟪Mount Google Drive; data folder; Hugging Face cache; read HF_TOKEN from Colab Secrets (never printed or saved).¦掛載雲端硬碟；資料夾；Hugging Face 快取；從 Colab Secrets 讀 HF_TOKEN（不印出、不存檔）。⟫
DATA_DIR = setting("data_dir", ⟨DRIVE_DATA⟩)
if setting("mount_drive", True):
    from google.colab import drive
    drive.mount("/content/drive")
    flush_drive = drive.flush_and_unmount
else:
    flush_drive = lambda: None
assert os.path.isdir(DATA_DIR), ⟦f"Data folder not found: {DATA_DIR}"¦f"找不到資料夾：{DATA_DIR}"⟧
os.chdir(DATA_DIR)
OUT = "15_replication_test"
os.makedirs(OUT, exist_ok=True)
HF_HOME = setting("hf_home", "/content/hf")
if HF_HOME:
    os.environ["HF_HOME"] = HF_HOME
HF_TOKEN = None
if setting("hf_token", True):
    try:
        from google.colab import userdata
        HF_TOKEN = userdata.get("HF_TOKEN")
    except Exception:   # ⟪no secret or no access: ask once, hidden input¦沒有設定或沒有權限：隱藏輸入問一次⟫
        import getpass
        HF_TOKEN = getpass.getpass("HF_TOKEN: ") or None
print(⟦f"Data folder: {os.getcwd()} | output: {OUT} | HF token set: {HF_TOKEN is not None}"¦f"資料夾：{os.getcwd()}｜輸出：{OUT}｜HF token 已設定：{HF_TOKEN is not None}"⟧)
''')

# ================================================================== cell 3
code(r'''
# 3. ⟪Install the registered versions of the pip packages, then check the versions really loaded; the Mamba kernels must be absent.¦安裝登錄的 pip 套件版本，再核對實際載入的版本；Mamba 類的三個加速套件必須無法 import。⟫
#    ⟪torch, numpy, scipy and Python stay Colab's own: cell 5 records them at the first run and stops if a later session differs.¦torch、numpy、scipy 與 Python 用 Colab 預裝版：第 5 格在第一次執行時記錄，之後任何一次不同就停下。⟫
PINS = {"transformers": "5.17.0", "tokenizers": "0.23.2", "huggingface-hub": "1.33.0", "safetensors": "0.8.0", "accelerate": "1.15.0"}
PREINSTALLED = ("torch", "numpy", "scipy")
MODULES = {"transformers": "transformers", "tokenizers": "tokenizers", "huggingface-hub": "huggingface_hub", "safetensors": "safetensors",
           "accelerate": "accelerate", "torch": "torch", "numpy": "numpy", "scipy": "scipy"}
if setting("install", True):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + [f"{k}=={v}" for k, v in PINS.items()])
from importlib.metadata import version


def stale_modules(installed):
    """Packages imported before the installation (cell 1's torch already pulled in numpy) whose loaded code is not the
    installed version: the session has to be restarted."""
    return {k: (sys.modules[m].__version__, installed[k]) for k, m in MODULES.items()
            if m in sys.modules and sys.modules[m].__version__ != installed[k]}


INSTALLED = {k: version(k) for k in MODULES}
bad = {k: (INSTALLED[k], v) for k, v in PINS.items() if INSTALLED[k] != v}
assert not bad, ⟦f"Installed versions differ from the registered ones: {bad}"¦f"安裝的版本和登錄的不同：{bad}"⟧
assert not stale_modules(INSTALLED), ⟦f"A package changed after it was imported {stale_modules(INSTALLED)}: restart the session (Runtime > Restart session), then Run All."¦f"套件在載入後換了版本 {stale_modules(INSTALLED)}：請重新啟動執行階段（Runtime > Restart session）再 Run All。"⟧
for pkg in ("mamba_ssm", "causal_conv1d", "mambapy"):   # ⟪Mamba and Zamba2 must run on transformers' reference PyTorch path¦Mamba 與 Zamba2 必須走 transformers 內建的 PyTorch 路徑⟫
    assert importlib.util.find_spec(pkg) is None, ⟦f"{pkg} is installed; the registered run uses the PyTorch path only."¦f"{pkg} 已安裝；登錄的做法只用 PyTorch 路徑。"⟧
import json, gc, glob, shutil, hashlib, pickle, platform, re, logging, statistics
from datetime import datetime, timezone
import numpy as np, pandas as pd, scipy
import transformers, tokenizers, huggingface_hub, safetensors, accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, GenerationConfig
from huggingface_hub import HfApi, hf_hub_download
from scipy.optimize import brentq
from scipy.special import ndtr
assert not stale_modules(INSTALLED), ⟦f"A package changed after it was imported {stale_modules(INSTALLED)}: restart the session (Runtime > Restart session), then Run All."¦f"套件在載入後換了版本 {stale_modules(INSTALLED)}：請重新啟動執行階段（Runtime > Restart session）再 Run All。"⟧
VERSIONS = dict(INSTALLED)   # ⟪recorded in stage0_record.json and the run record; only the pinned pip versions (PINS) enter the config hash, Colab's preinstalled ones and Python are checked in cell 5 (plan 5.4)¦記進 stage0_record.json 與執行紀錄；只有 pip 寫死的版本（PINS）進設定雜湊，Colab 預裝的版本與 Python 在第 5 格核對（計畫 5.4）⟫
print(VERSIONS)
''')

# ================================================================== cell 4
code(r'''
# 4. ⟪Human texts: check the raw file, the P/E manifest and every text hash; build P, E, the A and B halves of the prompts.¦人類文章：核對原始檔、P／E manifest 與逐篇文字雜湊；建立 P、E 與提示的 A 半、B 半。⟫
RAW, MAN, MAN_A = "external/m4gt/SubtaskB.jsonl", "manifests/m4gt_humans_PE_manifest.csv", "manifests/m4gt_humans_A_manifest.csv"
EXPECTED_RAW = "4d65fc1fb93b5c9d21b226aca06b610ee031fa9ccbc18fa6f39d16273dd281c7"
EXPECTED_MAN = "3880e0ccaa2fa904392b461ce7f41d6c4bd1b62e618d14711d112f3d3c22e598"   # ⟪implementation list item 1; registered at R0¦實作清單第 1 項產生；R0 登錄⟫
EXPECTED_MAN_A = "7c6f2b6529a8ef7b1a1920bc798d45c68a7a7162823566f52a73deb6ae8d1caf"
SOURCES = ("arxiv", "outfox", "peerread", "reddit", "wikihow", "wikipedia")
QUOTA = setting("quota", {"arxiv": (167, 167, 25, 25), "outfox": (167, 167, 25, 25), "peerread": (167, 167, 25, 25),
                          "reddit": (167, 167, 25, 25), "wikihow": (166, 166, 25, 25), "wikipedia": (166, 166, 25, 25)})   # ⟪(P, E, A-half prompts, B-half prompts) per source¦每個來源的（P、E、A 半提示、B 半提示）篇數⟫


def sha256_file(path):
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            d.update(block)
    return d.hexdigest()


def norm_text(t):
    return re.sub(r"\s+", " ", t).strip().lower()


def select_rows(man, quota):
    """The first rows of each source and role in manifest order: all of them with the registered quota."""
    parts = []
    for s in SOURCES:
        nP, nE, nA, nB = quota[s]
        for role, n in (("P", nP), ("prompt_A", nA), ("prompt_B", nB), ("reference", nE - nA - nB)):
            rows = man[(man.source == s) & (man.role == role)].head(n)
            assert len(rows) == n, ⟦f"{s}/{role}: {len(rows)} rows, expected {n}"¦f"{s}／{role}：{len(rows)} 列，應為 {n}"⟧
            parts.append(rows)
    return pd.concat(parts)


def read_texts(raw_path, wanted):
    """{line number: text} for the wanted lines, checking each text hash (wanted: line -> (text_sha256, source))."""
    found = {}
    with open(raw_path, encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            if ln in wanted:
                it = json.loads(line); sha, src = wanted[ln]
                assert hashlib.sha256(it["text"].encode("utf-8")).hexdigest() == sha, ⟦f"Text hash mismatch on line {ln}"¦f"第 {ln} 行文字雜湊不符"⟧
                assert it["source"] == src and it["model"] == "human", ⟦f"Line {ln} is not a human text from {src}"¦f"第 {ln} 行不是 {src} 的人類文章"⟧
                found[ln] = it["text"]
    assert len(found) == len(wanted), ⟦"Some manifest lines are missing from the raw file"¦"原始檔少了 manifest 裡的某些行"⟧
    return found


assert sha256_file(RAW) == EXPECTED_RAW, ⟦"Raw file SHA-256 does not match; stopping."¦"原始檔 SHA-256 不符，停止。"⟧
assert sha256_file(MAN) == EXPECTED_MAN, ⟦"P/E manifest SHA-256 does not match; stopping."¦"P／E manifest 的 SHA-256 不符，停止。"⟧
assert sha256_file(MAN_A) == EXPECTED_MAN_A, ⟦"A-group manifest SHA-256 does not match; stopping."¦"A 組 manifest 的 SHA-256 不符，停止。"⟧
man = pd.read_csv(MAN)
assert list(man.columns) == ["id", "group", "role", "line_number", "source", "word_count", "text_sha256", "norm_sha256"], ⟦"Unexpected manifest columns"¦"manifest 欄位和計畫不符"⟧
assert man.id.is_unique and set(man.group) <= {"P", "E"} and set(man.role) <= {"P", "prompt_A", "prompt_B", "reference"}
assert ((man.group == "P") == (man.role == "P")).all(), ⟦"P rows must have role P and only P rows"¦"只有 P 組的列角色是 P"⟧
sel = select_rows(man, QUOTA)
if setting("exact_manifest", True):
    assert len(sel) == len(man), ⟦"The manifest has rows outside the registered quota"¦"manifest 有配額以外的列"⟧
man_a = pd.read_csv(MAN_A)
a_src = man_a[man_a.role == "source"].sort_values("id")
wanted = {int(r.line_number): (r.text_sha256, r.source) for r in sel.itertuples()}
wanted.update({int(r.line_number): (r.text_sha256, r.source) for r in a_src.itertuples()})
texts = read_texts(RAW, wanted)
for r in sel.itertuples():
    assert hashlib.sha256(norm_text(texts[int(r.line_number)]).encode("utf-8")).hexdigest() == r.norm_sha256, ⟦f"Normalized hash mismatch, id {r.id}"¦f"正規化雜湊不符，編號 {r.id}"⟧
HUMANS = {int(r.id): dict(id=int(r.id), group=r.group, role=r.role, source=r.source, text=texts[int(r.line_number)]) for r in sel.itertuples()}
P_IDS = sorted(i for i, h in HUMANS.items() if h["group"] == "P")
E_IDS = sorted(i for i, h in HUMANS.items() if h["group"] == "E")
A_IDS = sorted(i for i, h in HUMANS.items() if h["role"] == "prompt_A")
B_IDS = sorted(i for i, h in HUMANS.items() if h["role"] == "prompt_B")
PROMPT_IDS = sorted(A_IDS + B_IDS)
A_PROMPTS = [texts[int(r.line_number)] for r in a_src.itertuples()]   # ⟪A group (first test): GPU check only¦A 組（第一次檢驗）：只用於 GPU 檢查⟫
# ⟪The main endpoint's human reference is the WHOLE E group (prompts included: 1,000 texts), not role == "reference" (700).¦主要終點的人類參考是整個 E 組（含提示，共 1,000 篇），不是 role == "reference" 的 700 篇。⟫
# ⟪T*_B uses E minus the A half (850); the direct estimate uses P (1,000).¦T*_B 用 E 組扣掉 A 半（850 篇）；直接估計用 P 組（1,000 篇）。⟫
n_quota = [sum(q[i] for q in QUOTA.values()) for i in range(4)]
assert [len(P_IDS), len(E_IDS), len(A_IDS), len(B_IDS)] == n_quota, ⟦f"group sizes {len(P_IDS), len(E_IDS), len(A_IDS), len(B_IDS)} differ from the quota {n_quota}"¦f"各組篇數 {len(P_IDS), len(E_IDS), len(A_IDS), len(B_IDS)} 和配額 {n_quota} 不符"⟧
assert set(PROMPT_IDS) <= set(E_IDS) and not set(P_IDS) & set(E_IDS)
assert len(set(E_IDS) - set(A_IDS)) == n_quota[1] - n_quota[2]
print(⟦f"P {len(P_IDS)}, E {len(E_IDS)} (prompts: A half {len(A_IDS)}, B half {len(B_IDS)}); A-group prompts for the GPU check: {len(A_PROMPTS)}"¦f"P {len(P_IDS)} 篇、E {len(E_IDS)} 篇（提示：A 半 {len(A_IDS)}、B 半 {len(B_IDS)}）；GPU 檢查用的 A 組提示 {len(A_PROMPTS)} 篇"⟧)
''')

# ================================================================== cell 5
code(r'''
# 5. ⟪Pre-registered settings (plan 8.8.24) and shared functions. Text-path scoring and every resume check get their input only from text_input.¦事先登錄的設定（計畫 8.8.24）與共用函式。文字路徑的評分與續跑檢查只能透過 text_input 取得輸入。⟫
TEMPS = setting("temps", (0.94, 0.97, 0.99, 1.00, 1.01, 1.02, 1.03, 1.04, 1.06, 1.09, 1.14))
PREFIX, MAX_RAW, MAX_TOK, BS, SBS, SEED = 30, 511, 512, 32, 16, 42
MIN_TOKENS, MIN_SCORED, ROUNDTRIP_MIN, PEAK_MAX = 50, 20, 0.99, 0.80
DTYPE = getattr(torch, setting("dtype", "bfloat16"))
SLOPE_K, LINE_A, LINE_K, KNOWN_MEAN, VARIANT_B_RANGE = 0.1451, 1.0004, 0.1409, 1.0089, (0.80, 1.30)
STRS = ("Hello world.", "a", " The cat sat.", "1234 5678", "Ünïcödé text")   # ⟪rule 2: the five fixed strings¦規則 2 的五個固定字串⟫
MODELS = setting("models", [
    dict(no=1, name="meta-llama/Llama-3.2-1B", family="Llama", group="G1", lang=None, strip=False, gated=True),
    dict(no=2, name="google/gemma-3-1b-pt", family="Gemma", group="G2", lang=None, strip=False, gated=True),
    dict(no=3, name="TinyLlama/TinyLlama_v1.1", family="TinyLlama", group="G3", lang="other", strip=True, gated=False),
    dict(no=4, name="togethercomputer/RedPajama-INCITE-Base-3B-v1", family="RedPajama", group="G4", lang="other", strip=False, gated=False),
    dict(no=5, name="stabilityai/stablelm-3b-4e1t", family="StableLM", group="G4", lang="other", strip=False, gated=False),
    dict(no=6, name="h2oai/h2o-danube2-1.8b-base", family="Danube", group="G5", lang="other", strip=True, gated=False),
    dict(no=7, name="h2oai/h2o-danube3-500m-base", family="Danube", group="G5", lang="other", strip=True, gated=False),
    dict(no=8, name="utter-project/EuroLLM-1.7B", family="EuroLLM", group="G6", lang="multilingual", strip=True, gated=False),
    dict(no=9, name="BSC-LT/salamandra-2b", family="Salamandra", group="G7", lang="multilingual", strip=True, gated=False),
    dict(no=10, name="facebook/xglm-1.7B", family="XGLM", group="G8", lang="multilingual", strip=True, gated=False),
    dict(no=11, name="state-spaces/mamba-1.4b-hf", family="Mamba", group="G4", lang="other", strip=False, gated=False),
    dict(no=12, name="PleIAs/Pleias-1.2b-Preview", family="Pleias", group="G9", lang="multilingual", strip=False, gated=False)])
CANDIDATE = setting("candidate", dict(no=13, name="Zyphra/Zamba2-1.2B", family="Zyphra", group=None, lang=None, strip=True, gated=False))
GPU_CHECK = setting("gpu_check", dict(prompts=32, new_tokens=64, context=400, steps=64, score_batch=16, score_len=512, repeats=3))
PM_MAX_MIN, TOTAL_MAX_H, MIN_FAMILIES = setting("pm_max_min", 120), setting("total_max_h", 7.5), setting("min_families", 8)
MAX_ATTEMPTS, STAGE2_DAYS = 4, 14   # ⟪first run plus three reruns per model and stage; stage 2 must finish within 14 days¦每個模型每個階段：第一次加最多重跑 3 次；第二階段開始後 14 天內完成⟫
DECLARED_LOAD_FAILURES = {}   # ⟪rule 1: {model name: reason}, only after loading has failed for 7 days after R0¦規則 1：{模型名稱: 原因}，只有 R0 之後 7 天仍無法載入才填⟫
DECLARED_STAGE1_FAILURES = {}   # ⟪stage 1: {model name: reason} declared by the host (e.g. the repository was taken down); "prediction not written", registered at R2¦第一階段：主持人宣告無法載入的模型 {模型名稱: 原因}（例如倉庫被下架）；記為「預測未寫出」，寫進 R2⟫
CLEAN_CACHE = setting("clean_cache", True)
SCHEMA = "nb15/v1"
R1_FILE, GO_FILE, PRED_FILE, RUNCFG_FILE = (f"{OUT}/{n}" for n in ("r1_models.json", "stage2_go.json", "predictions.json", "run_config.json"))
STAGE0_FILE, ENV_FILE = f"{OUT}/stage0_record.json", f"{OUT}/environment.json"
ATTEMPTS_FILE, STATUS2_FILE, START2_FILE = (f"{OUT}/{n}" for n in ("attempts.json", "stage2_status.json", "stage2_started.json"))
N_H = len(P_IDS) + len(E_IDS)


class StageDone(Exception):
    """Raised at the end of a stage so that Run All stops there; this is expected."""


class StageStop(Exception):
    """A registered result no longer matches its files: stop for review; nothing is redone."""


tag_of = lambda name: name.split("/")[-1]
fmt_t = lambda t: f"{t:.2f}"
utc = lambda: datetime.now(timezone.utc).isoformat()
canon = lambda obj: json.dumps(obj, sort_keys=True, separators=(",", ":"))
config_hash = lambda cfg: hashlib.sha256(canon(cfg).encode()).hexdigest()
sha256_list = lambda xs: hashlib.sha256(canon([int(x) for x in xs]).encode()).hexdigest()


def atomic_write(path, data):
    """Write to a temporary file, then rename: a file either exists complete or not at all."""
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def write_json(path, obj):
    atomic_write(path, (json.dumps(obj, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))


def move_invalid(path, reason):
    """Keep a file that failed its check (never delete it) and log why."""
    os.makedirs(f"{OUT}/_invalid", exist_ok=True)
    dst = f"{OUT}/_invalid/{os.path.basename(path)}.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
    shutil.move(path, dst)
    with open(f"{OUT}/_invalid/log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(utc=utc(), file=os.path.basename(path), moved_to=os.path.basename(dst), reason=reason), ensure_ascii=False) + "\n")
    print(⟦f"moved to _invalid/: {os.path.basename(path)} ({reason})"¦f"移到 _invalid/：{os.path.basename(path)}（{reason}）"⟧)


# ---- ⟪tokenizer rules 2-6¦tokenizer 規則 2 到 6⟫
def load_tokenizer(name, revision):
    return AutoTokenizer.from_pretrained(name, revision=revision, token=HF_TOKEN)


def model_meta(name, revision):
    cfg = AutoConfig.from_pretrained(name, revision=revision, token=HF_TOKEN)
    try:
        gen = GenerationConfig.from_pretrained(name, revision=revision, token=HF_TOKEN)
    except OSError:   # ⟪no generation_config.json¦沒有 generation_config.json⟫
        gen = None
    return cfg, gen


def vocab_size(cfg):
    v = getattr(cfg, "vocab_size", None)
    return int(v if v is not None else cfg.get_text_config().vocab_size)


def as_ids(x):
    return [] if x is None else [int(v) for v in (x if isinstance(x, (list, tuple)) else [x]) if v is not None]


def start_tokens(tok):
    """Rule 2: the ids tok(x) adds over tok(x, add_special_tokens=False); all at the start and the same for 5 strings."""
    outs = []
    for s in STRS:
        w, wo = tok(s)["input_ids"], tok(s, add_special_tokens=False)["input_ids"]
        if len(w) < len(wo) or w[len(w) - len(wo):] != wo:
            return None, ⟦f"extra tokens are not all at the start for {s!r}"¦f"{s!r} 多出的記號不全在開頭"⟧
        outs.append(w[:len(w) - len(wo)])
    if any(o != outs[0] for o in outs):
        return None, ⟦"the extra tokens differ between the five strings"¦"五個字串多出的記號不一致"⟧
    return [int(t) for t in outs[0]], None


def banned_ids(tok, cfg, gen):
    """Rule 3: special ids, special added tokens, bos/eos/pad of tokenizer, config and generation_config, and the model
    ids without text. Tokenizer ids at or above vocab_size cannot be sampled and are dropped (and recorded)."""
    V = vocab_size(cfg)
    tok_side = set(tok.all_special_ids) | {int(i) for i, t in tok.added_tokens_decoder.items() if t.special}
    for x in (tok.bos_token_id, tok.eos_token_id, tok.pad_token_id):
        tok_side |= set(as_ids(x))
    model_side = set()
    for src in (cfg, gen):
        for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
            model_side |= set(as_ids(getattr(src, key, None) if src is not None else None))
    problem = None
    if any(not 0 <= i < V for i in model_side):
        problem = ⟦f"config or generation_config ids outside 0..{V - 1}: {sorted(i for i in model_side if not 0 <= i < V)}"¦f"config 或 generation_config 的編號超出 0 到 {V - 1}：{sorted(i for i in model_side if not 0 <= i < V)}"⟧
    dropped = sorted(i for i in tok_side if not 0 <= i < V)
    banned = sorted({i for i in tok_side | model_side if 0 <= i < V} | set(range(len(tok), V)))
    assert None not in banned and all(0 <= i < V for i in banned)
    return banned, dropped, problem


def pad_id(tok, cfg, gen, banned):
    """Rule 4: pad of tokenizer, config, generation_config; then their eos; else the smallest banned id."""
    for src, key in ((tok, "pad_token_id"), (cfg, "pad_token_id"), (gen, "pad_token_id"),
                     (tok, "eos_token_id"), (cfg, "eos_token_id"), (gen, "eos_token_id")):
        v = as_ids(getattr(src, key, None) if src is not None else None)
        if v:
            return v[0]
    return min(banned)


def tokenizer_hash(name, revision):
    """Tokenizer grouping: vocabulary, merges, normalizer, pre-tokenizer and decoder of the repository's tokenizer.json
    (the file itself: transformers rebuilds some tokenizers in memory with an equivalent but differently written setup)."""
    j = json.load(open(hf_hub_download(name, "tokenizer.json", revision=revision, token=HF_TOKEN), encoding="utf-8"))
    core = dict(vocab=j["model"].get("vocab"), merges=j["model"].get("merges"), normalizer=j.get("normalizer"),
                pre_tokenizer=j.get("pre_tokenizer"), decoder=j.get("decoder"))
    return hashlib.sha256(canon(core).encode()).hexdigest()[:12]


def dec(tok, ids):
    return tok.decode(ids, clean_up_tokenization_spaces=False)


def split_text(tok, p, q):
    """Decode prefix and continuation together, then cut at the longest prefix p[:j] whose text is complete and starts
    the whole text (j = 0 always does). The same function for human and machine continuations."""
    full = dec(tok, p + q)
    for j in range(len(p), -1, -1):
        pre = dec(tok, p[:j])
        if not pre.endswith("�") and full.startswith(pre):
            return j, pre, full[len(pre):]
    raise AssertionError("unreachable: j = 0 always matches")


def text_input(tok, start, p, q):
    """The only text-path scoring input: split_text, start tokens + re-encoded whole text, cut to 512 tokens, mask start k
    (at least 1). Returns (ids, k, cut, j)."""
    j, pre, cont = split_text(tok, p, q)
    ids = start + tok(pre + cont, add_special_tokens=False)["input_ids"]
    cut = max(0, len(ids) - MAX_TOK)
    ids = ids[:MAX_TOK]
    k = max(1, len(start) + len(tok(pre, add_special_tokens=False)["input_ids"]))
    assert k < len(ids) and ids[:len(start)] == start
    return ids, k, cut, j


def encode_human(tok, text):
    ids = tok(text, add_special_tokens=False, truncation=True, max_length=MAX_RAW)["input_ids"]
    return ids[:PREFIX], ids[PREFIX:]


def text_row(tok, start, p, q):
    ids, k, cut, j = text_input(tok, start, p, q)
    return dict(ids=ids, k=k, cut=cut, j=j, len_p=len(p), same_prefix=ids[:k] == start + p, same_cont=ids[k:] == q)


def human_inputs(tok, start, ids):
    return {i: text_row(tok, start, *encode_human(tok, HUMANS[i]["text"])) for i in ids}


def prompt_items(tok, ids):
    """E prompts: prompt = p[:j] (j from split_text), target = the human token count minus j; sorted by target."""
    items = []
    for i in ids:
        p, q = encode_human(tok, HUMANS[i]["text"])
        j = split_text(tok, p, q)[0]
        items.append(dict(id=i, j=j, prompt=p[:j], target=len(p) + len(q) - j))
    return sorted(items, key=lambda b: (b["target"], b["id"]))


def steps_per_temperature(items):
    return sum(max(b["target"] for b in items[s:s + BS]) for s in range(0, len(items), BS))


def machine_inputs(tok, start, gen_rows, path):
    """Expected scoring input per machine text: text path from text_input, id path = start + prompt + sampled ids."""
    out = {}
    for r in gen_rows:
        if path == "text":
            out[r["id"]] = text_row(tok, start, r["prompt"], r["ids"])
        else:
            ids = start + r["prompt"] + r["ids"]
            out[r["id"]] = dict(ids=ids, k=max(1, len(start) + len(r["prompt"])), cut=0, j=len(r["prompt"]),
                                len_p=len(r["prompt"]), same_prefix=True, same_cont=True)
        assert out[r["id"]]["ids"][:len(start)] == start   # ⟪the same start tokens as generation, on both paths¦兩條路徑的開頭記號都和生成時相同⟫
    return out


# ---- ⟪models, generation, scoring¦模型、生成、評分⟫
class FallbackLog(logging.Handler):
    """Keeps transformers' 'falling back to its reference PyTorch implementation' warnings (Mamba, Zamba2)."""
    def __init__(self):
        super().__init__(); self.msgs = []

    def emit(self, record):
        m = record.getMessage()
        if "falling back" in m.lower():
            self.msgs.append(m)


FALLBACK = FallbackLog()
logging.getLogger("transformers").addHandler(FALLBACK)
FALLBACK_BY_MODEL = {}


def reset_fallback():
    """transformers prints each warning once per process (warning_once); forget that before every model, so each model's
    own fallback warnings are kept (Mamba and Zamba2 share some of them)."""
    logging.Logger.warning_once.cache_clear(); FALLBACK.msgs.clear()


def load_model(name, revision):
    reset_fallback(); t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=DTYPE, token=HF_TOKEN).to(DEVICE).eval()
    model.generation_config = GenerationConfig()   # ⟪decoding is fully set by the plan; checkpoint defaults are not used¦解碼設定完全照計畫；不用模型檔自帶的預設值⟫
    return model, time.time() - t0


def free_model(name):
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    if CLEAN_CACHE and HF_HOME:
        shutil.rmtree(f"{HF_HOME}/hub/models--{name.replace('/', '--')}", ignore_errors=True)


def sampling_config(model, temp, new, banned, pad):
    """Plain softmax sampling: top-p 1.0, top-k off, no repetition penalty, banned ids suppressed, exact length. The
    config generate() will really use is checked, so that no library or checkpoint default can slip in."""
    cfg = GenerationConfig(do_sample=True, temperature=temp, top_k=0, top_p=1.0, typical_p=1.0, repetition_penalty=1.0,
                           no_repeat_ngram_size=0, num_beams=1, max_new_tokens=new, min_new_tokens=new,
                           suppress_tokens=list(banned), pad_token_id=pad, renormalize_logits=False)
    used, _ = model._prepare_generation_config(cfg)
    neutral = dict(do_sample=(True,), temperature=(temp,), top_k=(0,), top_p=(1.0,), typical_p=(None, 1.0), min_p=(None,),
                   top_h=(None,), epsilon_cutoff=(None, 0.0), eta_cutoff=(None, 0.0), repetition_penalty=(1.0,),
                   encoder_repetition_penalty=(None, 1.0), no_repeat_ngram_size=(None, 0), encoder_no_repeat_ngram_size=(None, 0),
                   bad_words_ids=(None,), sequence_bias=(None,), guidance_scale=(None, 1.0), forced_bos_token_id=(None,),
                   forced_eos_token_id=(None,), begin_suppress_tokens=(None,), exponential_decay_length_penalty=(None,),
                   remove_invalid_values=(None, False), renormalize_logits=(None, False), watermarking_config=(None,),
                   num_beams=(1,), max_new_tokens=(new,), min_new_tokens=(new,))
    off = {k: getattr(used, k, None) for k, ok in neutral.items() if getattr(used, k, None) not in ok}
    assert not off, ⟦f"generation settings differ from the plan: {off}"¦f"生成設定和計畫不同：{off}"⟧
    return cfg


@torch.no_grad()
def generate(model, items, start, banned, pad, temp):
    """items sorted by target, batches of 32, seed 42 per call. A batch is split by prompt length, so no PAD is ever
    needed during generation."""
    torch.manual_seed(SEED)
    out, bset = {}, set(banned)
    for s in range(0, len(items), BS):
        batch = items[s:s + BS]
        for L in sorted({len(b["prompt"]) for b in batch}):
            sub = [b for b in batch if len(b["prompt"]) == L]
            x = torch.tensor([start + b["prompt"] for b in sub], dtype=torch.long)
            assert x.shape[1] >= 1
            new = max(b["target"] for b in sub)
            y = model.generate(input_ids=x.to(DEVICE), attention_mask=torch.ones_like(x).to(DEVICE),
                               generation_config=sampling_config(model, temp, new, banned, pad))
            for b, row in zip(sub, y[:, x.shape[1]:].cpu().tolist()):
                ids = row[:b["target"]]
                assert len(ids) == b["target"], ("wrong length", len(ids), b["target"])
                assert not (set(ids) & bset), ⟦"a banned id was generated"¦"生成了禁止清單裡的編號"⟧
                out[b["id"]] = ids
    return out


@torch.no_grad()
def score_inputs(model, pad, enc):
    """enc: [(key, ids, k)]. Right-padded batches of 16 sorted by length; per text the float32 array (4 rows: -log p of
    the token, entropy, variance of log p, top-10 mass) for the positions from k on. Every text is scored."""
    order = sorted(range(len(enc)), key=lambda i: (len(enc[i][1]), i)); out = {}
    for s in range(0, len(order), SBS):
        chunk = [enc[i] for i in order[s:s + SBS]]; L = max(len(e[1]) for e in chunk)
        x = torch.full((len(chunk), L), pad, dtype=torch.long); att = torch.zeros((len(chunk), L), dtype=torch.long)
        for i, (_, ids, _) in enumerate(chunk):
            x[i, :len(ids)] = torch.tensor(ids); att[i, :len(ids)] = 1
        assert bool((x[att == 0] == pad).all()) and all(x[i, :len(e[1])].tolist() == list(e[1]) for i, e in enumerate(chunk)), \
            ⟦"PAD must appear only where the mask is 0"¦"PAD 只能出現在遮罩為 0 的位置"⟧
        logits = model(input_ids=x.to(DEVICE), attention_mask=att.to(DEVICE)).logits
        for i, (key, ids, k) in enumerate(chunk):
            n = len(ids)
            lp = torch.log_softmax(logits[i, :n - 1].float(), -1)
            tgt = torch.tensor(ids[1:], device=lp.device)
            p = lp.exp(); e1 = (p * lp).sum(-1)
            arr = torch.stack([-lp.gather(1, tgt[:, None])[:, 0], -e1, (p * lp * lp).sum(-1) - e1 ** 2, p.topk(10, -1).values.sum(-1)])
            out[key] = arr[:, k - 1:].cpu().numpy().astype(np.float32)
    assert len(out) == len(enc), ⟦"every text must be scored"¦"每一篇都要評分"⟧
    return out


def file_config(e, kind, path, temp):
    """Everything a stored file depends on; its hash is written into the file and checked on every resume."""
    return dict(schema=SCHEMA, kind=kind, path=path, model=e["name"], revision=e["revision"], start=e["start"],
                banned_sha256=e["banned_sha256"], pad=e["pad"], temperature=temp, seed=SEED, dtype=str(DTYPE),
                prefix=PREFIX, max_tok=MAX_TOK, manifest_sha256=EXPECTED_MAN, packages=PINS, r1_commit=R1_COMMIT)


def save_scores(path, cfg, expected, arrs):
    rows = [dict(id=i, ids=np.asarray(x["ids"], np.int32), k=x["k"], cut=x["cut"], j=x["j"], len_p=x["len_p"],
                 same_prefix=bool(x["same_prefix"]), same_cont=bool(x["same_cont"]), arr=arrs[i]) for i, x in sorted(expected.items())]
    atomic_write(path, pickle.dumps(dict(schema=SCHEMA, kind=cfg["kind"], path=cfg["path"], model=cfg["model"], temperature=cfg["temperature"],
                                         config=cfg, config_hash=config_hash(cfg), written_utc=utc(), environment=ENV, rows=rows), protocol=4))


def check_scores(path, cfg, expected):
    """(ok, reason): readable, same config hash, the expected ids exactly once, each input equal to the expected one
    (so each length equals the path's target), and every score finite."""
    try:
        obj = pickle.load(open(path, "rb"))
        rows = obj["rows"]
    except Exception as err:
        return False, ⟦f"unreadable ({type(err).__name__})"¦f"讀不出來（{type(err).__name__}）"⟧
    if obj.get("config_hash") != config_hash(cfg) or config_hash(obj.get("config", {})) != obj.get("config_hash"):
        return False, ⟦"config hash differs"¦"設定雜湊不符"⟧
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        return False, ⟦"ids differ from the expected set or repeat"¦"編號集合不符或有重複"⟧
    for r in rows:
        x = expected[r["id"]]; arr = np.asarray(r["arr"])
        if np.asarray(r["ids"]).tolist() != x["ids"] or r["k"] != x["k"] or arr.shape != (4, len(x["ids"]) - x["k"]):
            return False, ⟦f"text {r['id']}: input or length differs from the target"¦f"第 {r['id']} 篇：輸入或長度和目標不符"⟧
        if not np.isfinite(arr).all():
            return False, ⟦f"text {r['id']}: non-finite scores"¦f"第 {r['id']} 篇：分數有非有限值"⟧
    return True, ""


def check_gen(path, cfg, items, banned):
    try:
        obj = json.load(open(path))
        rows = obj["rows"]
    except Exception as err:
        return False, ⟦f"unreadable ({type(err).__name__})"¦f"讀不出來（{type(err).__name__}）"⟧
    if obj.get("config_hash") != config_hash(cfg) or config_hash(obj.get("config", {})) != obj.get("config_hash"):
        return False, ⟦"config hash differs"¦"設定雜湊不符"⟧
    want = {b["id"]: b for b in items}; ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(want):
        return False, ⟦"ids differ from the expected set or repeat"¦"編號集合不符或有重複"⟧
    bset = set(banned)
    for r in rows:
        b = want[r["id"]]
        if r["prompt"] != b["prompt"] or r["j"] != b["j"] or r["target"] != b["target"] or len(r["ids"]) != b["target"]:
            return False, ⟦f"text {r['id']}: prompt or length differs from the target"¦f"第 {r['id']} 篇：提示或長度和目標不符"⟧
        if set(r["ids"]) & bset:
            return False, ⟦f"text {r['id']}: contains a banned id"¦f"第 {r['id']} 篇：含禁止清單裡的編號"⟧
    return True, ""


def valid_or_move(path, ok_reason):
    """True if the file exists and passed its check; a file that failed is moved to _invalid/."""
    ok, reason = ok_reason
    if not ok and os.path.exists(path):
        move_invalid(path, reason)
    return ok


def verify_frozen(e, tok, cfg, gen):
    """The start tokens, banned list and PAD registered at R1 must still come out of the same files."""
    start, prob = start_tokens(tok)
    banned, _, prob3 = banned_ids(tok, cfg, gen)
    assert prob is None and prob3 is None and start == e["start"], ⟦f"{e['name']}: start tokens differ from R1"¦f"{e['name']}：開頭記號和 R1 登錄不同"⟧
    assert banned == e["banned"] and sha256_list(banned) == e["banned_sha256"], ⟦f"{e['name']}: banned list differs from R1"¦f"{e['name']}：禁止清單和 R1 登錄不同"⟧
    assert pad_id(tok, cfg, gen, banned) == e["pad"], ⟦f"{e['name']}: PAD differs from R1"¦f"{e['name']}：PAD 和 R1 登錄不同"⟧


# ---- ⟪predictions (the same formulas as scripts/e12_replication_test.py)¦預測（和 scripts/e12_replication_test.py 相同的公式）⟫
def text_stats(arr):
    a = np.asarray(arr, dtype=np.float64)
    g = a[1] - a[0]
    sv = math.sqrt(a[2].sum())
    return g.mean(), a[2].mean(), g.sum() / sv, sv


def formula(c, V):
    return float(1 - np.mean(c) / np.mean(V))


def variant_b(fdg, sv):
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


def make_prediction(e, rows):
    st = {r["id"]: text_stats(r["arr"]) for r in rows}
    cP, VP, fP, sP = (np.array([st[i][k] for i in P_IDS]) for k in range(4))
    cE, VE = (np.array([st[i][k] for i in E_IDS]) for k in range(2))
    Ec = float(np.mean(cP)); vb, vb_status = variant_b(fP, sP)
    return dict(status="written", model=e["name"], revision=e["revision"], family=e["family"], tok_group=e["tok_group"],
                start=e["start"], banned_sha256=e["banned_sha256"], T_hat=formula(cP, VP), T_hat_E=formula(cE, VE),
                T_A=float(1 - np.mean(fP) / np.mean(sP)), variant_B=dict(value=vb, status=vb_status),
                baselines=dict(guess1=1.0, known_mean=KNOWN_MEAN, slope=1 - SLOPE_K * Ec, line=LINE_A - LINE_K * Ec),
                E_P_c=Ec, V_P=float(np.mean(VP)), E_E_c=float(np.mean(cE)), V_E=float(np.mean(VE)), n_P=len(P_IDS), n_E=len(E_IDS))


# ---- ⟪reruns: every attempt is logged; after the first run plus three reruns a model is given up¦重跑：每次嘗試都記錄；第一次加重跑 3 次仍失敗就放棄該模型⟫
def read_json(path, default):
    return json.load(open(path)) if os.path.exists(path) else default


def attempt_begin(stage, name):
    log = read_json(ATTEMPTS_FILE, {}); log.setdefault(stage, {}).setdefault(name, []).append(dict(start=utc()))
    write_json(ATTEMPTS_FILE, log)


def attempt_end(stage, name, error=None):
    log = read_json(ATTEMPTS_FILE, {}); log[stage][name][-1].update(end=utc(), ok=error is None, error=error)
    write_json(ATTEMPTS_FILE, log)


def failed_attempts(stage, name):
    return [a for a in read_json(ATTEMPTS_FILE, {}).get(stage, {}).get(name, []) if not a.get("ok")]


# ---- ⟪stage 0 checks (tokenizer only, then GPU with A-group prompts and random ids; no score is kept)¦第零階段的檢查（先只用 tokenizer，再用 A 組提示與隨機編號跑 GPU；不保存任何分數）⟫
def tokenizer_check(e, P_ids, E_ids, prompt_ids):
    rec = dict(no=e["no"], name=e["name"], family=e["family"], provisional_group=e["group"], utc=utc())
    try:
        rev = HfApi().model_info(e["name"], token=HF_TOKEN).sha
        tok = load_tokenizer(e["name"], rev); cfg, gen = model_meta(e["name"], rev); thash = tokenizer_hash(e["name"], rev)
    except Exception as err:
        return dict(rec, passed=False, load_error=f"{type(err).__name__}: {err}"[:300])
    V = vocab_size(cfg); rules = {}
    start, prob = start_tokens(tok)
    rules["2"] = dict(ok=prob is None, start=start, problem=prob)
    banned, dropped, prob = banned_ids(tok, cfg, gen)
    rules["3"] = dict(ok=prob is None, n=len(banned), sha256=sha256_list(banned), dropped_tokenizer_ids=dropped, problem=prob)
    pad = pad_id(tok, cfg, gen, banned)
    rules["4"] = dict(ok=0 <= pad < V, pad=pad, tokenizer_pad=tok.pad_token_id)
    rt, n_tok, n_scored, max_id = {}, [], [], 0
    for g, ids in (("P", P_ids), ("E", E_ids)):
        rows = human_inputs(tok, start or [], ids)
        same = sum(r["same_prefix"] and r["same_cont"] for r in rows.values())
        rt[g] = dict(n=len(ids), unchanged=int(same), share=same / len(ids), j_back=int(sum(r["j"] < r["len_p"] for r in rows.values())),
                     prefix_changed=int(sum(not r["same_prefix"] for r in rows.values())), continuation_changed=int(sum(not r["same_cont"] for r in rows.values())))
        n_tok += [r["len_p"] + len(encode_human(tok, HUMANS[i]["text"])[1]) for i, r in rows.items()]
        n_scored += [len(r["ids"]) - r["k"] for r in rows.values()]
        max_id = max([max_id] + [max(r["ids"]) for r in rows.values()])
    rules["5"] = dict(ok=all(v["share"] >= ROUNDTRIP_MIN for v in rt.values()), **rt)
    items = prompt_items(tok, prompt_ids)
    rules["6"] = dict(ok=min(n_tok) >= MIN_TOKENS and min(n_scored) >= MIN_SCORED and len(items) == len(prompt_ids) and max_id < V,
                      min_tokens=min(n_tok), min_scored=min(n_scored), n_prompts=len(items), max_id=int(max_id), vocab_size=V)
    return dict(rec, revision=rev, passed=all(r["ok"] for r in rules.values()), rules=rules, start=start, banned=banned,
                banned_sha256=sha256_list(banned), pad=pad, vocab_size=V, len_tokenizer=len(tok), tokenizer_hash=thash,
                S_m=steps_per_temperature(items), prompt_j_back=int(sum(b["j"] < PREFIX for b in items)))


def peak_memory():
    if DEVICE == "cuda":
        return torch.cuda.max_memory_allocated(), torch.cuda.get_device_properties(0).total_memory
    import psutil
    return psutil.Process().memory_info().rss, psutil.virtual_memory().total


def timed(fn):
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.time(); fn()
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return time.time() - t0


def gpu_check(e, rec):
    """Rule 8: generation check, per-step generation time, scoring time, peak memory and the projected minutes P_m."""
    G = GPU_CHECK; out = dict(name=e["name"], utc=utc()); model = None
    try:
        if DEVICE == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model, load_s = load_model(e["name"], rec["revision"])
        tok = load_tokenizer(e["name"], rec["revision"])
    except Exception as err:   # ⟪downloading or loading: a temporary cause (rule 1), retried by stage 0, never a rule-8 failure¦下載或載入：暫時性原因（規則 1），第零階段會重試，不算規則 8 失敗⟫
        del model; free_model(e["name"])
        return dict(out, load_error=f"{type(err).__name__}: {err}"[:300])
    try:
        start, banned, pad = rec["start"], rec["banned"], rec["pad"]
        items = [dict(id=i, j=PREFIX, prompt=tok(t, add_special_tokens=False)["input_ids"][:PREFIX], target=G["new_tokens"])
                 for i, t in enumerate(A_PROMPTS[:G["prompts"]])]
        gen_ok = len(generate(model, items, start, banned, pad, 1.0)) == len(items)
        rng = np.random.default_rng(SEED); allowed = np.setdiff1d(np.arange(rec["vocab_size"]), banned)
        ctx = [rng.choice(allowed, G["context"]).tolist() for _ in range(BS)]
        run = lambda n: generate(model, [dict(id=i, j=len(c), prompt=c, target=n) for i, c in enumerate(ctx)], start, banned, pad, 1.0)
        timed(lambda: run(2))   # ⟪warm-up¦暖機⟫
        t_step = (timed(lambda: run(8 + G["steps"])) - timed(lambda: run(8))) / G["steps"]
        enc = [(i, start + rng.choice(allowed, G["score_len"] - len(start)).tolist(), 1) for i in range(G["score_batch"])]
        t_score = statistics.median(timed(lambda: score_inputs(model, pad, enc)) for _ in range(G["repeats"]))
        peak, total = peak_memory()
        n_prompts = len(PROMPT_IDS)
        P_m = (len(TEMPS) * rec["S_m"] * max(t_step, 0.0) + t_score * (len(TEMPS) * 2 * math.ceil(n_prompts / SBS) + math.ceil(N_H / SBS))
               + 2 * load_s) / 60
        out.update(passed=gen_ok and P_m <= PM_MAX_MIN and peak <= PEAK_MAX * total, generation_ok=gen_ok, load_seconds=load_s,
                   t_step=t_step, t_score=t_score, peak_bytes=int(peak), total_bytes=int(total), peak_share=peak / total, P_m=P_m,
                   fallback_warnings=list(FALLBACK.msgs))
    except Exception as err:   # ⟪the model loaded but cannot run the checks (for example out of memory): rule 8¦模型載入了，但跑不完檢查（例如記憶體不足）：規則 8⟫
        out.update(passed=False, error=f"{type(err).__name__}: {err}"[:300])
    finally:
        del model; free_model(e["name"])
    return out


def apply_rules(entries, cand):
    """Replacement (one candidate, at most once) and rule 9 (total P_m within the budget), in that order."""
    final = [x for x in entries if x["passed"]]
    changes = [dict(model=x["name"], action="removed", reason=x["reason"]) for x in entries if not x["passed"]]
    used = False
    if len(final) < len(entries) and cand is not None and cand["passed"]:
        final.append(cand); used = True
        changes.append(dict(model=cand["name"], action="added", reason="replacement for a technical failure"))
    while final and sum(x["P_m"] for x in final) > TOTAL_MAX_H * 60:
        worst = max(final, key=lambda x: (x["P_m"], x["no"]))
        final.remove(worst)
        if cand is not None and cand["passed"] and not used:
            final.append(cand); used = True
            changes.append(dict(model=worst["name"], action="replaced", reason=f"rule 9: largest P_m {worst['P_m']:.1f} min", by=cand["name"]))
        else:
            changes.append(dict(model=worst["name"], action="removed", reason=f"rule 9: largest P_m {worst['P_m']:.1f} min"))
    final = sorted(final, key=lambda x: x["no"])
    return final, changes, len({x["family"] for x in final}) < MIN_FAMILIES


def stage0_entry(e, t, g):
    """One model's stage-0 outcome: None when loading has to be retried (rule 1: temporary causes are retried until they
    succeed and never count), else dict(passed, reason, P_m). Only DECLARED_LOAD_FAILURES fails rule 1."""
    base = dict(no=e["no"], name=e["name"], family=e["family"], P_m=float("inf"))
    if e["name"] in DECLARED_LOAD_FAILURES:
        return dict(base, passed=False, reason="rule 1: " + DECLARED_LOAD_FAILURES[e["name"]])
    if "load_error" in t:
        return None
    if not t["passed"]:
        return dict(base, passed=False, reason="; ".join(f"rule {k}" for k, v in t["rules"].items() if not v["ok"]))
    if g is None or "load_error" in g:
        return None
    if not g["passed"]:
        why = g.get("error") or "; ".join(x for x, failed in (("generation check", not g.get("generation_ok")),
                                                            (f"P_m {g.get('P_m', 0):.0f} > {PM_MAX_MIN} min", g.get("P_m", 0) > PM_MAX_MIN),
                                                            (f"peak memory {g.get('peak_share', 0):.0%}", g.get("peak_share", 0) > PEAK_MAX)) if failed)
        return dict(base, passed=False, reason="rule 8: " + why, P_m=g.get("P_m", float("inf")))
    return dict(base, passed=True, reason="", P_m=g["P_m"])


def tokenizer_groups(recs):
    """Freeze the tokenizer groups from the tokenizer.json hashes. A provisional label is kept when its models share one
    hash and no other hash carries it; any other hash gets the next free G number. Returns ({model: label}, notes)."""
    by_hash = {}
    for r in sorted(recs, key=lambda r: r["no"]):
        by_hash.setdefault(r["tokenizer_hash"], []).append(r)
    nxt = 1 + max([int(r["provisional_group"][1:]) for r in recs if r["provisional_group"] and r["provisional_group"][1:].isdigit()], default=0)
    labels, notes = {}, []
    for h, rs in by_hash.items():
        prov = {r["provisional_group"] for r in rs}
        label = next(iter(prov)) if len(prov) == 1 else None
        if label is None or any(r["provisional_group"] == label for h2, rs2 in by_hash.items() if h2 != h for r in rs2):
            notes.append(dict(models=[r["name"] for r in rs], provisional=sorted(str(p) for p in prov), frozen=f"G{nxt}", hash=h))
            label, nxt = f"G{nxt}", nxt + 1
        for r in rs:
            labels[r["name"]] = label
    return labels, notes


def human_file(e):
    return f"{OUT}/humans_{tag_of(e['name'])}.pkl"


def stage1_model(e):
    """'done', 'failed: why' (given up after the first run and three reruns) or 'retry: why'. A model whose human scores
    pass their check is done whatever its earlier attempts were; loading the tokenizer for that check is not an attempt."""
    name, path = e["name"], human_file(e)
    frozen = read_json(PRED_FILE, None)          # ⟪predictions.json exists: stage 1 only re-checks, it never scores again¦predictions.json 已存在：第一階段只核對，不再評分⟫
    if frozen is not None and frozen["models"][name]["status"] != "written":
        return "failed: " + frozen["models"][name]["reason"]
    if frozen is None and name in DECLARED_STAGE1_FAILURES:
        return "failed: " + ⟦"declared by the host: "¦"主持人宣告："⟧ + DECLARED_STAGE1_FAILURES[name]
    try:
        tok = load_tokenizer(name, e["revision"]); cfg, gen = model_meta(name, e["revision"])
    except Exception as err:   # ⟪temporary (rule 1): retried, not counted¦暫時性原因（規則 1）：重試，不計次數⟫
        return f"retry: {type(err).__name__}: {err}"[:300]
    verify_frozen(e, tok, cfg, gen)
    exp = human_inputs(tok, e["start"], P_IDS + E_IDS); fcfg = file_config(e, "human", "text", None)
    ok, why = check_scores(path, fcfg, exp) if os.path.exists(path) else (False, ⟦"missing"¦"不存在"⟧)
    if ok:
        return "done"
    if frozen is not None:   # ⟪the predictions were computed from this file: never rescore it, missing or not¦預測是用這個檔算的：不論遺失或損壞，絕不重評⟫
        raise StageStop(⟦f"{os.path.basename(path)} fails its check ({why}) but predictions.json already exists: stop for review."¦f"{os.path.basename(path)} 沒有通過檢查（{why}），但 predictions.json 已經存在：停下來檢查。"⟧)
    if os.path.exists(path):
        move_invalid(path, why)
    if len(failed_attempts("stage1", name)) >= MAX_ATTEMPTS:
        return "failed: " + str(failed_attempts("stage1", name)[-1].get("error") or ⟦"interrupted"¦"中斷"⟧)
    attempt_begin("stage1", name); model = None
    try:
        model, _ = load_model(name, e["revision"])
        save_scores(path, fcfg, exp, score_inputs(model, e["pad"], [(i, x["ids"], x["k"]) for i, x in exp.items()]))
        ok, why = check_scores(path, fcfg, exp); assert ok, why
        attempt_end("stage1", name)
        return "done"
    except Exception as err:
        attempt_end("stage1", name, f"{type(err).__name__}: {err}"[:300])
        return f"retry: {type(err).__name__}: {err}"[:300]
    finally:
        del model; free_model(name)


def stage2_files(e, t):
    tag = tag_of(e["name"])
    return (f"{OUT}/gen_{tag}_T{fmt_t(t)}.json", f"{OUT}/scores_text_{tag}_T{fmt_t(t)}.pkl", f"{OUT}/scores_ids_{tag}_T{fmt_t(t)}.pkl")


def stage2_paths(e):
    return [p for t in TEMPS for p in stage2_files(e, t) if os.path.exists(p)]


def file_environment(path):
    """The environment a stored file was made in; None when the file cannot be read (its own check moves it away)."""
    try:
        obj = json.load(open(path)) if path.endswith(".json") else pickle.load(open(path, "rb"))
        return canon(obj.get("environment"))
    except Exception:
        return None


def stage2_complete(e, tok, items):
    """True when every generation and scoring file of the model passes its check (a failed file is moved to _invalid/) and
    all of them were made in one environment."""
    if len({file_environment(p) for p in stage2_paths(e)} - {None}) > 1:
        return False
    for t in TEMPS:
        gpath, tpath, ipath = stage2_files(e, t)
        if not (os.path.exists(gpath) and valid_or_move(gpath, check_gen(gpath, file_config(e, "gen", "gen", float(t)), items, e["banned"]))):
            return False
        rows = json.load(open(gpath))["rows"]
        for path, kind in ((tpath, "text"), (ipath, "ids")):
            if not (os.path.exists(path) and valid_or_move(path, check_scores(path, file_config(e, "machine", kind, float(t)),
                                                                               machine_inputs(tok, e["start"], rows, kind)))):
                return False
    return True


def stage2_model(e):
    """'done', 'failed: why' or 'retry: why'. A model whose files are all complete is done whatever its attempts or the date;
    the rerun limit and the 14-day limit only stop new work (e12 applies the 14-day rule to the files' write times)."""
    name = e["name"]
    if PREDS["models"][name]["status"] != "written":
        return "failed: " + ⟦"prediction not written (state 3)"¦"預測未寫出（狀態③）"⟧
    try:
        tok = load_tokenizer(name, e["revision"]); cfg, gen = model_meta(name, e["revision"])
    except Exception as err:   # ⟪temporary (rule 1): retried, not counted¦暫時性原因（規則 1）：重試，不計次數⟫
        return f"retry: {type(err).__name__}: {err}"[:300]
    verify_frozen(e, tok, cfg, gen)
    items = prompt_items(tok, PROMPT_IDS)
    if stage2_complete(e, tok, items):
        return "done"
    if len(failed_attempts("stage2", name)) >= MAX_ATTEMPTS or time.time() > DEADLINE:
        return "failed: " + ⟦"rerun limit or 14-day limit reached (state 3)"¦"超過重跑上限或 14 天期限（狀態③）"⟧
    if any(file_environment(p) not in (None, canon(ENV)) for p in stage2_paths(e)):   # ⟪plan 5.4: the environment changed half-way: redo the model¦計畫 5.4：環境在做到一半時改變：整個模型從頭重做⟫
        for p in stage2_paths(e):
            move_invalid(p, ⟦"the environment changed while this model was half done: redone from scratch"¦"這個模型做到一半時執行環境改變：從頭重做"⟧)
    attempt_begin("stage2", name); model = None
    try:
        for t in TEMPS:
            gpath, tpath, ipath = stage2_files(e, t)
            gcfg = file_config(e, "gen", "gen", float(t))
            if not (os.path.exists(gpath) and valid_or_move(gpath, check_gen(gpath, gcfg, items, e["banned"]))):
                if model is None:
                    model, _ = load_model(name, e["revision"])
                t0 = time.time(); sampled = generate(model, items, e["start"], e["banned"], e["pad"], float(t))
                rows = [dict(id=b["id"], j=b["j"], prompt=b["prompt"], target=b["target"], ids=sampled[b["id"]]) for b in items]
                write_json(gpath, dict(schema=SCHEMA, kind="gen", model=name, temperature=float(t), config=gcfg, config_hash=config_hash(gcfg),
                                       written_utc=utc(), environment=ENV, rows=rows))
                ok, why = check_gen(gpath, gcfg, items, e["banned"]); assert ok, why
                print(f"  {tag_of(name)} T={fmt_t(t)}: {len(rows)} ({(time.time() - t0) / 60:.1f} min)", flush=True)
            rows = json.load(open(gpath))["rows"]
            for path, kind in ((tpath, "text"), (ipath, "ids")):
                scfg = file_config(e, "machine", kind, float(t)); exp = machine_inputs(tok, e["start"], rows, kind)
                if not (os.path.exists(path) and valid_or_move(path, check_scores(path, scfg, exp))):
                    if model is None:
                        model, _ = load_model(name, e["revision"])
                    save_scores(path, scfg, exp, score_inputs(model, e["pad"], [(i, x["ids"], x["k"]) for i, x in exp.items()]))
                    ok, why = check_scores(path, scfg, exp); assert ok, why
        FALLBACK_BY_MODEL[name] = list(FALLBACK.msgs)
        attempt_end("stage2", name)
        return "done"
    except Exception as err:
        attempt_end("stage2", name, f"{type(err).__name__}: {err}"[:300])
        return f"retry: {type(err).__name__}: {err}"[:300]
    finally:
        del model; free_model(name)


def check_environment(env):
    """Colab's preinstalled versions are recorded at the first run (stage 0) and must stay the same. A session with other
    versions stops before any file is read or written, so nothing already done is redone."""
    if not os.path.exists(ENV_FILE):
        write_json(ENV_FILE, env)
    first = read_json(ENV_FILE, None)
    assert first == env, ⟦f"Colab's preinstalled versions changed: first run {first}, now {env}. Stop for review (plan 5.4: revision note, then check_registration.py env-update); do not redo any file."¦f"Colab 預裝的版本變了：第一次執行是 {first}，現在是 {env}。請停下來檢查（計畫 5.4：先寫修訂紀錄，再用 check_registration.py env-update 更新）；不要重做任何檔案。"⟧
    return first


ENV = check_environment(dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, scipy=scipy.__version__,
                             cuda=torch.version.cuda))
R1_COMMIT = None   # ⟪set in stage 1 from r1_models.json¦第一階段從 r1_models.json 讀入⟫
print(⟦f"{len(MODELS)} models + candidate {CANDIDATE['name'] if CANDIDATE else None}; temperatures {TEMPS}; {DTYPE}"¦f"{len(MODELS)} 個模型加候補 {CANDIDATE['name'] if CANDIDATE else None}；溫度 {TEMPS}；{DTYPE}"⟧)
''')

# ================================================================== stage 0a
code(r'''
# ⟪Stage 0a: tokenizer checks (rules 2-6) for the 12 models and Zamba2; record the current revision of each.¦第零階段 0a：tokenizer 檢查（規則 2 到 6），12 個模型加 Zamba2；記錄每個模型當下的 revision。⟫
device_check()
STAGE0 = not os.path.exists(R1_FILE) and not os.path.exists(STAGE0_FILE)
TOKREC, GPUREC = {}, {}
if os.path.exists(R1_FILE):
    print(⟦"R1 is registered (r1_models.json exists): stage 0 is skipped."¦"R1 已登錄（r1_models.json 存在）：略過第零階段。"⟧)
elif not STAGE0:
    print(⟦"stage0_record.json is already written (it is written only once): the checks are not run again."¦"stage0_record.json 已經寫出（只寫一次）：不再重做檢查。"⟧)
else:
    os.makedirs(f"{OUT}/stage0", exist_ok=True)
    for e in MODELS + ([CANDIDATE] if CANDIDATE else []):
        path = f"{OUT}/stage0/tok_{tag_of(e['name'])}.json"
        rec = read_json(path, None)
        if rec is None or "load_error" in rec:   # ⟪a failed download is temporary (rule 1): try again¦下載失敗是暫時性原因（規則 1）：再試一次⟫
            rec = tokenizer_check(e, P_IDS, E_IDS, PROMPT_IDS)
            write_json(path, rec)
        TOKREC[e["name"]] = rec
        print(f"{tag_of(e['name'])}: {'OK' if rec['passed'] else 'FAIL'}", rec.get("load_error") or
              {k: v["ok"] for k, v in rec["rules"].items()}, rec.get("start"), rec.get("revision", "")[:12], flush=True)
''')

# ================================================================== stage 0b
code(r'''
# ⟪Stage 0b: GPU check (rule 8): generation, timing, peak memory and P_m, for every model that passed 0a.¦第零階段 0b：GPU 檢查（規則 8）：生成檢查、計時、峰值記憶體與 P_m，只做通過 0a 的模型。⟫
device_check()
if STAGE0:
    for e in MODELS + ([CANDIDATE] if CANDIDATE else []):
        rec = TOKREC[e["name"]]
        if not rec.get("passed"):
            continue
        path = f"{OUT}/stage0/gpu_{tag_of(e['name'])}.json"
        g = read_json(path, None)
        if g is None or "load_error" in g:   # ⟪a failed download is temporary (rule 1): try again¦下載失敗是暫時性原因（規則 1）：再試一次⟫
            g = gpu_check(e, rec)
            write_json(path, g)
        GPUREC[e["name"]] = g
        print(f"{tag_of(e['name'])}: {'OK' if g.get('passed') else 'FAIL'}", g.get("load_error") or g.get("error") or
              f"P_m {g['P_m']:.1f} min, peak {g['peak_share']:.0%}, t_step {g['t_step'] * 1000:.1f} ms", flush=True)
''')

# ================================================================== stage 0c
code(r'''
# ⟪Stage 0c: technical failures, replacement and rule 9; write stage0_record.json once (no scores, no generated text), then stop.¦第零階段 0c：技術失敗、遞補與規則 9；stage0_record.json 只寫一次（不含任何分數或生成文字），然後停止。⟫
device_check()
if STAGE0:
    every = MODELS + ([CANDIDATE] if CANDIDATE else [])
    results = {e["name"]: stage0_entry(e, TOKREC[e["name"]], GPUREC.get(e["name"])) for e in every}
    retry = [n for n, r in results.items() if r is None]
    if retry:
        raise StageDone(⟦f"Could not load {retry} (temporary causes are not failures): run stage 0 again later; nothing was recorded."¦f"{retry} 無法載入（暫時性原因不算技術失敗）：請稍後重跑第零階段；這次沒有寫出任何紀錄。"⟧)
    entries = [results[e["name"]] for e in MODELS]
    final, changes, stop = apply_rules(entries, results[CANDIDATE["name"]] if CANDIDATE else None)
    labels, group_notes = tokenizer_groups([TOKREC[e["name"]] for e in every if TOKREC[e["name"]]["passed"]])
    info = {e["name"]: e for e in every}
    draft = [dict(no=x["no"], name=x["name"], revision=TOKREC[x["name"]]["revision"], family=x["family"], tok_group=labels[x["name"]],
                  tokenizer_hash=TOKREC[x["name"]]["tokenizer_hash"], lang_group=info[x["name"]]["lang"],
                  strip_group="strip" if info[x["name"]]["strip"] else "keep", start=TOKREC[x["name"]]["start"],
                  banned=TOKREC[x["name"]]["banned"], banned_sha256=TOKREC[x["name"]]["banned_sha256"], pad=TOKREC[x["name"]]["pad"]) for x in final]
    write_json(STAGE0_FILE, dict(utc=utc(), dryrun=DRYRUN, gpu=GPU_NAME, versions=VERSIONS, environment=ENV, manifest_sha256=EXPECTED_MAN,
                                 tokenizer_checks=TOKREC, gpu_checks=GPUREC, entries=entries, final=[x["name"] for x in final], changes=changes,
                                 P_m_total_min=sum(x["P_m"] for x in final), n_families=len({x["family"] for x in final}), stop_for_boss=stop,
                                 tokenizer_groups=labels, tokenizer_group_notes=group_notes, r1_models_draft=draft))
if not os.path.exists(R1_FILE):
    REC0, SHA0 = read_json(STAGE0_FILE, {}), sha256_file(STAGE0_FILE)
    flush_drive()
    print(⟦f"Final list: {REC0['final']}\nChanges: {REC0['changes']}\nTotal P_m {REC0['P_m_total_min']:.0f} min; stage0_record.json SHA-256 {SHA0}"¦f"最終名單：{REC0['final']}\n變動：{REC0['changes']}\nP_m 總和 {REC0['P_m_total_min']:.0f} 分鐘；stage0_record.json 的 SHA-256 {SHA0}"⟧)
    if REC0["stop_for_boss"]:
        raise StageDone(⟦"Fewer than 8 families are left: stop and ask the boss."¦"家族少於 8 個：停止，請老闆決定。"⟧)
    raise StageDone(⟦"Stage 0 is done. Next: register R1 and write r1_models.json; then Run All again."¦"第零階段完成。下一步：完成 R1 登錄並寫出 r1_models.json，之後再按 Run All。"⟧)
''')

# ================================================================== stage 1
code(r'''
# ⟪Stage 1: score the P and E human texts (text path), write predictions.json, then stop (no text is generated before R2).¦第一階段：P、E 人類文章評分（文字路徑），寫出 predictions.json，然後停止（R2 之前不生成任何文字）。⟫
device_check()
assert os.path.exists(R1_FILE), ⟦"r1_models.json is missing: R1 is not registered yet."¦"缺少 r1_models.json：R1 還沒登錄。"⟧
R1_SHA = sha256_file(R1_FILE)
R1 = json.load(open(R1_FILE)); R1_COMMIT = R1["r1_commit"]; ENTRIES = R1["models"]
print(⟦f"r1_models.json SHA-256: {R1_SHA}; registration {R1.get('r1_tag')}, commit {R1.get('r1_commit')}"¦f"r1_models.json 的 SHA-256：{R1_SHA}；登錄 {R1.get('r1_tag')}，commit {R1.get('r1_commit')}"⟧)
RUN_CONFIG = json.loads(json.dumps(dict(dryrun=DRYRUN, temps=list(TEMPS), P_ids=P_IDS, E_ids=E_IDS, A_ids=A_IDS, B_ids=B_IDS, quota=QUOTA)))
if os.path.exists(GO_FILE):
    print(⟦"stage2_go.json exists: stage 1 is registered (R2) and is skipped."¦"stage2_go.json 已存在：第一階段已登錄（R2），略過。"⟧)
else:
    old_cfg = read_json(RUNCFG_FILE, None)
    assert old_cfg in (None, RUN_CONFIG), ⟦"run_config.json differs from this run's settings"¦"run_config.json 和這次的設定不同"⟧
    if old_cfg is None:
        write_json(RUNCFG_FILE, RUN_CONFIG)
    status = {}
    for e in ENTRIES:
        t0 = time.time(); status[e["name"]] = stage1_model(e)
        print(f"{tag_of(e['name'])}: {status[e['name']]} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    retry = {n: s for n, s in status.items() if s.startswith("retry")}
    if retry:
        raise StageDone(⟦f"Some models did not finish; Run All again to retry them: {retry}"¦f"有模型沒有完成，請再按 Run All 重跑：{retry}"⟧)
    preds = dict(r1_tag=R1.get("r1_tag"), r1_commit=R1_COMMIT, r1_models_sha256=R1_SHA, manifest_sha256=EXPECTED_MAN, dryrun=DRYRUN,
                 models={})
    for e in ENTRIES:
        if status[e["name"]] == "done":
            p = make_prediction(e, pickle.load(open(human_file(e), "rb"))["rows"])
            p["written_at_utc"] = utc()
        else:
            p = dict(status="not_written", model=e["name"], reason=status[e["name"]][len("failed: "):])
        preds["models"][e["name"]] = p
    if os.path.exists(PRED_FILE):   # ⟪a rerun must reproduce the stored file exactly (write times aside) and never overwrite it¦重跑必須算出和存檔完全相同的內容（寫入時間除外），不覆寫⟫
        old = json.load(open(PRED_FILE))
        strip_t = lambda d: {k: ({n: {a: b for a, b in m.items() if a != "written_at_utc"} for n, m in v.items()} if k == "models" else v)
                             for k, v in d.items()}
        assert strip_t(old) == strip_t(json.loads(json.dumps(preds))), ⟦"predictions.json differs from the recomputed predictions"¦"predictions.json 和重算的預測不同"⟧
    else:
        write_json(PRED_FILE, preds)
    PRED_SHA = sha256_file(PRED_FILE)
    flush_drive()
    print(⟦f"predictions.json SHA-256: {PRED_SHA}"¦f"predictions.json 的 SHA-256：{PRED_SHA}"⟧)
    raise StageDone(⟦"Stage 1 is done. Next: check the synced hash and register R2; stage 2 waits for stage2_go.json."¦"第一階段完成。下一步：核對同步後的雜湊並完成 R2；第二階段要等 stage2_go.json。"⟧)
''')

# ================================================================== stage 2 gate
code(r'''
# ⟪Stage 2 gate: predictions.json must match the SHA-256 in stage2_go.json (written after R2 and its time proof).¦第二階段閘門：predictions.json 必須和 stage2_go.json 裡的 SHA-256 相同（stage2_go.json 在 R2 與時間證明完成後寫入）。⟫
device_check()
if not os.path.exists(GO_FILE):
    raise StageDone(⟦"stage2_go.json is missing: stage 2 cannot start yet."¦"缺少 stage2_go.json：第二階段還不能開始。"⟧)
GO = json.load(open(GO_FILE))
missing = {"predictions_sha256", "r2_tag", "r2_commit", "ots_block_height", "ots_block_time"} - set(GO)
assert not missing, ⟦f"stage2_go.json lacks {missing}"¦f"stage2_go.json 缺少 {missing}"⟧
PRED_SHA = sha256_file(PRED_FILE)
if PRED_SHA != GO["predictions_sha256"]:
    raise StageDone(⟦f"predictions.json SHA-256 {PRED_SHA} differs from stage2_go.json: stopping."¦f"predictions.json 的 SHA-256 {PRED_SHA} 和 stage2_go.json 不同：停止。"⟧)
PREDS = json.load(open(PRED_FILE))
assert PREDS["r1_commit"] == R1_COMMIT and PREDS["r1_models_sha256"] == R1_SHA
assert read_json(RUNCFG_FILE, None) == RUN_CONFIG, ⟦"run_config.json differs from this run's settings"¦"run_config.json 和這次的設定不同"⟧
print(⟦f"Gate open: {GO['r2_tag']} {GO['r2_commit'][:12]}, Bitcoin block {GO['ots_block_height']} ({GO['ots_block_time']})"¦f"閘門通過：{GO['r2_tag']} {GO['r2_commit'][:12]}，比特幣區塊 {GO['ots_block_height']}（{GO['ots_block_time']}）"⟧)
''')

# ================================================================== stage 2
code(r'''
# ⟪Stage 2: 300 continuations at each temperature, then text-path and id-path scoring; every file is checked before reuse.¦第二階段：每個溫度續寫 300 篇，再做文字路徑與編號路徑評分；每個檔案沿用前都先檢查。⟫
device_check()
if not os.path.exists(START2_FILE):
    write_json(START2_FILE, dict(utc=utc()))
DEADLINE = datetime.fromisoformat(read_json(START2_FILE, {})["utc"]).timestamp() + STAGE2_DAYS * 86400
STATUS2 = {}
for e in ENTRIES:
    t0 = time.time(); STATUS2[e["name"]] = stage2_model(e)
    print(f"{tag_of(e['name'])}: {STATUS2[e['name']]} ({(time.time() - t0) / 60:.1f} min)", flush=True)
write_json(STATUS2_FILE, STATUS2)
retry = {n: s for n, s in STATUS2.items() if s.startswith("retry")}
if retry:
    raise StageDone(⟦f"Some models did not finish; Run All again to resume them: {retry}"¦f"有模型沒有完成，請再按 Run All 從存檔接著跑：{retry}"⟧)
''')

# ================================================================== final cell
code(r'''
# ⟪Record the run (versions, GPU, times, the config hash of every file) and flush unsynced files back to Google Drive.¦記錄執行資訊（套件版本、GPU、時間、每個檔案的設定雜湊），把尚未同步的檔案寫回雲端硬碟。⟫
files = {}
for p in sorted(glob.glob(f"{OUT}/humans_*.pkl") + glob.glob(f"{OUT}/scores_*.pkl")):
    files[os.path.basename(p)] = pickle.load(open(p, "rb"))["config_hash"]
for p in sorted(glob.glob(f"{OUT}/gen_*.json")):
    files[os.path.basename(p)] = json.load(open(p))["config_hash"]
run_info = dict(completed_at_utc=utc(), dryrun=DRYRUN, gpu=GPU_NAME, versions=VERSIONS, environment=ENV, python=sys.version, platform=platform.platform(),
                predictions_sha256=PRED_SHA, r1_models_sha256=R1_SHA, stage2_go=GO, stage2_status=STATUS2, temperatures=list(TEMPS),
                models=[e["name"] for e in ENTRIES], file_config_hashes=files, attempts=read_json(ATTEMPTS_FILE, {}),
                fallback_warnings=FALLBACK_BY_MODEL)
write_json(f"{OUT}/run_info.json", run_info)
flush_drive()
print(⟦f"Done: {len(files)} files; results are in {OUT}/"¦f"完成：{len(files)} 個檔案；結果在 {OUT}/"⟧)
''')


# ================================================================== rendering
MSG = re.compile(r"⟦(.*?)¦(.*?)⟧")
TXT = re.compile(r"⟪(.*?)¦(.*?)⟫", re.S)


def literal_parts(src):
    """The f-string expressions of one string literal (checks that a message side is exactly one literal)."""
    tree = ast.parse(src, mode="eval").body
    assert isinstance(tree, (ast.Constant, ast.JoinedStr)) and (not isinstance(tree, ast.Constant) or isinstance(tree.value, str)), src
    return [ast.dump(v) for v in getattr(tree, "values", []) if isinstance(v, ast.FormattedValue)]


def render(src, lang, pairs):
    def msg(m):
        en, zh = m.group(1), m.group(2)
        assert literal_parts(en) == literal_parts(zh), f"translation changes the f-string expressions: {en} / {zh}"
        if (en, zh) not in pairs:
            pairs.append((en, zh))
        return en if lang == "en" else zh
    out = MSG.sub(msg, src)
    out = TXT.sub(lambda m: m.group(1) if lang == "en" else m.group(2), out)
    out = out.replace("⟨DRIVE_DATA⟩", DRIVE_DATA[lang])
    assert not re.search("[⟦⟧⟪⟫⟨⟩¦]", out), "unrendered marker"
    return out


def cell(kind, src):
    lines = src.split("\n")
    c = {"cell_type": kind, "metadata": {}, "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


def build():
    pairs = []
    for lang in ("en", "zh"):
        cells = [cell(kind, render(src, lang, pairs)) for kind, src in CELLS]
        for c in cells:                                  # every code cell must at least parse once magics are gone
            if c["cell_type"] == "code":
                compile("".join(c["source"]), "<cell>", "exec")
        nb = {"cells": cells, "metadata": META, "nbformat": 4, "nbformat_minor": 0}
        DST[lang].parent.mkdir(parents=True, exist_ok=True)
        DST[lang].write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print("wrote", DST[lang])
    allowed = {"about": "Differences allowed between notebooks/15_replication_test.ipynb (en) and the Drive copy (zh), written by "
                        "build_nb15.py. Everything else in the code cells must be identical once comments are removed.",
               "messages": [{"en": en, "zh": zh} for en, zh in pairs],
               "cell2_paths": [{"cell": 2, "en": DRIVE_DATA["en"], "zh": DRIVE_DATA["zh"]}]}
    ALLOWED.write_text(json.dumps(allowed, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("wrote", ALLOWED, f"({len(pairs)} message pairs)")


if __name__ == "__main__":
    build()
