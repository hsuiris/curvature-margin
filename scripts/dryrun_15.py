"""Laptop dry run of notebook 15 (plan 8.8.24, implementation list item 7). CPU only, about 20 minutes.

1. Copies the Chinese notebook, changes only `DRYRUN = False` to True (cell 1), and runs it with nbclient through stage 0,
   a simulated R1, stage 1 (twice: a rerun must reproduce predictions.json), a simulated R2 and the stage-2 gate, then
   stage 2 with gpt2 and h2o-danube3-500m (P and E 60 texts each, 12 prompts, temperatures 1.00 and 1.14, float32).
   Checks along the way: stage 0 keeps no scores; no text is generated before R2; the gate stops without stage2_go.json
   and after a one-byte change to predictions.json; an exception injected during generation is resumed from the saved
   files; damaged score files are moved to _invalid/ with the reason and redone; a Colab image change stops the notebook
   until a revision note and env-update, then a model finished in the old environment is kept and a half-done one is
   redone. Then e12 runs twice on the output (byte-identical results required). The registrations get real
   OpenTimestamps proofs (ots stamp, so the network is needed); every refusal must give the expected reason.
2. Runs the notebook's own functions outside the kernel: cell 4 with the registered quota on the real manifest (P and E
   1,000 each; the main reference is the whole E group), rules 2-6 for every tokenizer that can be downloaded (12 models
   and Zamba2; Llama 3.2 and Gemma 3 through the unsloth mirrors when no HF token is set), the 512-token truncation test
   for gpt2, h2o-danube3-500m and TinyLlama v1.1, and the replacement / rule-9 logic.
Usage: python scripts/dryrun_15.py [--log FILE] [--work DIR]     (at R0: --log registration/R0_dryrun_log.txt)
Exit code 0 means every check passed.
"""
import argparse, contextlib, copy, hashlib, io, json, os, pathlib, pickle, re, shutil, subprocess, sys, time, traceback
from datetime import datetime, timedelta, timezone

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_registration as reg_tool   # noqa: E402

GDRIVE = pathlib.Path("/Users/xuyunqin/Library/CloudStorage/GoogleDrive-emilyhuang12380@gmail.com/我的雲端硬碟/AI-Text/03_實驗")
ZH = GDRIVE / "notebooks" / "15_replication_test.ipynb"
DRIVE_DATA = GDRIVE / "data"
TMP = pathlib.Path("/private/tmp/cm_work/tmp")
PY = sys.executable
# cell layout (both versions): 0 notes, 1-5 setup, 6-8 stage 0, 9 stage 1, 10 gate, 11 stage 2, 12 run record
SETUP, S0, S1, GATE, S2, FINAL = [1, 2, 3, 4, 5], [6, 7, 8], 9, 10, 11, 12
FIRST_LINE = {1: "# 1.", 2: "# 2.", 3: "# 3.", 4: "# 4.", 5: "# 5.", 6: "# 第零階段 0a", 7: "# 第零階段 0b", 8: "# 第零階段 0c",
              9: "# 第一階段", 10: "# 第二階段閘門", 11: "# 第二階段：", 12: "# 記錄執行資訊"}
TRUNCATION = {"gpt2": (None, 511, 513, 512, 1, 30, 482), "h2oai/h2o-danube3-500m-base": (None, 511, 513, 512, 1, 30, 482),
              "TinyLlama/TinyLlama_v1.1": ([1], 512, 513, 512, 1, 31, 481)}
TOKENIZERS = [(1, "meta-llama/Llama-3.2-1B", "unsloth/Llama-3.2-1B"), (2, "google/gemma-3-1b-pt", "unsloth/gemma-3-1b-pt"),
              (3, "TinyLlama/TinyLlama_v1.1", None), (4, "togethercomputer/RedPajama-INCITE-Base-3B-v1", None),
              (5, "stabilityai/stablelm-3b-4e1t", None), (6, "h2oai/h2o-danube2-1.8b-base", None),
              (7, "h2oai/h2o-danube3-500m-base", None), (8, "utter-project/EuroLLM-1.7B", None), (9, "BSC-LT/salamandra-2b", None),
              (10, "facebook/xglm-1.7B", None), (11, "state-spaces/mamba-1.4b-hf", None), (12, "PleIAs/Pleias-1.2b-Preview", None),
              (13, "Zyphra/Zamba2-1.2B", None)]
DAN = "h2oai/h2o-danube3-500m-base"
PLAN_START = {1: [128000], 2: [2], 3: [1], 8: [1], 9: [1], 10: [2]}   # the plan's table; every other model: none


class Log:
    def __init__(self):
        self.lines, self.fails = [], []

    def say(self, *parts):
        line = " ".join(str(p) for p in parts); self.lines.append(line); print(line, flush=True)

    def check(self, label, ok, detail=""):
        self.say(("通過" if ok else "失敗") + f"｜{label}" + (f"｜{detail}" if detail else ""))
        if not ok:
            self.fails.append(label)
        return ok


LOG = Log()
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------ setup
def prepare(work):
    data = work / "data"; out = data / "15_replication_test"
    if work.exists():
        shutil.rmtree(work)
    (data / "external" / "m4gt").mkdir(parents=True); (data / "manifests").mkdir()
    os.symlink(DRIVE_DATA / "external" / "m4gt" / "SubtaskB.jsonl", data / "external" / "m4gt" / "SubtaskB.jsonl")
    for f in ("m4gt_humans_PE_manifest.csv", "m4gt_humans_A_manifest.csv"):
        shutil.copy(DRIVE_DATA / "manifests" / f, data / "manifests" / f)
    (data / "dry_protocol.md").write_text("dry-run stand-in for docs/prereg/8824_protocol.md\n", encoding="utf-8")
    return data, out


def dry_notebook():
    nb = nbformat.read(str(ZH), as_version=4)
    for i, head in FIRST_LINE.items():
        assert nb.cells[i].cell_type == "code" and nb.cells[i].source.startswith(head), f"cell {i} is not {head!r}"
    src = nb.cells[1].source
    assert src.count("DRYRUN = False") == 1 and all("DRYRUN = False" not in c.source for c in nb.cells[2:])
    before = [c.source for c in nb.cells]
    nb.cells[1].source = src.replace("DRYRUN = False", "DRYRUN = True", 1)
    changed = [i for i, c in enumerate(nb.cells) if c.source != before[i]]
    LOG.check("只改第 1 格的 DRYRUN = False（計畫第 5.2 版：旗標在第 1 格）", changed == [1], f"改動的格：{changed}；中文版 SHA-256 {sha(ZH)}")
    return nb


class Session:
    """One kernel = one Colab session. run(i) executes notebook cell i; inject(code) runs code that is not in the notebook."""

    def __init__(self, nb, work):
        self.nb = copy.deepcopy(nb)
        self.client = NotebookClient(self.nb, timeout=7200, kernel_name="python3", resources={"metadata": {"path": str(work)}})

    def __enter__(self):
        self._cm = self.client.setup_kernel(); self._cm.__enter__(); return self

    def __exit__(self, *exc):
        return self._cm.__exit__(*exc)

    def run(self, i):
        try:
            self.client.execute_cell(self.nb.cells[i], i)
            return "ok", ""
        except CellExecutionError as err:
            return err.ename, err.evalue

    def text(self, i):
        return "".join(o.get("text", "") for o in self.nb.cells[i].get("outputs", []) if o.get("output_type") == "stream")

    def inject(self, code):
        """nbclient writes a cell's result back to its index, so the extra cell is appended for the call and removed."""
        self.nb.cells.append(nbformat.v4.new_code_cell(code))
        try:
            self.client.execute_cell(self.nb.cells[-1], len(self.nb.cells) - 1)
        finally:
            self.nb.cells.pop()


def run_cells(s, cells, expect_last=None):
    for i in cells:
        t0 = time.time(); r = s.run(i)
        LOG.say(f"  第 {i} 格：{r[0]} {r[1][:160]}（{time.time() - t0:.0f} 秒）")
        if r[0] != "ok":
            return r
    return "ok", ""


def registration(args):
    return reg_tool.main(["--registration", str(REG), "--root", str(DATA)] + args)


def proof(root, sec):
    return str(root / "registration" / f"{sec}.SHA256SUMS.ots")


def stamp(run, root, sec, *extra):
    """SHA256SUMS of a registration, then a real OpenTimestamps proof of it (sent to the calendars; needs the network)."""
    ok = run(["sums", sec, *extra]) == 0
    pathlib.Path(proof(root, sec)).unlink(missing_ok=True)   # ots stamp will not overwrite an old proof
    return ok and reg_tool.run_tool("ots", "stamp", str(root / "registration" / f"{sec}.SHA256SUMS"))[0] == 0


def refused(run, args, why):
    """The command must refuse for the expected reason: a refusal for some other reason would hide a broken check."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = run(args)
    print(buf.getvalue(), end="")
    return code == 1 and why in buf.getvalue()


# ------------------------------------------------------------------ part 1: the notebook, stage by stage
RELEASE = {"R0": "2026-01-01T00:00:00+00:00", "R1": "2026-01-02T00:00:00+00:00", "R2": "2026-01-03T00:00:00+00:00"}   # dry-run Release times
FLAKY = ("_real_load_model, _flaky = load_model, {flaky!r}\n"
         "def load_model(name, revision):\n"
         "    if name in _flaky:\n"
         "        raise OSError(\"dry run: We couldn't connect to 'https://huggingface.co'\")\n"
         "    return _real_load_model(name, revision)\n")


def add_failed_attempts(stage, name, n):
    att = json.load(open(OUT / "attempts.json")) if (OUT / "attempts.json").exists() else {}
    now = datetime.now(timezone.utc).isoformat()           # failed attempts happen during the stage, after the last registration
    att.setdefault(stage, {}).setdefault(name, []).extend([dict(start=now, end=now, ok=False, error="OSError: dry run")] * n)
    json.dump(att, open(OUT / "attempts.json", "w"), indent=1)


def part1(nb, work):
    spec = work / "spec.json"
    put = lambda obj: spec.write_text(json.dumps(obj), encoding="utf-8")
    # stage 0 while the GPU-check download of gpt2 fails (review A1/B1): retried, nothing recorded, no candidate used
    with Session(nb, work) as s:
        run_cells(s, SETUP); s.inject(FLAKY.format(flaky=("gpt2",))); r = run_cells(s, S0)
    LOG.check("第零階段下載暫時失敗：0c 要求重跑，不寫 stage0_record.json，不動用候補", r[0] == "StageDone" and "無法載入" in r[1]
              and not (OUT / "stage0_record.json").exists() and "load_error" in json.load(open(OUT / "stage0" / "gpu_gpt2.json")), r[1][:90])
    with Session(nb, work) as s:
        r = run_cells(s, SETUP + S0)
    rec = json.load(open(OUT / "stage0_record.json"))
    LOG.check("下載恢復後第零階段完成，最後一格 raise", r[0] == "StageDone" and rec["final"] == ["gpt2", "h2oai/h2o-danube3-500m-base"]
              and not rec["changes"], r[1][:80])
    files = sorted(str(p.relative_to(OUT)) for p in OUT.rglob("*") if p.is_file())
    LOG.check("第零階段只寫出預檢紀錄，沒有分數或生成文字",
              all(f in ("stage0_record.json", "environment.json") or re.fullmatch(r"stage0/(tok|gpu)_.+\.json", f) for f in files)
              and '"arr"' not in (OUT / "stage0_record.json").read_text(), f"{files}")
    sha0, mt0 = sha(OUT / "stage0_record.json"), (OUT / "stage0_record.json").stat().st_mtime_ns
    with Session(nb, work) as s:
        r = run_cells(s, SETUP + S0)
    LOG.check("R1 之前再按 Run All：stage0_record.json 只寫一次，不被改寫（審查 B3）", r[0] == "StageDone"
              and sha(OUT / "stage0_record.json") == sha0 and (OUT / "stage0_record.json").stat().st_mtime_ns == mt0)
    # simulated R0 / R1
    put(dict(protocol="dry_protocol.md", files=["dry_protocol.md", "manifests/m4gt_humans_PE_manifest.csv", "manifests/m4gt_humans_A_manifest.csv"],
             decisions=dict(criterion2="form R", equivalence_margin=0.002, alpha="2.5% then 1.25% + 1.25%", gpu_budget_hours=8,
                            candidate="Zyphra/Zamba2-1.2B", osf=False), seeds=dict(dryrun=True), packages=dict(transformers="5.17.0")))
    LOG.check("模擬 R0：make R0、check R0、SHA256SUMS 與真正送出的時間證明（ots stamp）",
              registration(["make", "R0", "--spec", str(spec)]) == 0 and registration(["check", "R0"]) == 0 and stamp(registration, DATA, "R0"))
    shutil.copy(REG, work / "registration_R0.json")
    put(dict(previous=dict(tag="prereg-R0", commit="dryrun-r0", release_created_at=RELEASE["R0"])))
    ok = registration(["make", "R1", "--spec", str(spec), "--stage0", str(OUT / "stage0_record.json")]) == 0
    r1m = ["r1-models", "--out", str(OUT / "r1_models.json"), "--r1-commit", "dryrun-r1", "--ots-r0", proof(DATA, "R0"),
           "--ots-r1", proof(DATA, "R1"), "--skip-gh"]
    LOG.check("r1-models：R1 的時間證明還沒送出時不寫 r1_models.json（審查 A3）",
              ok and refused(registration, r1m, "不是有效的時間證明") and not (OUT / "r1_models.json").exists())
    pathlib.Path(proof(DATA, "R1")).write_bytes(b"")
    LOG.check("r1-models：空的 .ots 不算時間證明（審查 A3）",
              refused(registration, r1m, "不是有效的時間證明") and not (OUT / "r1_models.json").exists())
    ok = stamp(registration, DATA, "R1", "15_replication_test/stage0_record.json") and registration(r1m) == 0
    c1 = ["check", "R1", "--previous", str(work / "registration_R0.json"), "--r1-models", str(OUT / "r1_models.json")]
    LOG.check("check R1 沒帶 --stage0 就不通過（審查 A3）", refused(registration, c1, "一定要帶 --stage0"))
    ok &= registration(c1 + ["--stage0", str(OUT / "stage0_record.json")]) == 0
    LOG.check("模擬 R1：make R1、真正送出（未升級）的 R0／R1 時間證明、r1-models、check R1（分群與雜湊、第零階段晚於 R0 的 Release）", ok)
    shutil.copy(REG, work / "registration_R1.json")
    # stage 1: danube3's model download fails once; gpt2 finishes (review A2)
    with Session(nb, work) as s:
        run_cells(s, SETUP + S0); s.inject(FLAKY.format(flaky=("h2oai/h2o-danube3-500m-base",))); r = run_cells(s, [S1])
    LOG.check("第一階段有模型暫時失敗：raise 要求重跑，不寫 predictions.json", r[0] == "StageDone" and "重跑" in r[1]
              and not (OUT / "predictions.json").exists() and (OUT / "humans_gpt2.pkl").exists(), r[1][:90])
    add_failed_attempts("stage1", "gpt2", 4)
    with Session(nb, work) as s:
        r = run_cells(s, SETUP + S0 + [S1])
        printed = re.findall(r"predictions\.json 的 SHA-256：([0-9a-f]{64})", s.text(S1))
    preds = json.load(open(OUT / "predictions.json"))
    LOG.check("已完成的模型有 4 次失敗紀錄仍算完成，預測照常寫出（審查 A2）", r[0] == "StageDone"
              and {k: v["status"] for k, v in preds["models"].items()} == {"gpt2": "written", "h2oai/h2o-danube3-500m-base": "written"})
    colab_sha = printed[0] if printed else ""
    LOG.check("第一階段最後一格 raise，印出的 SHA-256 等於寫入的 predictions.json", colab_sha == sha(OUT / "predictions.json"), colab_sha)
    LOG.check("第二階段之前沒有任何生成檔", not list(OUT.glob("gen_*")))
    before = sha(OUT / "predictions.json")
    with Session(nb, work) as s:
        r = run_cells(s, SETUP + S0 + [S1])
    LOG.check("第一階段重跑：重算後和存檔完全相同，檔案不變", r[0] == "StageDone" and sha(OUT / "predictions.json") == before)
    hp = OUT / "humans_gpt2.pkl"; keep = hp.read_bytes()
    obj = pickle.load(open(hp, "rb")); obj["rows"][0]["arr"] = obj["rows"][0]["arr"].copy(); obj["rows"][0]["arr"][0, 0] = float("nan")
    hp.write_bytes(pickle.dumps(obj))
    with Session(nb, work) as s:
        r = run_cells(s, SETUP + S0 + [S1])
    LOG.check("predictions.json 寫出後人類分數檔損壞：停止，不重評、不移走（審查 B3）", r[0] == "StageStop" and hp.read_bytes() != keep
              and not (OUT / "_invalid").exists() and sha(OUT / "predictions.json") == before, r[1][:90])
    hp.write_bytes(keep)
    # simulated R2 and the gate
    put(dict(previous=dict(tag="prereg-R1", commit="dryrun-r1", release_created_at=RELEASE["R1"])))
    ok = registration(["make", "R2", "--spec", str(spec), "--predictions", str(OUT / "predictions.json"), "--colab-sha", colab_sha]) == 0
    c2 = ["check", "R2", "--previous", str(work / "registration_R1.json")]
    LOG.check("check R2 沒帶 --attempts 就不通過（審查 A3）", refused(registration, c2, "一定要帶 --attempts"))
    ok &= registration(c2 + ["--attempts", str(OUT / "attempts.json")]) == 0 and stamp(registration, DATA, "R2", "15_replication_test/predictions.json")
    LOG.check("模擬 R2：make R2、check R2（第一階段晚於 R1 的 Release）、真正送出的 R2 時間證明", ok)
    go = ["go", "--predictions", str(OUT / "predictions.json"), "--r2-commit", "dryrun-r2", "--block-height", "2",
          "--block-time", RELEASE["R2"], "--out", str(OUT / "stage2_go.json"), "--skip-gh", "--ots", proof(DATA, "R2")]
    LOG.check("go：R2 的時間證明還沒升級到比特幣區塊時不寫閘門檔（R2 維持原規則）",
              refused(registration, go + ["--colab-sha", colab_sha], "還沒有升級到比特幣區塊") and not (OUT / "stage2_go.json").exists())
    go.append("--skip-upgrade")   # a dry run cannot wait hours for a Bitcoin attestation; the proof itself is still checked
    LOG.check("go：Colab 印出的雜湊不同時不寫閘門檔", refused(registration, go + ["--colab-sha", "0" * 64], "和 Colab 印出的") and not (OUT / "stage2_go.json").exists())
    with Session(nb, work) as s:
        run_cells(s, SETUP + S0 + [S1])
        r = s.run(GATE)
        LOG.check("沒有 stage2_go.json 時閘門 raise", r[0] == "StageDone" and "stage2_go.json" in r[1], r[1][:80])
        LOG.check("go：三項核對通過後寫入 stage2_go.json", registration(go + ["--colab-sha", colab_sha]) == 0 and (OUT / "stage2_go.json").exists())
        good = (OUT / "predictions.json").read_bytes()
        bad = bytearray(good); k = bad.index(b'"T_hat": ') + 12; bad[k] = ord("9") if bad[k] != ord("9") else ord("8")
        (OUT / "predictions.json").write_bytes(bytes(bad))
        r = s.run(GATE)
        LOG.check("predictions.json 改一個位元組後閘門 raise", r[0] == "StageDone" and "SHA-256" in r[1], r[1][:80])
        (OUT / "predictions.json").write_bytes(good)
        r = s.run(GATE)
        LOG.check("還原後閘門通過", r[0] == "ok", r[1][:80])
    # stage 2 with an exception injected in the middle of generation
    with Session(nb, work) as s:
        run_cells(s, SETUP + S0 + [S1, GATE])
        s.inject("_generate_real, _calls = generate, [0]\n"
                 "def generate(*a, **k):\n"
                 "    _calls[0] += 1\n"
                 "    if _calls[0] == 2:\n"
                 "        raise RuntimeError('dry run: exception injected during generation')\n"
                 "    return _generate_real(*a, **k)\n")
        r = s.run(S2)
    LOG.check("生成途中丟出例外：第二階段 raise 並記錄這次嘗試", r[0] == "StageDone" and
              "injected" in json.dumps(json.load(open(OUT / "attempts.json"))["stage2"]), r[1][:120])
    LOG.check("例外之前完成的溫度已存檔、例外那個溫度沒有檔案", (OUT / "gen_gpt2_T1.00.json").exists() and not (OUT / "gen_gpt2_T1.14.json").exists())
    kept = {p.name: p.stat().st_mtime_ns for p in OUT.glob("*_gpt2_T1.00.*")}
    with Session(nb, work) as s:
        r = run_cells(s, SETUP + S0 + [S1, GATE, S2, FINAL])
    LOG.check("重跑後從存檔接著跑完（先前的檔案沒有重做）", r[0] == "ok" and all(OUT.joinpath(n).stat().st_mtime_ns == t for n, t in kept.items())
              and (OUT / "gen_gpt2_T1.14.json").exists() and (OUT / "run_info.json").exists(), r[1][:120])
    # damaged score files
    (OUT / "scores_ids_gpt2_T1.14.pkl").write_bytes(b"damaged")
    p = OUT / "scores_text_h2o-danube3-500m-base_T1.00.pkl"
    obj = pickle.load(open(p, "rb")); obj["rows"][0]["arr"] = obj["rows"][0]["arr"].copy(); obj["rows"][0]["arr"][0, 0] = float("nan")
    p.write_bytes(pickle.dumps(obj))
    with Session(nb, work) as s:
        r = run_cells(s, SETUP + S0 + [S1, GATE, S2, FINAL])
    log = [json.loads(x) for x in (OUT / "_invalid" / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    reasons = {x["file"]: x["reason"] for x in log}
    LOG.check("損壞的評分檔移到 _invalid/ 並記錄原因，之後重做完成", r[0] == "ok"
              and "讀不出來" in reasons.get("scores_ids_gpt2_T1.14.pkl", "") and "非有限值" in reasons.get(p.name, "")
              and (OUT / "scores_ids_gpt2_T1.14.pkl").exists() and p.exists(), json.dumps(reasons, ensure_ascii=False))
    # Colab changed its preinstalled versions (plan 5.4). gpt2 had finished stage 2 in the old environment, danube3 was half
    # done. The notebook stops; after a revision note and env-update, gpt2 is kept and danube3 is redone from scratch.
    envf = OUT / "environment.json"; cur = json.load(open(envf)); old = dict(cur, torch=f"{cur['torch']}-before")
    for p in [*OUT.glob("gen_*_T*.json"), *OUT.glob("scores_*_T*.pkl")]:
        if p.suffix == ".json":
            obj = json.load(open(p)); obj["environment"] = old; p.write_text(json.dumps(obj), encoding="utf-8")
        else:
            obj = pickle.load(open(p, "rb")); obj["environment"] = old; p.write_bytes(pickle.dumps(obj, protocol=4))
    gone = "scores_ids_h2o-danube3-500m-base_T1.14.pkl"; (OUT / gone).unlink()
    json.dump(old, open(envf, "w"))
    kept = {p.name: p.stat().st_mtime_ns for p in OUT.glob("*_gpt2_T*.*")}
    tries = len(json.load(open(OUT / "attempts.json"))["stage2"].get(DAN, []))
    with Session(nb, work) as s:
        r = run_cells(s, SETUP)
    LOG.check("Colab 預裝版本和第一次執行不同：第 5 格停下，提示修訂紀錄與 env-update（計畫 5.4）",
              r[0] == "AssertionError" and "env-update" in r[1], r[1][:90])
    note = DATA / "docs" / "revisions" / "dryrun_colab.md"; note.parent.mkdir(parents=True)
    upd = ["env-update", "--environment", str(envf), "--new", json.dumps(cur), "--note", "docs/revisions/dryrun_colab.md", "--skip-git"]
    LOG.check("沒有修訂紀錄時 env-update 不放行，environment.json 不變", refused(registration, upd, "不存在") and json.load(open(envf)) == old)
    note.write_text(f"{datetime.now(timezone.utc):%Y-%m-%d} 試跑：模擬 Colab 更新映像。原因：預裝的 torch 版本改變"
                    f"（torch {old['torch']} → {cur['torch']}）。\n", encoding="utf-8")
    ok = registration(upd) == 0 and json.load(open(envf)) == cur and (OUT / "environment_history.json").exists()
    with Session(nb, work) as s:
        r = run_cells(s, SETUP + S0 + [S1, GATE, S2, FINAL])
    moved = sorted(x["file"] for x in map(json.loads, (OUT / "_invalid" / "log.jsonl").read_text(encoding="utf-8").splitlines())
                   if "執行環境改變" in x["reason"])
    redone = sorted(OUT.glob("*_h2o-danube3-500m-base_T*.*"))
    env_of = lambda p: (json.load(open(p)) if p.suffix == ".json" else pickle.load(open(p, "rb")))["environment"]
    LOG.check("寫修訂紀錄、用 env-update 更新後繼續：舊環境已做完的 gpt2 不重做；做到一半的 danube3 五個舊檔移到 _invalid/、"
              "從頭重做並記一次嘗試（計畫 5.4）", ok and r[0] == "ok" and all(OUT.joinpath(n).stat().st_mtime_ns == t for n, t in kept.items())
              and moved == [p.name for p in redone if p.name != gone] and len(redone) == 6 and all(env_of(p) == cur for p in redone)
              and len(json.load(open(OUT / "attempts.json"))["stage2"][DAN]) == tries + 1, f"移走：{moved}")
    # finished models past the 14-day limit and the rerun limit (review A2/B4): the notebook keeps them, e12 applies state 3
    started = (OUT / "stage2_started.json").read_bytes(); attempts = (OUT / "attempts.json").read_bytes()
    json.dump(dict(utc="2026-01-04T00:00:00+00:00"), open(OUT / "stage2_started.json", "w")); add_failed_attempts("stage2", "gpt2", 4)
    with Session(nb, work) as s:
        r = run_cells(s, SETUP + S0 + [S1, GATE, S2, FINAL])
    status2 = json.load(open(OUT / "stage2_status.json"))
    LOG.check("已完成的模型過了 14 天、有 4 次失敗紀錄，notebook 仍記為完成（審查 B4）", r[0] == "ok" and set(status2.values()) == {"done"}, json.dumps(status2))
    o = work / "e12_late"
    rr = subprocess.run([PY, str(ROOT / "scripts" / "e12_replication_test.py"), "--data", str(DATA), "--registration", str(REG), "--out", str(o), "--dryrun"],
                        capture_output=True, text=True)
    late = pd_read(o / "per_model.csv")
    LOG.check("e12 讀 stage2_started.json：期限後才寫入的檔案判為狀態③", rr.returncode == 0 and late and all(str(x["state"]) == "3" and "超過 14 天期限" in str(x["issues"]) for x in late),
              str([(x["model"], x["state"]) for x in late]))
    (OUT / "stage2_started.json").write_bytes(started); (OUT / "attempts.json").write_bytes(attempts)
    # e12 twice on the output
    outs = []
    for k in (1, 2):
        o = work / f"e12_out{k}"
        r = subprocess.run([PY, str(ROOT / "scripts" / "e12_replication_test.py"), "--data", str(DATA), "--registration", str(REG),
                            "--out", str(o), "--dryrun"], capture_output=True, text=True)
        LOG.say(r.stdout.strip()[-600:] + r.stderr.strip()[-600:])
        outs.append(o)
    names = ("per_model.csv", "per_family.csv", "criteria.json", "claims.md")
    LOG.check("e12 在試跑輸出上跑完，兩次結果逐位元組相同", all((o / n).exists() for o in outs for n in names)
              and all((outs[0] / n).read_bytes() == (outs[1] / n).read_bytes() for n in names))
    env2 = {x["model"]: x["environment_stage2"] for x in pd_read(outs[0] / "per_model.csv")}
    LOG.check("e12 分開列出環境改變前後的模型（gpt2 舊版、danube3 新版），兩者都不因此判為狀態③（計畫 5.4）",
              "-before" in env2.get("gpt2", "") and "-before" not in env2.get(DAN, "-before") and "|" not in "".join(env2.values()),
              json.dumps(env2, ensure_ascii=False))


def registration_fragment(work):
    """Only the registration steps (R0, R1, r1-models, R2, go), replayed on the notebook outputs an earlier dry run left in
    the work folder, in a fresh registration file. Release times are placed before or after the real stage start times."""
    root = work / "reg_fragment"; shutil.rmtree(root, ignore_errors=True); (root / "manifests").mkdir(parents=True)
    for f in ("m4gt_humans_PE_manifest.csv", "m4gt_humans_A_manifest.csv"):
        shutil.copy(DATA / "manifests" / f, root / "manifests" / f)
    (root / "dry_protocol.md").write_text("dry-run stand-in for docs/prereg/8824_protocol.md\n", encoding="utf-8")
    rg = lambda args: reg_tool.main(["--registration", str(root / "registration.json"), "--root", str(root)] + args)
    spec = root / "spec.json"; put = lambda obj: spec.write_text(json.dumps(obj), encoding="utf-8")
    rec, att = json.load(open(OUT / "stage0_record.json")), json.load(open(OUT / "attempts.json"))
    t0 = min(datetime.fromisoformat(r["utc"]) for r in rec["tokenizer_checks"].values())
    t1 = min(datetime.fromisoformat(x["start"]) for v in att["stage1"].values() for x in v)
    before = lambda t: (t - timedelta(hours=1)).isoformat(); after = lambda t: (t + timedelta(hours=1)).isoformat()
    put(dict(protocol="dry_protocol.md", files=["dry_protocol.md", "manifests/m4gt_humans_PE_manifest.csv"],
             decisions=dict(criterion2="form R", equivalence_margin=0.002, alpha="split", gpu_budget_hours=8, candidate="Zyphra/Zamba2-1.2B",
                            osf=False), seeds=dict(dryrun=True), packages=dict(transformers="5.17.0")))
    LOG.check("片段：make R0 與 check R0", rg(["make", "R0", "--spec", str(spec)]) == 0 and rg(["check", "R0"]) == 0)
    shutil.copy(root / "registration.json", root / "r0.json")
    put(dict(previous=dict(tag="prereg-R0", commit="frag-r0", release_created_at=before(t0))))
    ok = rg(["make", "R1", "--spec", str(spec), "--stage0", str(OUT / "stage0_record.json")]) == 0
    r1m = ["r1-models", "--out", str(root / "r1_models.json"), "--r1-commit", "frag-r1", "--ots-r0", proof(root, "R0"),
           "--ots-r1", proof(root, "R1"), "--skip-gh"]
    no_r1m = lambda why: refused(rg, r1m, why) and not (root / "r1_models.json").exists()
    LOG.check("片段：R0、R1 的時間證明都還沒送出時，r1-models 不寫 r1_models.json", ok and no_r1m("不是有效的時間證明"))
    LOG.check("片段：只有 R0 送出時，r1-models 不寫 r1_models.json", stamp(rg, root, "R0") and no_r1m("不是有效的時間證明"))
    stamp(rg, root, "R1"); pathlib.Path(proof(root, "R1")).write_bytes(b"\x00garbage")
    LOG.check("片段：R1 的 .ots 是亂碼時，r1-models 不寫 r1_models.json", no_r1m("不是有效的時間證明"))
    shutil.copy(proof(root, "R0"), proof(root, "R1"))
    LOG.check("片段：R1 的位置放的是 R0 的證明時，r1-models 不寫 r1_models.json", no_r1m("證明的不是 R1.SHA256SUMS"))
    LOG.check("片段：R0、R1 都真正送出（未升級，計畫 5.3）時，r1-models 寫出 r1_models.json",
              stamp(rg, root, "R1") and rg(r1m) == 0 and (root / "r1_models.json").exists())
    chk = ["check", "R1", "--previous", str(root / "r0.json"), "--r1-models", str(root / "r1_models.json")]
    LOG.check("片段：check R1 沒帶 --stage0 就不通過", refused(rg, chk, "一定要帶 --stage0"))
    chk += ["--stage0", str(OUT / "stage0_record.json")]
    LOG.check("片段：check R1（第零階段晚於 R0 的 Release、分群與雜湊一一對應、R0 的證明為真）", rg(chk) == 0)
    data = json.load(open(root / "registration.json")); data["R1"]["previous"]["release_created_at"] = after(t0)
    json.dump(data, open(root / "late.json", "w"))
    LOG.check("片段：第零階段早於 R0 的 Release 時，check R1 判為不符",
              refused(lambda x: reg_tool.main(["--registration", str(root / "late.json"), "--root", str(root)] + x), chk, "第零階段 的開始時間"))
    shutil.copy(root / "registration.json", root / "r1.json")
    psha = sha(OUT / "predictions.json")
    put(dict(previous=dict(tag="prereg-R1", commit="frag-r1", release_created_at=before(t1))))
    ok = rg(["make", "R2", "--spec", str(spec), "--predictions", str(OUT / "predictions.json"), "--colab-sha", psha]) == 0
    c2 = ["check", "R2", "--previous", str(root / "r1.json")]
    LOG.check("片段：check R2 沒帶 --attempts 就不通過", ok and refused(rg, c2, "一定要帶 --attempts"))
    LOG.check("片段：check R2（第一階段晚於 R1 的 Release、R0 與 R1 的證明為真）",
              rg(c2 + ["--attempts", str(OUT / "attempts.json")]) == 0 and stamp(rg, root, "R2"))
    go = ["go", "--predictions", str(OUT / "predictions.json"), "--colab-sha", psha, "--r2-commit", "frag-r2", "--block-height", "900000",
          "--out", str(root / "stage2_go.json"), "--skip-gh", "--ots", proof(root, "R2")]
    past, future = before(datetime.now(timezone.utc)), "2999-01-01T00:00:00+00:00"
    LOG.check("片段：R2 的證明未升級時，go 不寫 stage2_go.json（R2 維持原規則）",
              refused(rg, go + ["--block-time", past], "還沒有升級到比特幣區塊 900000") and not (root / "stage2_go.json").exists())
    LOG.check("片段：區塊時間不早於現在時，go 不寫 stage2_go.json",
              refused(rg, go + ["--block-time", future, "--skip-upgrade"], "不早於現在") and not (root / "stage2_go.json").exists())
    LOG.check("片段：證明為真且區塊時間早於現在時（試跑以 --skip-upgrade 代替升級），go 寫出 stage2_go.json",
              rg(go + ["--block-time", past, "--skip-upgrade"]) == 0 and (root / "stage2_go.json").exists())


def pd_read(path):
    import csv
    return list(csv.DictReader(open(path, encoding="utf-8"))) if path.exists() else []


# ------------------------------------------------------------------ part 2: the notebook's functions outside the kernel
def part2(nb):
    from IPython.core.inputtransformer2 import TransformerManager
    tm, g = TransformerManager(), {"__name__": "nb15"}
    for i in SETUP:
        exec(compile(tm.transform_cell(nb.cells[i].source), f"<cell {i}>", "exec"), g)
    g["DRYRUN"] = False   # cell 4 once more with the registered quota: every row of the real manifest
    exec(compile(tm.transform_cell(nb.cells[4].source), "<cell 4, registered quota>", "exec"), g)
    g["DRYRUN"] = True
    n = (len(g["P_IDS"]), len(g["E_IDS"]), len(g["A_IDS"]), len(g["B_IDS"]), len(set(g["E_IDS"]) - set(g["A_IDS"])))
    LOG.check("第 4 格用登錄配額讀真正的 manifest：P、E、A 半、B 半、E 扣 A 半", n == (1000, 1000, 150, 150, 850), f"{n}")
    # rules 2-6 for every tokenizer that can be downloaded
    rows = []
    for no, name, mirror in TOKENIZERS:
        use = mirror or name
        rec = g["tokenizer_check"](dict(no=no, name=use, family="-", group=None), g["P_IDS"], g["E_IDS"], g["PROMPT_IDS"])
        if "load_error" in rec:
            LOG.check(f"規則 2 到 6：{use}", False, rec["load_error"]); continue
        r5 = rec["rules"]["5"]; r6 = rec["rules"]["6"]
        rows.append(dict(model=use, start=rec["start"], pad=rec["pad"], banned=len(rec["banned"]), dropped=rec["rules"]["3"]["dropped_tokenizer_ids"],
                         P=f"{r5['P']['share']:.3f}", E=f"{r5['E']['share']:.3f}", min_tokens=r6["min_tokens"], min_scored=r6["min_scored"],
                         S_m=rec["S_m"], j_back=rec["prompt_j_back"], hash=rec["tokenizer_hash"]))
        LOG.check(f"規則 2 到 6：{use}{'（非官方鏡像）' if mirror else ''}", rec["passed"] and (no > 12 or rec["start"] == PLAN_START.get(no, [])),
                  json.dumps(rows[-1], ensure_ascii=False))
    groups = {}
    for r in rows:
        groups.setdefault(r["hash"], []).append(r["model"].split("/")[-1])
    LOG.say("tokenizer 分群（依 tokenizer.json 核心雜湊）：" + json.dumps(list(groups.values()), ensure_ascii=False))
    want = [["Llama-3.2-1B"], ["gemma-3-1b-pt"], ["TinyLlama_v1.1"], ["RedPajama-INCITE-Base-3B-v1", "stablelm-3b-4e1t", "mamba-1.4b-hf"],
            ["h2o-danube2-1.8b-base", "h2o-danube3-500m-base"], ["EuroLLM-1.7B"], ["salamandra-2b"], ["xglm-1.7B"],
            ["Pleias-1.2b-Preview"], ["Zamba2-1.2B"]]
    LOG.check("tokenizer 分群和計畫的暫定 9 群相同（Zamba2 自成一群）", sorted(map(sorted, groups.values())) == sorted(map(sorted, want)))
    # truncation at 512
    import numpy as np, torch
    for name, (start_want, id_len, reenc, after, cut_want, k_want, scored_want) in TRUNCATION.items():
        tok = g["AutoTokenizer"].from_pretrained(name)
        start, _ = g["start_tokens"](tok)
        sp = "Ġ" if name == "gpt2" else "▁"; vocab = tok.get_vocab()
        filler = tok(" The committee reviewed the proposal and the results were discussed at length by all members present." * 60,
                     add_special_tokens=False)["input_ids"]
        p = filler[:30]; body = filler[30:30 + 479]
        q = body[:200] + [vocab[sp + "modern"], vocab["events"]] + body[200:]
        ids, k, cut, j = g["text_input"](tok, start, p, q)
        raw = len(start) + len(tok(g["dec"](tok, p + q), add_special_tokens=False)["input_ids"])
        got = (start or None, len(start) + len(p) + len(q), raw, len(ids), cut, k, len(ids) - k)
        LOG.check(f"截斷測試數字：{name}", got == (start_want, id_len, reenc, after, cut_want, k_want, scored_want), f"{got}")
        e = dict(name=name, revision="dryrun", start=start, banned_sha256="dryrun", pad=0)
        g["R1_COMMIT"] = "dryrun"
        cfg = g["file_config"](e, "machine", "text", 1.0)
        row = dict(id=0, j=30, prompt=p, target=481, ids=q)
        exp = g["machine_inputs"](tok, start, [row], "text")
        path = str(TMP / f"trunc_{name.split('/')[-1]}.pkl")
        if name == "TinyLlama/TinyLlama_v1.1":   # tokenizer only: a fake score file with the right shape
            arrs = {0: np.zeros((4, len(exp[0]["ids"]) - exp[0]["k"]), np.float32)}
        else:
            cfg_m, gen_m = g["model_meta"](name, None)
            pad = g["pad_id"](tok, cfg_m, gen_m, g["banned_ids"](tok, cfg_m, gen_m)[0])
            model = g["AutoModelForCausalLM"].from_pretrained(name, dtype=torch.float32).eval()
            arrs = g["score_inputs"](model, pad, [(0, exp[0]["ids"], exp[0]["k"])])
            del model
        g["save_scores"](path, cfg, exp, arrs)
        ok, why = g["check_scores"](path, cfg, exp)
        LOG.check(f"截斷檔案的續跑檢查判為正確、不移到 _invalid/：{name}", ok and g["valid_or_move"](path, (ok, why)) and os.path.exists(path), why)
        long_ids = start + tok(g["dec"](tok, p + q), add_special_tokens=False)["input_ids"]
        obj = pickle.load(open(path, "rb")); obj["rows"][0]["ids"] = np.asarray(long_ids, np.int32)
        obj["rows"][0]["arr"] = np.zeros((4, len(long_ids) - k), np.float32)
        pickle.dump(obj, open(path, "wb"))
        ok, why = g["check_scores"](path, cfg, exp)
        LOG.check(f"長度 513 的假檔案判為錯誤：{name}", not ok and len(long_ids) == 513, why)
        os.remove(path)
    # replacement and rule 9
    g["TOTAL_MAX_H"], g["MIN_FAMILIES"] = 7.5, 8
    ent = lambda no, fam, pm, ok=True: dict(no=no, name=f"m{no}", family=fam, passed=ok, reason="" if ok else "rule 5", P_m=pm)
    base = [ent(i, f"F{i}", 30) for i in range(1, 12)] + [ent(12, "F7", 30)]
    cand = ent(13, "Z", 40)
    f, ch, stop = g["apply_rules"](base, cand)
    LOG.check("規則 9：總和在 7.5 小時內時名單不變", [x["name"] for x in f] == [x["name"] for x in base] and not ch and not stop)
    b2 = copy.deepcopy(base); b2[2]["passed"] = False; b2[4]["passed"] = False
    f, ch, stop = g["apply_rules"](b2, cand)
    LOG.check("遞補：兩個模型技術失敗時只補回一個（Zamba2）", len(f) == 11 and "m13" in [x["name"] for x in f] and not stop, json.dumps(ch))
    b3 = copy.deepcopy(base); b3[10]["P_m"] = 90; b3[11]["P_m"] = 90; b3[5]["P_m"] = 60   # 510 minutes > 7.5 h
    f, ch, stop = g["apply_rules"](b3, cand)
    names = [x["name"] for x in f]
    LOG.check("規則 9：先用 Zamba2 換掉 P_m 最大的模型（同分取編號較大的），再移除最大的", ch[0]["model"] == "m12" and ch[0]["action"] == "replaced"
              and ch[1]["model"] == "m11" and ch[1]["action"] == "removed" and "m13" in names and sum(x["P_m"] for x in f) <= 450, json.dumps(ch))
    b4 = [ent(i, f"F{i}", 30, ok=i > 6) for i in range(1, 13)]   # six failures: 6 left + Zamba2 = 7 families
    f, ch, stop = g["apply_rules"](b4, cand)
    LOG.check("家族少於 8 個時停下請老闆決定", stop and len({x["family"] for x in f}) == 7, f"{len(f)} 個模型")
    b5 = [ent(i, f"F{i}", 30, ok=i > 5) for i in range(1, 13)]   # five failures: 7 left + Zamba2 = 8 families, no stop
    f, ch, stop = g["apply_rules"](b5, cand)
    LOG.check("家族剛好 8 個時不停", not stop and len({x["family"] for x in f}) == 8, f"{len(f)} 個模型")


def main():
    global WORK, DATA, OUT, REG
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(TMP / "nb15_dryrun")); ap.add_argument("--log", default=str(TMP / "nb15_dryrun_log.txt"))
    ap.add_argument("--skip-notebook", action="store_true", help="only part 2 (debugging)")
    ap.add_argument("--registration-only", action="store_true", help="only the registration steps, on an earlier run's outputs")
    a = ap.parse_args()
    log_path = pathlib.Path(a.log).resolve()   # the notebook's cell 2 changes the working directory later on
    WORK = pathlib.Path(a.work)
    os.environ["TMPDIR"] = str(WORK.parent)   # the notebook's DRY data folder is tempfile.gettempdir()/nb15_dryrun/data
    assert WORK == pathlib.Path(os.environ["TMPDIR"]) / "nb15_dryrun", "the work folder must be <TMPDIR>/nb15_dryrun"
    import tempfile; tempfile.tempdir = None
    t0 = time.time()
    LOG.say(f"notebook 15 筆電試跑｜{time.strftime('%Y-%m-%d %H:%M:%S')}｜中文版 {ZH.name} SHA-256 {sha(ZH)}｜python {sys.version.split()[0]}")
    keep = a.skip_notebook or a.registration_only
    DATA, OUT = prepare(WORK) if not keep else (WORK / "data", WORK / "data" / "15_replication_test")
    REG = WORK / "registration.json"
    nb = dry_notebook()
    try:
        if a.registration_only:
            registration_fragment(WORK)
        else:
            if not a.skip_notebook:
                part1(nb, WORK)
            part2(nb)
    except Exception:
        LOG.check("試跑沒有未預期的錯誤", False, traceback.format_exc()[-1500:])
    LOG.say(f"共 {time.time() - t0:.0f} 秒；{'全部通過' if not LOG.fails else f'{len(LOG.fails)} 項失敗：{LOG.fails}'}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(LOG.lines) + "\n", encoding="utf-8")
    return 0 if not LOG.fails else 1


if __name__ == "__main__":
    sys.exit(main())
