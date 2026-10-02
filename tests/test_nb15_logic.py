"""Notebook 15's own stage logic, run outside a kernel (cells 1-5 executed in DRYRUN mode in this process).
Covers code review A1-A4, B1-B4, B7: temporary download failures are retried and never use the candidate; finished models
are never failed by the rerun limit or the 14-day limit; a registered file that no longer passes its check stops the run;
version checks; one fallback-warning record per model; tokenizer groups frozen from the tokenizer.json hashes."""
import json, logging, os, pathlib, pickle, shutil, sys, tempfile, types

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
GDRIVE = pathlib.Path("/Users/xuyunqin/Library/CloudStorage/GoogleDrive-emilyhuang12380@gmail.com/我的雲端硬碟/AI-Text/03_實驗")
ZH = GDRIVE / "notebooks" / "15_replication_test.ipynb"
pytestmark = pytest.mark.skipif(not (ZH.exists() and (GDRIVE / "data" / "external" / "m4gt" / "SubtaskB.jsonl").exists()),
                                reason="needs the Drive notebook and data")


@pytest.fixture(scope="module")
def nb(tmp_path_factory):
    from IPython.core.inputtransformer2 import TransformerManager
    base = tmp_path_factory.mktemp("nb15"); old_tmp, old_cwd = tempfile.tempdir, os.getcwd()
    tempfile.tempdir = str(base)
    data = base / "nb15_dryrun" / "data"; (data / "external" / "m4gt").mkdir(parents=True); (data / "manifests").mkdir()
    os.symlink(GDRIVE / "data" / "external" / "m4gt" / "SubtaskB.jsonl", data / "external" / "m4gt" / "SubtaskB.jsonl")
    for f in ("m4gt_humans_PE_manifest.csv", "m4gt_humans_A_manifest.csv"):
        shutil.copy(GDRIVE / "data" / "manifests" / f, data / "manifests" / f)
    cells = json.loads(ZH.read_text(encoding="utf-8"))["cells"]
    g, tm = {"__name__": "nb15"}, TransformerManager()
    for i in range(1, 6):
        src = "".join(cells[i]["source"])
        if i == 1:
            src = src.replace("DRYRUN = False", "DRYRUN = True", 1)
        exec(compile(tm.transform_cell(src), f"<cell {i}>", "exec"), g)
    g["R1_COMMIT"] = "test-r1"; g["DATA"] = data
    os.chdir(old_cwd)
    yield g
    tempfile.tempdir = old_tmp


@pytest.fixture
def out(nb, monkeypatch):
    """A clean output folder for each test, and the notebook's working directory."""
    monkeypatch.chdir(nb["DATA"])
    o = nb["DATA"] / nb["OUT"]
    for p in o.glob("*"):
        if p.name != "environment.json":
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    return o


def entry(nb, name="gpt2"):
    tok = nb["load_tokenizer"](name, None); cfg, gen = nb["model_meta"](name, None)
    start, _ = nb["start_tokens"](tok); banned, _, _ = nb["banned_ids"](tok, cfg, gen)
    return dict(no=1, name=name, revision=None, family="F", tok_group="D1", start=start, banned=banned,
                banned_sha256=nb["sha256_list"](banned), pad=nb["pad_id"](tok, cfg, gen, banned)), tok


def fake_human_file(nb, e, tok):
    exp = nb["human_inputs"](tok, e["start"], nb["P_IDS"] + nb["E_IDS"])
    arrs = {i: np.zeros((4, len(x["ids"]) - x["k"]), np.float32) for i, x in exp.items()}
    nb["save_scores"](nb["human_file"](e), nb["file_config"](e, "human", "text", None), exp, arrs)


def fake_stage2_files(nb, e, tok):
    items = nb["prompt_items"](tok, nb["PROMPT_IDS"])
    for t in nb["TEMPS"]:
        gpath, tpath, ipath = nb["stage2_files"](e, t); gcfg = nb["file_config"](e, "gen", "gen", float(t))
        rows = [dict(id=b["id"], j=b["j"], prompt=b["prompt"], target=b["target"], ids=[262] * b["target"]) for b in items]
        nb["write_json"](gpath, dict(schema=nb["SCHEMA"], kind="gen", model=e["name"], temperature=float(t), config=gcfg,
                                     config_hash=nb["config_hash"](gcfg), written_utc=nb["utc"](), environment=nb["ENV"], rows=rows))
        for path, kind in ((tpath, "text"), (ipath, "ids")):
            exp = nb["machine_inputs"](tok, e["start"], rows, kind)
            nb["save_scores"](path, nb["file_config"](e, "machine", kind, float(t)), exp,
                              {i: np.zeros((4, len(x["ids"]) - x["k"]), np.float32) for i, x in exp.items()})


def failed(nb, stage, name, n):
    nb["write_json"](nb["ATTEMPTS_FILE"], {stage: {name: [dict(start="2026-10-01T00:00:00+00:00", end="x", ok=False, error="OSError")] * n}})


def boom(*a, **k):
    raise OSError("We couldn't connect to 'https://huggingface.co'")


# ---------------------------------------------------------------- A1 / B1: stage 0
def test_gpu_check_download_failure_is_a_retry_not_a_failure(nb, out, monkeypatch):
    e, tok = entry(nb)
    t = dict(passed=True, revision=None, start=e["start"], banned=e["banned"], pad=e["pad"], vocab_size=50257, S_m=100)
    monkeypatch.setitem(nb, "load_model", boom)
    g = nb["gpu_check"](dict(no=1, name="gpt2", family="F"), t)
    assert "load_error" in g and "passed" not in g and "error" not in g
    assert nb["stage0_entry"](dict(no=1, name="gpt2", family="F"), t, g) is None          # 0c retries; the candidate is untouched
    assert nb["stage0_entry"](dict(no=1, name="gpt2", family="F"), dict(load_error="x"), None) is None
    monkeypatch.setitem(nb, "DECLARED_LOAD_FAILURES", {"gpt2": "7 days without loading"})
    r = nb["stage0_entry"](dict(no=1, name="gpt2", family="F"), t, g)
    assert r["passed"] is False and r["reason"].startswith("rule 1")


def test_rule8_failure_reasons(nb, monkeypatch):
    monkeypatch.setitem(nb, "PM_MAX_MIN", 120)
    t = dict(passed=True)
    for g, why in ((dict(passed=False, generation_ok=True, P_m=150.0, peak_share=0.5), "P_m 150 > 120"),
                   (dict(passed=False, generation_ok=False, P_m=10.0, peak_share=0.5), "generation check"),
                   (dict(passed=False, error="OutOfMemoryError: CUDA"), "OutOfMemoryError")):
        r = nb["stage0_entry"](dict(no=1, name="m", family="F"), t, g)
        assert r["passed"] is False and r["reason"].startswith("rule 8") and why in r["reason"]


# ---------------------------------------------------------------- A2 / B4 / B3: stages 1 and 2
def test_stage1_finished_model_is_done_whatever_its_attempts(nb, out, monkeypatch):
    e, tok = entry(nb)
    fake_human_file(nb, e, tok); failed(nb, "stage1", "gpt2", 4)
    monkeypatch.setitem(nb, "load_model", boom)                     # must not be called for a finished model
    assert nb["stage1_model"](e) == "done"


def test_stage1_tokenizer_download_failure_is_not_an_attempt(nb, out, monkeypatch):
    e, tok = entry(nb)
    monkeypatch.setitem(nb, "load_tokenizer", boom)
    assert nb["stage1_model"](e).startswith("retry") and not os.path.exists(nb["ATTEMPTS_FILE"])


def test_stage1_gives_up_only_on_unfinished_work(nb, out, monkeypatch):
    e, tok = entry(nb); failed(nb, "stage1", "gpt2", 4)
    monkeypatch.setitem(nb, "load_model", boom)
    assert nb["stage1_model"](e).startswith("failed")


def test_stage1_never_rescores_once_predictions_exist(nb, out, monkeypatch):
    e, tok = entry(nb); fake_human_file(nb, e, tok)
    path = nb["human_file"](e); obj = pickle.load(open(path, "rb")); obj["rows"][0]["arr"][0, 0] = np.nan
    pickle.dump(obj, open(path, "wb")); nb["write_json"](nb["PRED_FILE"], {"models": {"gpt2": {"status": "written"}}})
    monkeypatch.setitem(nb, "load_model", boom)
    with pytest.raises(nb["StageStop"]):
        nb["stage1_model"](e)
    assert os.path.exists(path) and not (out / "_invalid").exists()       # kept in place for inspection, not moved or redone


def test_stage2_finished_model_is_done_after_the_deadline(nb, out, monkeypatch):
    e, tok = entry(nb); fake_stage2_files(nb, e, tok); failed(nb, "stage2", "gpt2", 4)
    monkeypatch.setitem(nb, "PREDS", {"models": {"gpt2": {"status": "written"}}})
    monkeypatch.setitem(nb, "DEADLINE", 0.0)                         # 14 days long gone
    monkeypatch.setitem(nb, "load_model", boom)
    assert nb["stage2_model"](e) == "done"
    os.remove(nb["stage2_files"](e, nb["TEMPS"][-1])[2])              # one file missing: new work is needed, so the limits apply
    assert nb["stage2_model"](e).startswith("failed")


# ---------------------------------------------------------------- A4 / B2: versions
def test_stale_module_and_environment_checks(nb, out, monkeypatch):
    monkeypatch.setitem(nb, "MODULES", {"fakepkg": "fakepkg"})
    monkeypatch.setitem(sys.modules, "fakepkg", types.SimpleNamespace(__version__="1.0"))
    assert nb["stale_modules"]({"fakepkg": "2.0"}) == {"fakepkg": ("1.0", "2.0")} and nb["stale_modules"]({"fakepkg": "1.0"}) == {}
    env = dict(python="3.12.13", torch="2.8.0+cu126", numpy="2.0.2", scipy="1.16.2", cuda="12.6")
    os.remove(nb["ENV_FILE"]) if os.path.exists(nb["ENV_FILE"]) else None
    assert nb["check_environment"](env) == env and json.loads(open(nb["ENV_FILE"]).read()) == env    # first run records it
    with pytest.raises(AssertionError):
        nb["check_environment"](dict(env, python="3.12.14"))
    os.remove(nb["ENV_FILE"])
    assert "python" not in nb["VERSIONS"] and nb["file_config"](entry(nb)[0], "human", "text", None)["packages"] == nb["PINS"]
    assert set(nb["PINS"]) == {"transformers", "tokenizers", "huggingface-hub", "safetensors", "accelerate"}
    assert all(nb["VERSIONS"][k] == v for k, v in nb["PINS"].items())


# ---------------------------------------------------------------- B7: one fallback record per model
def test_fallback_warnings_are_kept_for_every_model(nb):
    log = logging.getLogger("transformers.integrations.hub_kernels")
    msg = "`causal_conv1d_fn` is falling back to its reference PyTorch implementation because `causal-conv1d` is not installed."
    nb["reset_fallback"](); log.warning_once(msg); log.warning_once(msg)
    assert nb["FALLBACK"].msgs == [msg]                                # printed once per process
    nb["reset_fallback"](); log.warning_once(msg)
    assert nb["FALLBACK"].msgs == [msg]                                # the next model gets its own copy


# ---------------------------------------------------------------- A6 / B5: tokenizer groups
def test_tokenizer_groups_are_frozen_from_hashes(nb):
    rec = lambda no, name, prov, h: dict(no=no, name=name, provisional_group=prov, tokenizer_hash=h)
    labels, notes = nb["tokenizer_groups"]([rec(4, "RedPajama", "G4", "a"), rec(5, "StableLM", "G4", "a"), rec(11, "Mamba", "G4", "a"),
                                            rec(6, "Danube2", "G5", "b"), rec(7, "Danube3", "G5", "b"), rec(13, "Zamba2", None, "c")])
    assert labels == {"RedPajama": "G4", "StableLM": "G4", "Mamba": "G4", "Danube2": "G5", "Danube3": "G5", "Zamba2": "G6"}
    assert [n["models"] for n in notes] == [["Zamba2"]]
    labels, notes = nb["tokenizer_groups"]([rec(6, "Danube2", "G5", "b"), rec(7, "Danube3", "G5", "x")])   # one label, two hashes
    assert labels["Danube2"] != labels["Danube3"] and "G5" not in labels.values() and len(notes) == 2


# ---------------------------------------------------------------- re-review: B3, A2 note, B6, A4/B2 note
def test_stage1_missing_human_file_after_predictions_stops(nb, out, monkeypatch):
    e, tok = entry(nb)
    nb["write_json"](nb["PRED_FILE"], {"models": {"gpt2": {"status": "written"}}})
    monkeypatch.setitem(nb, "load_model", boom)                     # must never be reached
    with pytest.raises(nb["StageStop"]):
        nb["stage1_model"](e)


def test_stage1_after_predictions_keeps_recorded_failures_and_host_declarations(nb, out, monkeypatch):
    e, tok = entry(nb)
    monkeypatch.setitem(nb, "load_tokenizer", boom)                 # neither case may load anything
    nb["write_json"](nb["PRED_FILE"], {"models": {"gpt2": {"status": "not_written", "reason": "earlier failure"}}})
    assert nb["stage1_model"](e) == "failed: earlier failure"
    os.remove(nb["PRED_FILE"])
    monkeypatch.setitem(nb, "DECLARED_STAGE1_FAILURES", {"gpt2": "repository taken down"})
    r = nb["stage1_model"](e)
    assert r.startswith("failed: ") and "repository taken down" in r


def test_registered_constants_of_the_notebook(nb, monkeypatch):
    """Review B6: changing a registered number in the notebook must fail a test (cells 4 and 5 with DRYRUN False)."""
    from IPython.core.inputtransformer2 import TransformerManager
    monkeypatch.chdir(nb["DATA"])
    monkeypatch.setitem(nb, "DRYRUN", False)          # setting() reads DRYRUN from the namespace it was defined in
    g = dict(nb)
    cells = json.loads(ZH.read_text(encoding="utf-8"))["cells"]
    for i in (4, 5):
        exec(compile(TransformerManager().transform_cell("".join(cells[i]["source"])), f"<cell {i}>", "exec"), g)
    assert g["TEMPS"] == (0.94, 0.97, 0.99, 1.00, 1.01, 1.02, 1.03, 1.04, 1.06, 1.09, 1.14)
    assert (g["PREFIX"], g["MAX_RAW"], g["MAX_TOK"], g["BS"], g["SBS"], g["SEED"]) == (30, 511, 512, 32, 16, 42)
    assert (g["MIN_TOKENS"], g["MIN_SCORED"], g["ROUNDTRIP_MIN"], g["PEAK_MAX"]) == (50, 20, 0.99, 0.80)
    assert (g["MAX_ATTEMPTS"], g["STAGE2_DAYS"], g["PM_MAX_MIN"], g["TOTAL_MAX_H"], g["MIN_FAMILIES"]) == (4, 14, 120, 7.5, 8)
    assert (g["SLOPE_K"], g["LINE_A"], g["LINE_K"], g["KNOWN_MEAN"], g["VARIANT_B_RANGE"]) == (0.1451, 1.0004, 0.1409, 1.0089, (0.80, 1.30))
    assert g["GPU_CHECK"] == dict(prompts=32, new_tokens=64, context=400, steps=64, score_batch=16, score_len=512, repeats=3)
    assert str(g["DTYPE"]) == "torch.bfloat16" and len(g["MODELS"]) == 12 and g["CANDIDATE"]["name"] == "Zyphra/Zamba2-1.2B"
    assert len({m["family"] for m in g["MODELS"]}) == 11 and g["DECLARED_LOAD_FAILURES"] == {} and g["DECLARED_STAGE1_FAILURES"] == {}
    assert sum(q[0] for q in g["QUOTA"].values()) == 1000 and sum(q[1] for q in g["QUOTA"].values()) == 1000
    assert all(q[2:] == (25, 25) for q in g["QUOTA"].values())


def test_environment_change_half_way_redoes_the_model(nb, out, monkeypatch):
    """Plan 5.4: a model caught half-done by a Colab image change is redone from scratch (the work counts as attempts)."""
    e, tok = entry(nb)
    old = dict(nb["ENV"], torch="0.0.0-old")
    monkeypatch.setitem(nb, "ENV", old); fake_stage2_files(nb, e, tok)
    monkeypatch.setitem(nb, "ENV", dict(old, torch="9.9.9-new"))
    os.remove(nb["stage2_files"](e, nb["TEMPS"][-1])[2])              # the model is not finished
    monkeypatch.setitem(nb, "PREDS", {"models": {"gpt2": {"status": "written"}}})
    monkeypatch.setitem(nb, "DEADLINE", 4e9); monkeypatch.setitem(nb, "load_model", boom)
    assert nb["stage2_model"](e).startswith("retry")
    assert not nb["stage2_paths"](e)                                   # every old-environment file was moved away
    log = [json.loads(x) for x in (out / "_invalid" / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(log) == 5 and all("執行環境改變" in x["reason"] for x in log)
    assert len(nb["failed_attempts"]("stage2", "gpt2")) == 1           # the redo counts against the rerun limit


def test_model_finished_in_the_old_environment_is_not_redone(nb, out, monkeypatch):
    e, tok = entry(nb)
    monkeypatch.setitem(nb, "ENV", dict(nb["ENV"], torch="0.0.0-old")); fake_stage2_files(nb, e, tok)
    monkeypatch.setitem(nb, "ENV", dict(nb["ENV"], torch="9.9.9-new"))
    monkeypatch.setitem(nb, "PREDS", {"models": {"gpt2": {"status": "written"}}})
    monkeypatch.setitem(nb, "DEADLINE", 4e9); monkeypatch.setitem(nb, "load_model", boom)
    assert nb["stage2_model"](e) == "done" and not (out / "_invalid").exists()
    assert nb["file_config"](e, "gen", "gen", 1.0)["packages"] == nb["PINS"]           # the environment is not hashed


def test_damaged_file_is_not_an_environment_change(nb, out, monkeypatch):
    """A file that cannot be read is moved for its own reason. It must not crash the environment check, and the model's
    good files must not be redone as if the environment had changed (found by the 5.4 dry run)."""
    e, tok = entry(nb)
    fake_stage2_files(nb, e, tok)
    bad = nb["stage2_files"](e, nb["TEMPS"][-1])[2]
    with open(bad, "wb") as f:
        f.write(b"damaged")
    monkeypatch.setitem(nb, "PREDS", {"models": {"gpt2": {"status": "written"}}})
    monkeypatch.setitem(nb, "DEADLINE", 4e9); monkeypatch.setitem(nb, "load_model", boom)
    assert nb["stage2_model"](e).startswith("retry")                  # only the damaged file needs the model
    log = [json.loads(x) for x in (out / "_invalid" / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [x["file"] for x in log] == [os.path.basename(bad)] and "讀不出來" in log[0]["reason"]
    assert len(nb["stage2_paths"](e)) == 3 * len(nb["TEMPS"]) - 1         # the good files stay
