"""Tests for scripts/check_registration.py (plan 8.8.24, implementation list items 9 and 11) on a throwaway folder.
Time proofs are checked with the real ots client against tests/fixtures/ots_probe.txt.ots, a genuine (pending) proof made
with `ots stamp`; copying the stamped text next to it under another name keeps the proof valid for that name."""
import json, pathlib, shutil, sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"
sys.path.insert(0, str(ROOT / "scripts"))
import check_registration as reg   # noqa: E402

ENTRY = dict(no=1, name="m/a", revision="r", family="A", tok_group="G1", tokenizer_hash="h1", lang_group="other", strip_group="keep",
             start=[], banned=[0, 3], banned_sha256=reg.sha256_list([0, 3]), pad=0)
R0_REL, R1_REL = "2026-09-02T00:00:00+00:00", "2026-09-03T00:00:00+00:00"   # GitHub Release creation times (server time)
pytestmark = pytest.mark.skipif(reg.run_tool("ots", "--version")[0] is None, reason="the ots client is not installed")


@pytest.fixture
def root(tmp_path):
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests" / "m4gt_humans_PE_manifest.csv").write_text("id\n0\n", encoding="utf-8")
    (tmp_path / "protocol.md").write_text("protocol\n", encoding="utf-8")
    return tmp_path


def call(root, *args):
    return reg.main(["--registration", str(root / "registration.json"), "--root", str(root)] + list(args))


def spec(root, obj):
    p = root / "spec.json"; p.write_text(json.dumps(obj), encoding="utf-8"); return str(p)


def prove(root, sec, how="real"):
    """registration/<sec>.SHA256SUMS with its OpenTimestamps proof: real, empty, or real but for other content."""
    d = root / "registration"; d.mkdir(exist_ok=True)
    (d / f"{sec}.SHA256SUMS").write_bytes((FIX / "ots_probe.txt").read_bytes() + (b"changed" if how == "other" else b""))
    if how == "empty":
        (d / f"{sec}.SHA256SUMS.ots").write_bytes(b"")
    else:
        shutil.copy(FIX / "ots_probe.txt.ots", d / f"{sec}.SHA256SUMS.ots")


def unprove(root, sec):
    """Remove the stand-in R<sec> SHA256SUMS: `check` reads the section's own SHA256SUMS as a list of files."""
    for suffix in ("", ".ots"):
        (root / "registration" / f"{sec}.SHA256SUMS{suffix}").unlink()


def make_r0(root):
    s = dict(protocol="protocol.md", files=["protocol.md", "manifests/m4gt_humans_PE_manifest.csv"], seeds={"perm": 20260930},
             packages={"transformers": "5.17.0"}, decisions=dict(criterion2="form R", equivalence_margin=0.002, alpha="split",
                                                                  gpu_budget_hours=8, candidate="Zyphra/Zamba2-1.2B", osf=False))
    assert call(root, "make", "R0", "--spec", spec(root, s)) == 0


def make_r1(root, models=None, stage0_start="2026-09-02T06:00:00+00:00"):
    draft = models or [ENTRY, dict(ENTRY, no=2, name="m/b", family="B", tok_group="G2", tokenizer_hash="h2")]
    rec = dict(r1_models_draft=draft, gpu_checks={m["name"]: dict(P_m=30.0) for m in draft}, versions={"transformers": "5.17.0"},
               environment={"python": "3.12", "torch": "2.8.0"}, changes=[], entries=[],
               tokenizer_checks={m["name"]: dict(utc=stage0_start) for m in draft})
    (root / "stage0_record.json").write_text(json.dumps(rec), encoding="utf-8")
    prev = dict(previous=dict(tag="prereg-R0", commit="c0", release_created_at=R0_REL))
    assert call(root, "make", "R1", "--spec", spec(root, prev), "--stage0", str(root / "stage0_record.json")) == 0


def r1_models(root, *extra):
    return call(root, "r1-models", "--out", str(root / "r1_models.json"), "--r1-commit", "c1", "--skip-gh",
                "--ots-r0", str(root / "registration" / "R0.SHA256SUMS.ots"), "--ots-r1", str(root / "registration" / "R1.SHA256SUMS.ots"), *extra)


def check_r1(root, *extra):
    return call(root, "check", "R1", "--previous", str(root / "r0.json"), *extra)


def test_r0_make_check_and_sums(root):
    make_r0(root)
    assert call(root, "R0") == 0
    assert call(root, "sums", "R0") == 0 and (root / "registration" / "R0.SHA256SUMS").exists()
    assert call(root, "check", "R0") == 0
    (root / "protocol.md").write_text("changed\n", encoding="utf-8")
    assert call(root, "check", "R0") == 1


def test_sections_are_append_only(root):
    make_r0(root)
    with pytest.raises(SystemExit):
        call(root, "make", "R0", "--spec", spec(root, {}))
    shutil.copy(root / "registration.json", root / "r0.json"); make_r1(root); prove(root, "R0"); prove(root, "R1")
    assert r1_models(root) == 0
    unprove(root, "R1")
    args = ["--r1-models", str(root / "r1_models.json"), "--stage0", str(root / "stage0_record.json")]
    assert check_r1(root, *args) == 0
    data = json.loads((root / "registration.json").read_text(encoding="utf-8")); data["R0"]["seeds"]["perm"] = 1
    (root / "registration.json").write_text(json.dumps(data), encoding="utf-8")
    assert check_r1(root, *args) == 1


@pytest.mark.parametrize("bad", ["empty", "other", "missing"])
def test_r1_models_needs_real_proofs_of_r0_and_r1(root, bad):
    """Review A3: stage 1 waits for the R0 and R1 Releases and real (stamped, not necessarily upgraded) proofs; an empty file,
    a proof of other content or a missing file does not pass."""
    make_r0(root); make_r1(root); prove(root, "R0")
    if bad != "missing":
        prove(root, "R1", how=bad)
    assert r1_models(root) == 1 and not (root / "r1_models.json").exists()
    prove(root, "R1")
    assert r1_models(root) == 0 and json.loads((root / "r1_models.json").read_text())["r1_commit"] == "c1"


def test_check_r1_needs_stage0_and_a_real_r0_proof(root):
    """Review A3: `check R1` without --stage0 fails; so does an R0 proof that is not real; stage 0 must follow R0's Release."""
    make_r0(root); shutil.copy(root / "registration.json", root / "r0.json"); make_r1(root); prove(root, "R0")
    st = str(root / "stage0_record.json")
    assert check_r1(root) == 1 and check_r1(root, "--stage0", st) == 0
    prove(root, "R0", how="empty")
    assert check_r1(root, "--stage0", st) == 1
    prove(root, "R0")
    early = json.loads((root / "stage0_record.json").read_text())
    for v in early["tokenizer_checks"].values():
        v["utc"] = "2026-09-01T00:00:00+00:00"                                             # before R0's Release
    (root / "stage0_record.json").write_text(json.dumps(early), encoding="utf-8")
    assert check_r1(root, "--stage0", st) == 1


def test_r1_table_and_groups(root):
    make_r0(root); shutil.copy(root / "registration.json", root / "r0.json"); make_r1(root); prove(root, "R0"); prove(root, "R1")
    assert r1_models(root) == 0
    unprove(root, "R1")
    assert check_r1(root, "--stage0", str(root / "stage0_record.json"), "--r1-models", str(root / "r1_models.json")) == 0
    r1m = json.loads((root / "r1_models.json").read_text()); r1m["models"][0]["pad"] = 5
    (root / "bad_r1.json").write_text(json.dumps(r1m), encoding="utf-8")
    assert check_r1(root, "--stage0", str(root / "stage0_record.json"), "--r1-models", str(root / "bad_r1.json")) == 1


def test_r1_group_must_match_tokenizer_hash(root):
    make_r0(root); shutil.copy(root / "registration.json", root / "r0.json"); prove(root, "R0")
    make_r1(root, models=[ENTRY, dict(ENTRY, no=2, name="m/b", family="B", tok_group="G1", tokenizer_hash="h2")])   # one group, two hashes
    assert check_r1(root, "--stage0", str(root / "stage0_record.json")) == 1


def test_r2_and_go(root):
    make_r0(root); make_r1(root); prove(root, "R0"); prove(root, "R1")
    preds = root / "predictions.json"; preds.write_text(json.dumps(dict(models={"m/a": dict(status="written")})), encoding="utf-8")
    sha = reg.sha256_file(preds)
    prev = spec(root, dict(previous=dict(tag="prereg-R1", commit="c1", release_created_at=R1_REL)))
    with pytest.raises(SystemExit):                       # the hash printed on Colab must equal the synced file's hash
        call(root, "make", "R2", "--spec", prev, "--predictions", str(preds), "--colab-sha", "0" * 64)
    assert call(root, "make", "R2", "--spec", prev, "--predictions", str(preds), "--colab-sha", sha) == 0
    data = json.loads((root / "registration.json").read_text()); del data["R2"]; (root / "r1.json").write_text(json.dumps(data))
    att = root / "attempts.json"
    c2 = ["check", "R2", "--previous", str(root / "r1.json")]
    att.write_text(json.dumps({"stage1": {"m/a": [dict(start="2026-09-03T12:00:00+00:00")]}}), encoding="utf-8")
    assert call(root, *c2) == 1                                         # review A3: --attempts is required
    assert call(root, *c2, "--attempts", str(att)) == 0
    att.write_text(json.dumps({"stage1": {"m/a": [dict(start="2026-09-02T12:00:00+00:00")]}}), encoding="utf-8")   # before R1's Release
    assert call(root, *c2, "--attempts", str(att)) == 1
    prove(root, "R2")
    go = ["go", "--predictions", str(preds), "--r2-commit", "c2", "--block-height", "3", "--block-time", "2026-09-04T00:00:00+00:00",
          "--out", str(root / "stage2_go.json"), "--skip-gh", "--ots", str(root / "registration" / "R2.SHA256SUMS.ots")]
    assert call(root, *go, "--colab-sha", sha) == 1 and not (root / "stage2_go.json").exists()   # R2: real but not upgraded
    future = [x if x != "2026-09-04T00:00:00+00:00" else "2999-01-01T00:00:00+00:00" for x in go]
    assert call(root, *future, "--colab-sha", sha, "--skip-upgrade") == 1 and not (root / "stage2_go.json").exists()
    assert call(root, *go, "--colab-sha", "0" * 64, "--skip-upgrade") == 1 and not (root / "stage2_go.json").exists()
    prove(root, "R2", how="empty")
    assert call(root, *go, "--colab-sha", sha, "--skip-upgrade") == 1                          # even the dry run needs a real proof
    prove(root, "R2")
    assert call(root, *go, "--colab-sha", sha, "--skip-upgrade") == 0
    g = json.loads((root / "stage2_go.json").read_text(encoding="utf-8"))
    assert list(g) == list(reg.GO_FIELDS) and g["predictions_sha256"] == sha and g["ots_block_height"] == 3
    preds.write_text(preds.read_text(encoding="utf-8") + " ", encoding="utf-8")   # one byte more: refused
    (root / "stage2_go.json").unlink()
    assert call(root, *go, "--colab-sha", sha, "--skip-upgrade") == 1 and not (root / "stage2_go.json").exists()


def test_env_update_needs_a_revision_note(root):
    """Plan 5.4 recovery after a Colab image change: environment.json only changes through a revision note in the repository
    that names the reason, the date and the old and new versions (and, outside tests, is committed and pushed)."""
    env = root / "environment.json"
    old = dict(python="3.12.11", torch="2.8.0+cu126", numpy="2.0.2", scipy="1.16.2", cuda="12.6")
    new = dict(old, torch="2.9.0+cu126", numpy="2.1.0")
    env.write_text(json.dumps(old), encoding="utf-8")
    note = root / "docs" / "revisions" / "colab.md"; note.parent.mkdir(parents=True)
    upd = ["env-update", "--environment", str(env), "--new", json.dumps(new), "--note", "docs/revisions/colab.md"]
    assert call(root, *upd, "--skip-git") == 1                                                  # no note yet
    note.write_text("2026-10-20 Colab 更新映像。原因：預裝版本改變。torch 2.8.0+cu126 → 2.9.0+cu126\n", encoding="utf-8")
    assert call(root, *upd, "--skip-git") == 1 and json.loads(env.read_text()) == old         # numpy versions not written
    note.write_text(note.read_text(encoding="utf-8") + "numpy 2.0.2 → 2.1.0\n", encoding="utf-8")
    assert call(root, *upd) == 1 and json.loads(env.read_text()) == old                        # not committed (no git here)
    assert call(root, *upd, "--skip-git") == 0 and json.loads(env.read_text()) == new
    hist = json.loads((root / "environment_history.json").read_text(encoding="utf-8"))
    assert hist[-1]["old"] == old and hist[-1]["new"] == new and hist[-1]["note"] == "docs/revisions/colab.md"
    assert call(root, *upd, "--skip-git") == 1                                                  # the same versions again: nothing to do
