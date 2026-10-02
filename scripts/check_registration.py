"""Registration records of the replication test (plan 8.8.24, implementation list items 9 and 11).

registration/registration.json has three sections added in order (R0, R1, R2); a later registration may only add a
section, never change an earlier one. Commands (run from the repository root):
  check R0|R1|R2           required fields; every listed file's SHA-256; earlier sections identical to the previous tag
                           (git show prereg-R0:registration/registration.json, or --previous FILE); the earlier registrations'
                           OpenTimestamps proofs (ots info reads a proof of that SHA256SUMS); registration/<R>.SHA256SUMS, if
                           present, matches and lists registration.json itself. R1 needs --stage0 stage0_record.json (its hash, and
                           stage 0 began after R0's Release); R2 needs --attempts attempts.json (stage 1 began after R1's Release);
                           --r1-models compares R1's model table with r1_models.json. "check_registration.py R0" means "check R0".
  make R0|R1|R2 --spec F   append a section from a JSON spec written by the host (hashes and UTC time are added here).
  sums R0|R1|R2 [FILE...]  write registration/<R>.SHA256SUMS: R0's listed files, or the FILEs given for R1 and R2, plus
                           registration.json.
  r1-models --out F --r1-commit C
                           save R1's model table as r1_models.json (the start condition of stage 1), only after the prereg-R0 and
                           prereg-R1 Releases exist and both OpenTimestamps proofs are real (ots info reads a proof of that
                           SHA256SUMS; an empty or foreign .ots fails, plan 5.4). They need not be upgraded yet: R0 and R1 are
                           ordered by GitHub's server times (boss, 2026-10-01; plan 5.3).
  go ...                   stage2_go.json, only after the prereg-R2 Release exists and the R2 proof is upgraded to a Bitcoin block
                           whose time is earlier than now (item 11).
  env-update --environment E --new JSON --note N
                           Colab changed its preinstalled versions (plan 5.4 recovery): only with a revision note in the repository
                           that gives the reason, the old and new versions and the date, committed and pushed; the old record is
                           kept in environment_history.json. Nothing in the notebook lets a changed environment through by itself.
Exit code 0 means every check passed.
"""
import argparse, hashlib, json, os, pathlib, re, shutil, subprocess, sys
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
SECTIONS = ("R0", "R1", "R2")
TAGS = {s: f"prereg-{s}" for s in SECTIONS}
REQUIRED = {
    "R0": ("utc", "protocol", "decisions", "files", "pe_manifest_sha256", "seeds", "packages"),
    "R1": ("utc", "previous", "models", "packages", "environment", "P_m_min", "P_m_total_min", "changes", "stage0_record_sha256"),
    "R2": ("utc", "previous", "predictions_sha256", "predictions_sha256_colab", "not_written"),
}
MODEL_FIELDS = ("no", "name", "revision", "family", "tok_group", "tokenizer_hash", "lang_group", "strip_group", "start", "banned",
                "banned_sha256", "pad")
PREVIOUS_FIELDS = ("tag", "commit", "release_created_at")   # GitHub's server time orders R0 -> stage 0 -> R1 -> stage 1
DECISIONS = ("criterion2", "equivalence_margin", "alpha", "gpu_budget_hours", "candidate", "osf")
GO_FIELDS = ("predictions_sha256", "r2_tag", "r2_commit", "ots_block_height", "ots_block_time")


def sha256_file(path):
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            d.update(block)
    return d.hexdigest()


def sha256_list(xs):
    return hashlib.sha256(json.dumps([int(x) for x in xs], sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def when(text):
    """ISO 8601 time with a time zone (block times are written this way in the specs)."""
    t = datetime.fromisoformat(str(text))
    assert t.tzinfo is not None, f"{text}: 時間要帶時區"
    return t


def load(path):
    return json.load(open(path, encoding="utf-8")) if pathlib.Path(path).exists() else {}


def save(path, obj):
    path = pathlib.Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class Problems(list):
    def need(self, cond, msg):
        if not cond:
            self.append(msg)


# ------------------------------------------------------------------ check
def check_section(reg, sec, root, problems):
    s = reg.get(sec)
    problems.need(s is not None, f"{sec}: 段落不存在")
    if s is None:
        return
    for key in REQUIRED[sec]:
        problems.need(key in s, f"{sec}: 缺少欄位 {key}")
    if sec != "R0":
        prev = s.get("previous", {})
        for key in PREVIOUS_FIELDS:
            problems.need(key in prev, f"{sec}: previous 缺少 {key}")
        problems.need(prev.get("tag") == TAGS[SECTIONS[SECTIONS.index(sec) - 1]], f"{sec}: previous.tag 應為 {TAGS[SECTIONS[SECTIONS.index(sec) - 1]]}")
    if sec == "R0":
        for key in DECISIONS:
            problems.need(key in s.get("decisions", {}), f"R0: decisions 缺少 {key}")
        files = s.get("files", {})
        protocol = s.get("protocol", {})
        problems.need(protocol.get("path") in files, "R0: 協定檔要列在 files 裡")
        manifest = [p for p in files if p.endswith("m4gt_humans_PE_manifest.csv")]
        problems.need(len(manifest) == 1 and files.get(manifest[0] if manifest else "") == s.get("pe_manifest_sha256"),
                      "R0: pe_manifest_sha256 要等於 files 裡 manifest 的雜湊")
        for p, h in files.items():
            f = root / p
            problems.need(f.exists(), f"R0: 檔案不存在 {p}")
            if f.exists():
                problems.need(sha256_file(f) == h, f"R0: SHA-256 不符 {p}")
    if sec == "R1":
        models = s.get("models", [])
        problems.need(len(models) > 0, "R1: 模型表是空的")
        for m in models:
            for key in MODEL_FIELDS:
                problems.need(key in m and m[key] is not None, f"R1: {m.get('name', '?')} 缺少 {key}")
            if "banned" in m and "banned_sha256" in m:
                problems.need(sha256_list(m["banned"]) == m["banned_sha256"], f"R1: {m.get('name')} 的禁止清單和雜湊不符")
        groups = {}
        for m in models:
            groups.setdefault(m.get("tok_group"), set()).add(m.get("tokenizer_hash"))
        hashes = {}
        for m in models:
            hashes.setdefault(m.get("tokenizer_hash"), set()).add(m.get("tok_group"))
        problems.need(all(len(v) == 1 for v in groups.values()) and all(len(v) == 1 for v in hashes.values()),
                      "R1: tokenizer 分群要和 tokenizer.json 雜湊一一對應（同群同雜湊、不同群不同雜湊）")
        problems.need(set(s.get("P_m_min", {})) >= {m.get("name") for m in models}, "R1: 每個名單上的模型都要有 P_m")
        problems.need(abs(sum(s.get("P_m_min", {}).get(m.get("name"), 0) for m in models) - s.get("P_m_total_min", -1)) < 1e-6,
                      "R1: P_m_total_min 要等於名單上模型的 P_m 總和")
    if sec == "R2":
        problems.need(s.get("predictions_sha256") == s.get("predictions_sha256_colab"),
                      "R2: Colab 印出的與雲端硬碟同步後重算的 predictions.json 雜湊不同")
    if sec != "R0":
        try:
            when(s.get("previous", {}).get("release_created_at"))
        except Exception:                                      # noqa: BLE001
            problems.append(f"{sec}: previous.release_created_at 要是帶時區的 ISO 8601 時間（gh release view 的 createdAt）")


def started_after(problems, label, times, released):
    """The step that follows a registration must start after that registration's GitHub Release (server time)."""
    try:
        first, rel = min(when(t) for t in times), when(released)
    except Exception as err:                                    # noqa: BLE001
        problems.append(f"{label}: 無法比較開始時間與 Release 建立時間（{err}）"); return
    problems.need(first > rel, f"{label} 的開始時間 {first.isoformat()} 早於上一次登錄的 Release 建立時間 {rel.isoformat()}")


def previous_version(root, sec, previous):
    """registration.json as it was at the previous registration's tag."""
    if previous:
        return load(previous)
    tag = TAGS[SECTIONS[SECTIONS.index(sec) - 1]]
    out = subprocess.run(["git", "-C", str(root), "show", f"{tag}:registration/registration.json"], capture_output=True, text=True)
    return json.loads(out.stdout) if out.returncode == 0 else None


def check_sums(root, reg_path, sec, problems):
    sums = root / "registration" / f"{sec}.SHA256SUMS"
    if not sums.exists():
        return False
    listed = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        m = re.fullmatch(r"([0-9a-f]{64}) [ *](.+)", line.strip())
        problems.need(m is not None, f"{sums.name}: 無法解析 {line!r}")
        if m:
            listed[m.group(2)] = m.group(1)
    for p, h in listed.items():
        f = root / p
        problems.need(f.exists() and sha256_file(f) == h, f"{sums.name}: {p} 的 SHA-256 不符或檔案不存在")
    rel = os.path.relpath(reg_path, root)
    problems.need(rel in listed, f"{sums.name}: 沒有列出 {rel}")
    return True


def cmd_check(a):
    root = pathlib.Path(a.root).resolve(); reg_path = pathlib.Path(a.registration)
    reg = load(reg_path); problems = Problems(); sec = a.section
    for earlier in SECTIONS[:SECTIONS.index(sec) + 1]:
        check_section(reg, earlier, root, problems)
    if sec != "R0":
        prev = previous_version(root, sec, a.previous)
        problems.need(prev is not None, f"讀不到上一次登錄（{TAGS[SECTIONS[SECTIONS.index(sec) - 1]]}）的 registration.json")
        if prev is not None:
            for earlier in SECTIONS[:SECTIONS.index(sec)]:
                problems.need(prev.get(earlier) == reg.get(earlier), f"{earlier} 段落和上一次登錄時不同（只能新增，不能修改）")
            problems.need(sec not in prev, f"{sec} 在上一次登錄時就已存在")
    if sec == "R1" and a.r1_models:
        r1m = load(a.r1_models)
        problems.need(r1m.get("models") == reg.get("R1", {}).get("models"), "r1_models.json 的模型表和 R1 段落不同")
    for earlier in SECTIONS[:SECTIONS.index(sec)]:             # the proofs of the earlier registrations must be real
        proof_problems(problems, getattr(a, f"ots_{earlier.lower()}") or root / "registration" / f"{earlier}.SHA256SUMS.ots")
    if sec == "R1":
        problems.need(a.stage0, "check R1 一定要帶 --stage0 stage0_record.json")
    if sec == "R2":
        problems.need(a.attempts, "check R2 一定要帶 --attempts attempts.json")
    if sec == "R1" and a.stage0:
        rec = load(a.stage0)
        problems.need(sha256_file(a.stage0) == reg.get("R1", {}).get("stage0_record_sha256"), "stage0_record.json 和 R1 登錄的雜湊不同")
        started_after(problems, "第零階段", [r["utc"] for r in rec.get("tokenizer_checks", {}).values()],
                      reg.get("R1", {}).get("previous", {}).get("release_created_at"))
    if sec == "R2" and a.attempts:
        att = load(a.attempts).get("stage1", {})
        started_after(problems, "第一階段", [x["start"] for v in att.values() for x in v], reg.get("R2", {}).get("previous", {}).get("release_created_at"))
    has_sums = check_sums(root, reg_path, sec, problems)
    for p in problems:
        print("不符：" + p)
    print(f"{sec}: {'通過' if not problems else f'{len(problems)} 項不符'}" + ("" if has_sums else f"（registration/{sec}.SHA256SUMS 尚未產生）"))
    return 0 if not problems else 1


# ------------------------------------------------------------------ make / sums / r1-models
def cmd_make(a):
    root = pathlib.Path(a.root).resolve(); reg_path = pathlib.Path(a.registration)
    reg = load(reg_path); sec = a.section; spec = load(a.spec)
    if sec in reg:
        sys.exit(f"{sec} 已經存在：登錄只能新增段落，不能修改")
    missing = [s for s in SECTIONS[:SECTIONS.index(sec)] if s not in reg]
    if missing:
        sys.exit(f"要先有 {missing} 段落")
    s = dict(utc=utc())
    if sec == "R0":
        s.update(protocol=spec["protocol"], decisions=spec["decisions"], seeds=spec["seeds"], packages=spec["packages"],
                 notes=spec.get("notes", ""))
        s["files"] = {p: sha256_file(root / p) for p in spec["files"]}
        s["protocol"] = dict(path=spec["protocol"], sha256=s["files"].get(spec["protocol"]))
        man = [p for p in s["files"] if p.endswith("m4gt_humans_PE_manifest.csv")]
        s["pe_manifest_sha256"] = s["files"][man[0]] if len(man) == 1 else None
    elif sec == "R1":
        rec_path = pathlib.Path(a.stage0)
        rec = load(rec_path)
        models = []
        for m in rec["r1_models_draft"]:
            m = dict(m); m.update(spec.get("model_overrides", {}).get(m["name"], {}))
            models.append({k: m[k] for k in MODEL_FIELDS})
        pm = {n: g["P_m"] for n, g in rec["gpu_checks"].items() if "P_m" in g}
        s.update(previous=spec["previous"], models=models, packages=rec["versions"], environment=rec["environment"],
                 P_m_min={m["name"]: pm[m["name"]] for m in models}, P_m_total_min=sum(pm[m["name"]] for m in models),
                 changes=rec["changes"], stage0_record_sha256=sha256_file(rec_path), notes=spec.get("notes", ""))
    else:
        s.update(previous=spec["previous"], predictions_sha256=sha256_file(a.predictions), predictions_sha256_colab=a.colab_sha,
                 not_written=[dict(model=n, reason=p.get("reason", "")) for n, p in load(a.predictions)["models"].items()
                              if p.get("status") != "written"], notes=spec.get("notes", ""))
        if s["predictions_sha256"] != a.colab_sha:
            sys.exit("雲端硬碟同步後的 predictions.json 雜湊和 Colab 印出的不同：不能登錄 R2")
    reg[sec] = s
    save(reg_path, reg)
    print(f"已新增 {sec} 段落到 {reg_path}")
    return 0


def cmd_sums(a):
    root = pathlib.Path(a.root).resolve(); reg_path = pathlib.Path(a.registration); reg = load(reg_path); sec = a.section
    files = list(reg[sec].get("files", {})) if sec == "R0" else list(a.extra or [])
    files.append(os.path.relpath(reg_path, root))
    out = root / "registration" / f"{sec}.SHA256SUMS"; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(f"{sha256_file(root / p)}  {p}\n" for p in files), encoding="utf-8")
    print("wrote", out)
    return 0


def run_tool(name, *args):
    """(return code, stdout) of a command-line tool found on PATH or next to this Python (the ots client lives in the
    virtual environment); (None, "") when the tool is not installed."""
    exe = shutil.which(name) or shutil.which(name, path=str(pathlib.Path(sys.executable).parent))
    if exe is None:
        return None, ""
    out = subprocess.run([exe, *args], capture_output=True, text=True)
    return out.returncode, out.stdout


def proof_problems(problems, ots_file, block_height=None):
    """The .ots file must be a real OpenTimestamps proof (ots info reads it and lists an attestation) of the file it stamps,
    that is the same path without .ots. With a block height (R2), it must already be upgraded to that Bitcoin block."""
    ots_file = pathlib.Path(ots_file); stamped = ots_file.with_suffix("")
    code, out = run_tool("ots", "info", str(ots_file))
    if code is None:
        problems.append("找不到 ots 指令"); return
    ok = code == 0 and ("PendingAttestation" in out or "BitcoinBlockHeaderAttestation" in out)
    problems.need(ok, f"{ots_file} 不是有效的時間證明（ots info 讀不出證明；空檔或亂碼都不算）")
    if not ok:
        return
    m = re.search(r"File sha256 hash: ([0-9a-f]{64})", out)
    problems.need(stamped.exists() and m is not None and m.group(1) == sha256_file(stamped), f"{ots_file} 證明的不是 {stamped.name}")
    if block_height is not None:
        problems.need(f"BitcoinBlockHeaderAttestation({block_height})" in out,
                      f"{ots_file} 還沒有升級到比特幣區塊 {block_height}（ots info 找不到這個區塊的證明）")


def release_and_proof(problems, tag, ots_file, skip_gh, block_height=None):
    """Before the next step: the GitHub Release exists and the proof of its SHA256SUMS is real; with a block height (R2),
    the proof is also upgraded to that Bitcoin block. A missing tool counts as a failed check."""
    if not skip_gh:
        code, out = run_tool("gh", "release", "view", tag, "--json", "tagName,createdAt")
        problems.need(code == 0 and json.loads(out or "{}").get("tagName") == tag, f"GitHub Release {tag} 不存在（或找不到 gh 指令）")
    proof_problems(problems, ots_file, block_height)


def cmd_r1_models(a):
    reg = load(a.registration); problems = Problems()
    if "R1" not in reg:
        sys.exit("還沒有 R1 段落")
    for sec, ots in (("R0", a.ots_r0), ("R1", a.ots_r1)):     # stamped is enough; the upgrade may come later (plan 5.3)
        release_and_proof(problems, TAGS[sec], ots, a.skip_gh)
    for p in problems:
        print("不符：" + p)
    if problems:
        return 1
    save(a.out, dict(r1_tag=TAGS["R1"], r1_commit=a.r1_commit, models=reg["R1"]["models"]))
    print("wrote", a.out)
    return 0


# ------------------------------------------------------------------ env-update (plan 5.4: Colab changed its preinstalled versions)
def cmd_env_update(a):
    root = pathlib.Path(a.root).resolve(); env_path = pathlib.Path(a.environment); problems = Problems()
    old, new = load(env_path), json.loads(a.new)
    problems.need(bool(old), f"{env_path} 不存在")
    problems.need(set(new) == set(old) and new != old, "新版本要列出和舊紀錄相同的欄位，而且要和舊紀錄不同")
    note = root / a.note
    problems.need(note.exists(), f"修訂紀錄 {a.note} 不存在（要放在倉庫裡）")
    commit = ""
    if note.exists():
        text = note.read_text(encoding="utf-8")
        problems.need("原因" in text, "修訂紀錄要寫出原因（含「原因」二字）")
        problems.need(re.search(r"\d{4}-\d{2}-\d{2}", text) is not None, "修訂紀錄要寫出時間（YYYY-MM-DD）")
        for k in sorted(set(old) | set(new)):
            if old.get(k) != new.get(k):
                problems.need(str(old.get(k)) in text and str(new.get(k)) in text, f"修訂紀錄要寫出 {k} 的舊版本 {old.get(k)} 與新版本 {new.get(k)}")
        if not a.skip_git:
            code, commit = run_tool("git", "-C", str(root), "log", "-1", "--format=%H", "--", str(note.relative_to(root)))
            commit = (commit or "").strip()
            problems.need(code == 0 and bool(commit), "修訂紀錄還沒 commit")
            if commit:
                code, out = run_tool("git", "-C", str(root), "branch", "-r", "--contains", commit)
                problems.need(code == 0 and "origin/" in (out or ""), "修訂紀錄的 commit 還沒推上 GitHub")
    for p in problems:
        print("不符：" + p)
    if problems:
        return 1
    hist_path = env_path.with_name("environment_history.json")
    history = json.load(open(hist_path, encoding="utf-8")) if hist_path.exists() else []
    history.append(dict(utc=utc(), old=old, new=new, note=a.note, note_sha256=sha256_file(note), note_commit=commit or None))
    save(hist_path, history); save(env_path, new)
    print("updated", env_path, "; history in", hist_path)
    return 0


# ------------------------------------------------------------------ go (implementation list item 11)
def cmd_go(a):
    reg = load(a.registration); problems = Problems()
    got = sha256_file(a.predictions)
    problems.need(got == a.colab_sha, f"雲端硬碟同步後的雜湊 {got} 和 Colab 印出的 {a.colab_sha} 不同")
    problems.need(got == reg.get("R2", {}).get("predictions_sha256"), "predictions.json 的雜湊和 R2 登錄值不同")
    release_and_proof(problems, TAGS["R2"], a.ots, a.skip_gh, block_height=None if a.skip_upgrade else a.block_height)
    try:                                                         # R2 only: Bitcoin's time must already lie in the past
        problems.need(when(a.block_time) < datetime.now(timezone.utc), f"區塊時間 {a.block_time} 不早於現在")
    except AssertionError:
        problems.append("--block-time 要是帶時區的 ISO 8601 時間")
    except ValueError:
        problems.append("--block-time 要是帶時區的 ISO 8601 時間")
    for p in problems:
        print("不符：" + p)
    if problems:
        return 1
    save(a.out, dict(predictions_sha256=got, r2_tag=TAGS["R2"], r2_commit=a.r2_commit, ots_block_height=int(a.block_height),
                     ots_block_time=a.block_time))
    print("wrote", a.out)
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    i = 0
    while i < len(argv) and argv[i].startswith("--"):          # skip global options and their values
        i += 2 if "=" not in argv[i] else 1
    if i < len(argv) and argv[i] in SECTIONS:                  # "check_registration.py R0" is short for "check R0"
        argv.insert(i, "check")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registration", default=str(ROOT / "registration" / "registration.json"))
    ap.add_argument("--root", default=str(ROOT))
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check"); c.add_argument("section", choices=SECTIONS); c.add_argument("--previous")
    c.add_argument("--r1-models"); c.add_argument("--stage0", help="R1: stage0_record.json (hash, and stage 0 began after R0's block)")
    c.add_argument("--attempts", help="R2: attempts.json (stage 1 began after R1's Release)")
    c.add_argument("--ots-r0"); c.add_argument("--ots-r1")
    m = sub.add_parser("make"); m.add_argument("section", choices=SECTIONS); m.add_argument("--spec", required=True)
    m.add_argument("--stage0"); m.add_argument("--predictions"); m.add_argument("--colab-sha")
    s = sub.add_parser("sums"); s.add_argument("section", choices=SECTIONS); s.add_argument("extra", nargs="*")
    r = sub.add_parser("r1-models")
    for opt in ("--out", "--r1-commit"):
        r.add_argument(opt, required=True)
    r.add_argument("--ots-r0", default=str(ROOT / "registration" / "R0.SHA256SUMS.ots"))
    r.add_argument("--ots-r1", default=str(ROOT / "registration" / "R1.SHA256SUMS.ots"))
    r.add_argument("--skip-gh", action="store_true", help="dry run only")
    g = sub.add_parser("go")
    for opt in ("--predictions", "--colab-sha", "--r2-commit", "--block-height", "--block-time", "--out"):
        g.add_argument(opt, required=True)
    g.add_argument("--ots", default=str(ROOT / "registration" / "R2.SHA256SUMS.ots"))
    g.add_argument("--skip-gh", action="store_true", help="dry run only")
    g.add_argument("--skip-upgrade", action="store_true", help="dry run only: a real but not yet upgraded proof")
    e = sub.add_parser("env-update")
    for opt in ("--environment", "--new", "--note"):
        e.add_argument(opt, required=True)
    e.add_argument("--skip-git", action="store_true", help="tests and dry runs only")
    a = ap.parse_args(argv)
    return {"check": cmd_check, "make": cmd_make, "sums": cmd_sums, "r1-models": cmd_r1_models, "go": cmd_go,
            "env-update": cmd_env_update}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
