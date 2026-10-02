# Re-check the frozen P/E human-text manifest from the files on disk (protocol v5, implementation list item 1).
"""Checks m4gt_humans_PE_manifest.csv and m4gt_humans_PE_audit.json against the protocol without trusting the script that
froze them: steps 1-3 and the seeded draw are recomputed here, and the step-4 near-duplicate screen is re-implemented with
scikit-learn word 3-grams and a sparse product, as the protocol review computed it. The last test re-runs the script and
compares bytes.

Run: CMARGIN_DATA=<data folder> python scripts/prepare/test_m4gt_humans_PE.py   (pytest also works). The data folder
holds external/m4gt/SubtaskB.jsonl and manifests/; default is the repository's data/.
"""
import csv
import difflib
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np

DATA = Path(os.environ.get("CMARGIN_DATA", Path(__file__).resolve().parents[2] / "data"))
SCRIPT = Path(__file__).with_name("m4gt_humans_PE.py")
OLD = ("m4gt_subtaskb_manifest.csv", "m4gt_humans_A_manifest.csv")
MANIFEST, AUDIT = "m4gt_humans_PE_manifest.csv", "m4gt_humans_PE_audit.json"
FIELDS = ["id", "group", "role", "line_number", "source", "word_count", "text_sha256", "norm_sha256"]
ROLES = ("prompt_A", "prompt_B", "reference", "P")
# The protocol's numbers, restated on purpose rather than imported from the script under test.
RAW_SHA256 = "4d65fc1fb93b5c9d21b226aca06b610ee031fa9ccbc18fa6f39d16273dd281c7"
QUOTA = {"arxiv": 167, "outfox": 167, "peerread": 167, "reddit": 167, "wikihow": 166, "wikipedia": 166}
AFTER_1 = {"outfox": 673, "peerread": 866}
EXCLUDED_2 = {"peerread": 31, "outfox": 10, "reddit": 1}
AFTER_3 = {"arxiv": 2624, "peerread": 775, "reddit": 769, "outfox": 642, "wikipedia": 538, "wikihow": 406}
EXCLUDED_4 = {"arxiv": 5, "wikihow": 4, "outfox": 2, "reddit": 2, "peerread": 1}
AFTER_4 = {"arxiv": 2619, "peerread": 774, "reddit": 767, "outfox": 640, "wikipedia": 538, "wikihow": 402}
REVIEW_STEP4 = {4411, 4607, 4869, 4958, 5555, 17984, 19014, 19641, 19650, 56321, 56455, 74604, 105353, 107779}


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def norm(text):
    return re.sub(r"\s+", " ", text).strip().lower()


def read_csv(path):
    return list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"), newline="")))


@lru_cache(maxsize=None)
def load():
    raw = (DATA / "external/m4gt/SubtaskB.jsonl").read_bytes()
    old = {f: read_csv(DATA / "manifests" / f) for f in OLD}
    old_lines = {int(r["line_number"]) for rows in old.values() for r in rows}
    doc = {}                                  # line -> facts, for every human row and every old-manifest row
    for ln, line in enumerate(raw.splitlines(), start=1):
        it = json.loads(line)
        if it["model"] == "human" or ln in old_lines:
            doc[ln] = dict(source=it["source"], human=it["model"] == "human", words=len(it["text"].split()),
                           sha=sha(it["text"]), norm=norm(it["text"]))
    return SimpleNamespace(raw_sha=hashlib.sha256(raw).hexdigest(), old=old, old_lines=old_lines, doc=doc,
                           old_human=sorted(x for x in old_lines if doc[x]["human"]),
                           manifest=read_csv(DATA / "manifests" / MANIFEST),
                           audit=json.loads((DATA / "manifests" / AUDIT).read_text(encoding="utf-8")))


def per_source(lines):
    c = Counter(load().doc[x]["source"] for x in lines)
    return {s: c[s] for s in sorted(QUOTA)}


@lru_cache(maxsize=None)
def pools():
    """Candidate lines after steps 0 to 3, recomputed from the raw file and the old manifests (texts compared directly)."""
    d = load()
    old_sha = {d.doc[x]["sha"] for x in d.old_lines}
    old_norm = {d.doc[x]["norm"] for x in d.old_lines}
    p0 = [x for x, r in d.doc.items() if r["human"] and r["source"] in QUOTA and 120 <= r["words"] <= 300]
    p1 = [x for x in p0 if x not in d.old_lines and d.doc[x]["sha"] not in old_sha]
    p2 = [x for x in p1 if d.doc[x]["norm"] not in old_norm]
    seen, p3 = set(), []
    for x in sorted(p2):
        if d.doc[x]["norm"] not in seen:
            seen.add(d.doc[x]["norm"])
            p3.append(x)
    return p0, p1, p2, p3


def step4(cands, olds):
    """Screened and confirmed pairs (sorted line tuples) of protocol step 4, the way the review computed the screen."""
    from sklearn.feature_extraction.text import CountVectorizer
    lines = list(cands) + list(olds)
    texts = [load().doc[x]["norm"] for x in lines]
    X = CountVectorizer(analyzer="word", ngram_range=(3, 3), token_pattern=r"(?u)\b\w+\b", lowercase=False,
                        binary=True).fit_transform(texts)
    size = np.diff(X.indptr).astype(np.int64)            # distinct 3-grams per text
    C = (X[:len(cands)] @ X.T).tocoo()                   # rows: candidates; columns: everything, so old-old never appears
    c = C.data.astype(np.int64)
    keep = (C.row < C.col) & (100 * c * c >= 9 * size[C.row] * size[C.col])
    found = set(zip(C.row[keep].tolist(), C.col[keep].tolist()))
    for cut in (slice(None, 200), slice(-200, None)):
        groups = {}
        for k, t in enumerate(texts):
            groups.setdefault(t[cut], []).append(k)
        found |= {(i, j) for g in groups.values() for i in g for j in g if i < j and i < len(cands)}
    screened = {tuple(sorted((lines[i], lines[j]))) for i, j in found}
    dup = {(a, b) for a, b in screened if difflib.SequenceMatcher(
        None, load().doc[a]["norm"], load().doc[b]["norm"], autojunk=False).ratio() >= 0.7}
    return screened, dup


def dropped(cands, dup):
    """Step-4 exclusions: clusters of candidate duplicates keep their smallest line; candidates duplicating old rows go."""
    cand = set(cands)
    label = {x: x for pair in dup for x in pair if x in cand}
    changed = True
    while changed:                                        # propagate the smallest line through each cluster
        changed = False
        for a, b in dup:
            if a in cand and b in cand and label[a] != label[b]:
                label[a] = label[b] = min(label[a], label[b])
                changed = True
    return {x for x, lab in label.items() if lab != x} | {a if a in cand else b for a, b in dup if (a in cand) != (b in cand)}


def test_files_match_audit():
    d = load()
    assert d.raw_sha == RAW_SHA256 == d.audit["raw_sha256"]
    assert d.audit["seed"] == 20260930 and d.audit["quota_per_group"] == QUOTA
    for f in OLD:
        assert d.audit["old_manifests"][f]["sha256"] == hashlib.sha256((DATA / "manifests" / f).read_bytes()).hexdigest()
    assert d.audit["manifest_sha256"] == hashlib.sha256((DATA / "manifests" / MANIFEST).read_bytes()).hexdigest()
    assert d.audit["manifest_rows"] == len(d.manifest) == 2000


def test_columns_order_and_quotas():
    m = load().manifest
    assert list(m[0].keys()) == FIELDS
    assert [r["id"] for r in m] == [str(i) for i in range(len(m))]
    assert m == sorted(m, key=lambda r: (ROLES.index(r["role"]), r["source"], r["text_sha256"]))
    assert all(r["group"] == ("P" if r["role"] == "P" else "E") for r in m)
    for g in ("P", "E"):
        assert Counter(r["source"] for r in m if r["group"] == g) == Counter(QUOTA)
    for role in ("prompt_A", "prompt_B"):
        assert Counter(r["source"] for r in m if r["role"] == role) == Counter(dict.fromkeys(QUOTA, 25))
    assert Counter(r["role"] for r in m) == Counter(prompt_A=150, prompt_B=150, reference=700, P=1000)


def test_rows_match_raw_text():
    d = load()
    for r in d.manifest:
        x = d.doc[int(r["line_number"])]
        assert x["human"] and x["source"] == r["source"] and str(x["words"]) == r["word_count"] and 120 <= x["words"] <= 300
        assert x["sha"] == r["text_sha256"] and sha(x["norm"]) == r["norm_sha256"]


def test_unique_within_groups_and_disjoint_from_everything_used():
    d = load()
    for key in ("line_number", "text_sha256", "norm_sha256"):
        group = {g: [r[key] for r in d.manifest if r["group"] == g] for g in ("P", "E")}
        assert all(len(set(v)) == len(v) for v in group.values()), key
        assert not set(group["P"]) & set(group["E"]), key
    lines = {int(r["line_number"]) for r in d.manifest}
    assert not lines & d.old_lines
    assert not {d.doc[x]["sha"] for x in lines} & {d.doc[x]["sha"] for x in d.old_lines}
    assert not {d.doc[x]["norm"] for x in lines} & {d.doc[x]["norm"] for x in d.old_lines}


def test_audit_steps_match_recomputation_and_protocol():
    d = load()
    p0, p1, p2, p3 = pools()
    st = {s["step"]: s for s in d.audit["steps"]}
    out = {k: {x for v in s["excluded"].values() for x in v["lines"]} for k, s in st.items() if "excluded" in s}
    assert all(v["n"] == len(v["lines"]) for s in st.values() for v in s.get("excluded", {}).values())
    assert st["0"]["remaining"] == per_source(p0)
    assert out["1a"] | out["1b"] == set(p0) - set(p1) and st["1b"]["remaining"] == per_source(p1)
    assert all(per_source(p1)[s] == n for s, n in AFTER_1.items())
    assert out["2"] == set(p1) - set(p2) and {s: n for s, n in per_source(out["2"]).items() if n} == EXCLUDED_2
    assert out["3"] == set(p2) - set(p3) and st["3"]["remaining"] == per_source(p3) == AFTER_3
    assert out["4"] == REVIEW_STEP4 and {s: n for s, n in per_source(out["4"]).items() if n} == EXCLUDED_4
    assert st["4"]["remaining"] == AFTER_4
    assert [17062, 19641, 19650] in st["4"]["clusters"]
    assert any(p["line_a"] == 4868 and p["line_b"] == 4869 and p["duplicate"] for p in st["4"]["pairs"])
    assert 3943 in p3 and 3943 not in out["4"]


def test_step4_reimplemented_gives_the_review_exclusions():
    d = load()
    p3 = pools()[3]
    screened, dup = step4(p3, d.old_human)
    pairs = d.audit["steps"][-1]["pairs"]
    assert screened == {(p["line_a"], p["line_b"]) for p in pairs}
    assert dup == {(p["line_a"], p["line_b"]) for p in pairs if p["duplicate"]}
    assert dropped(p3, dup) == REVIEW_STEP4


def test_no_near_duplicate_left_in_P_and_E():
    d = load()
    screened, dup = step4([int(r["line_number"]) for r in d.manifest], d.old_human)
    assert not dup, sorted(dup)
    assert screened == {(p["line_a"], p["line_b"]) for p in d.audit["recheck_P_union_E"]["pairs"]}


def test_seeded_draw_reproduces_the_roles():
    d = load()
    pool = [x for x in pools()[3] if x not in REVIEW_STEP4]
    rng = np.random.default_rng(20260930)
    role = {}
    for s in sorted(QUOTA):
        c = sorted((x for x in pool if d.doc[x]["source"] == s), key=lambda x: sha(d.doc[x]["norm"]))
        drawn = [c[i] for i in rng.choice(len(c), 2 * QUOTA[s], replace=False)]
        e = drawn[:QUOTA[s]]
        prompts = sorted(e[:50], key=lambda x: d.doc[x]["sha"])
        role.update(dict.fromkeys(prompts[:25], "prompt_A") | dict.fromkeys(prompts[25:], "prompt_B")
                    | dict.fromkeys(e[50:], "reference") | dict.fromkeys(drawn[QUOTA[s]:], "P"))
    assert {int(r["line_number"]): r["role"] for r in d.manifest} == role


def test_rerun_is_byte_identical():
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, str(SCRIPT), str(DATA / "external/m4gt/SubtaskB.jsonl"), str(DATA / "manifests"),
                        "--out", tmp], check=True, capture_output=True)
        for f in (MANIFEST, AUDIT):
            assert (Path(tmp) / f).read_bytes() == (DATA / "manifests" / f).read_bytes(), f


if __name__ == "__main__":
    for name, fn in [(n, f) for n, f in globals().items() if n.startswith("test_")]:
        fn()
        print("ok", name)
