# Freeze the P and E human texts of the second replication test (protocol v5, section 人類文章; implementation list item 1).
"""Held-out human texts for the 8.8.24 replication test: 2,000 M4GT Subtask B human texts in two disjoint groups.
P (1,000) feeds the predictions and the human reference of the direct estimate. E (1,000) holds the 300 continuation prompts
(halves A and B of 150) and is the human reference of the primary endpoint. No text overlaps the earlier manifests exactly,
after normalization, or as a near duplicate. Stores line numbers and hashes, no text. Every number the protocol states is
asserted before anything is written.

Usage: python scripts/prepare/m4gt_humans_PE.py data/external/m4gt/SubtaskB.jsonl data/manifests [--out DIR]
"""
import argparse
import csv
import difflib
import hashlib
import io
import json
import math
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

RAW_SHA256 = "4d65fc1fb93b5c9d21b226aca06b610ee031fa9ccbc18fa6f39d16273dd281c7"
OLD = ("m4gt_subtaskb_manifest.csv", "m4gt_humans_A_manifest.csv")
MANIFEST, AUDIT = "m4gt_humans_PE_manifest.csv", "m4gt_humans_PE_audit.json"
SEED = 20260930
QUOTA = {"arxiv": 167, "outfox": 167, "peerread": 167, "reddit": 167, "wikihow": 166, "wikipedia": 166}  # per group, as A's N_PER
SOURCES = tuple(sorted(QUOTA))
N_PROMPT, N_HALF = 50, 25            # per source: the first 50 drawn into E are prompts, split 25 / 25 into halves A and B
WORDS = (120, 300)                   # whitespace words, as m4gt_heldout_humans.py
KEY = 200                            # step 4 screen (i): same first or last 200 normalized characters
RATIO = 0.7                          # step 4 confirmation threshold
ROLES = ("prompt_A", "prompt_B", "reference", "P")
FIELDS = ["id", "group", "role", "line_number", "source", "word_count", "text_sha256", "norm_sha256"]

# Numbers the protocol states (implementation list item 1). Step 1 is 1a + 1b.
EXPECT_AFTER_1 = {"outfox": 673, "peerread": 866}
EXPECT_EXCLUDED_2 = {"peerread": 31, "outfox": 10, "reddit": 1}
EXPECT_AFTER_3 = {"arxiv": 2624, "peerread": 775, "reddit": 769, "outfox": 642, "wikipedia": 538, "wikihow": 406}
EXPECT_EXCLUDED_4 = {"arxiv": 5, "wikihow": 4, "outfox": 2, "reddit": 2, "peerread": 1}
EXPECT_AFTER_4 = {"arxiv": 2619, "peerread": 774, "reddit": 767, "outfox": 640, "wikipedia": 538, "wikihow": 402}
# The 14 lines the protocol review excluded at step 4 (its re-computation behind the counts above).
REVIEW_STEP4 = {4411, 4607, 4869, 4958, 5555, 17984, 19014, 19641, 19650, 56321, 56455, 74604, 105353, 107779}

RULES = {
    "0": f"model == 'human', source in {list(SOURCES)}, {WORDS[0]}-{WORDS[1]} whitespace words (str.split)",
    "1a": "drop rows whose line number is listed in an old manifest",
    "1b": "drop rows whose text SHA-256 equals that of any old-manifest row (human or machine)",
    "2": "drop rows whose normalized text equals that of any old-manifest row (human or machine); "
         "normalized = re.sub(r'\\s+', ' ', text).strip().lower()",
    "3": "identical normalized texts among the remaining rows keep the smallest line number",
    "4": "screen candidate-candidate and candidate-old-human pairs: same first or last 200 normalized characters, or "
         "cosine >= 0.3 between word-3-gram presence vectors (words = re.findall(r'\\w+', normalized text)); confirm with "
         "difflib.SequenceMatcher(None, a, b, autojunk=False).ratio() >= 0.7 on the normalized texts, a = the smaller "
         "line number; duplicate candidate pairs join into clusters that keep the smallest line number; a candidate "
         "duplicating an old human row is dropped",
}
SAMPLING = ("rng = numpy.random.default_rng(20260930); sources in alphabetical order; each source's candidates sorted by "
            "norm_sha256; rng.choice(len(c), 2 * quota, replace=False); the first quota drawn form group E, the rest group P; "
            "the first 50 drawn into E are prompts, sorted by text_sha256: first 25 prompt_A, last 25 prompt_B; the other "
            "E rows are reference")
ORDER = "rows sorted by role (prompt_A, prompt_B, reference, P), then source, then text_sha256; id = row index"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def near_dup_pairs(cands: list[dict], olds: list[dict]) -> list[dict]:
    """Protocol step 4 on candidates and old human rows: every screened pair with both similarities and the verdict.
    Old-old pairs are out of scope. Candidates come first, so a pair (i, j) with i < j is in scope exactly when i < len(cands)."""
    docs, nc = cands + olds, len(cands)
    found = defaultdict(set)
    for tag, cut in (("prefix200", lambda t: t[:KEY]), ("suffix200", lambda t: t[-KEY:])):
        groups = defaultdict(list)
        for k, d in enumerate(docs):
            groups[cut(d["norm"])].append(k)
        for g in groups.values():
            for i, j in combinations(g, 2):
                if i < nc:
                    found[(i, j)].add(tag)
    grams = [set(zip(w, w[1:], w[2:])) for w in (re.findall(r"\w+", d["norm"]) for d in docs)]
    size = np.array([len(g) for g in grams], dtype=np.int64)
    postings = defaultdict(list)
    for k, g in enumerate(grams):
        for x in g:
            postings[x].append(k)
    postings = {x: np.array(v, dtype=np.int64) for x, v in postings.items()}
    for i in range(nc):
        if not grams[i]:
            continue
        inter = np.bincount(np.concatenate([postings[x] for x in grams[i]]), minlength=len(docs))[i + 1:]
        # cosine = inter / sqrt(size_i * size_j) >= 0.3, tested in exact integers so no pair sits on a rounding edge
        hit = (inter > 0) & (100 * inter * inter >= 9 * size[i] * size[i + 1:])
        for j in np.flatnonzero(hit) + i + 1:
            found[(i, int(j))].add("trigram")
    pairs = []
    for (i, j), tags in found.items():
        a, b = sorted((docs[i], docs[j]), key=lambda d: d["line"])
        cos = len(grams[i] & grams[j]) / math.sqrt(size[i] * size[j]) if size[i] and size[j] else 0.0
        # autojunk off: by default difflib skips frequent characters and the ratio depends on argument order
        ratio = difflib.SequenceMatcher(None, a["norm"], b["norm"], autojunk=False).ratio()
        pairs.append(dict(line_a=a["line"], line_b=b["line"], source_a=a["source"], source_b=b["source"],
                          set_a=a["old"] or "candidate", set_b=b["old"] or "candidate", screen=sorted(tags),
                          trigram_cosine=round(cos, 6), difflib_ratio=round(ratio, 6), duplicate=ratio >= RATIO))
    return sorted(pairs, key=lambda p: (p["line_a"], p["line_b"]))


def per_source(lines) -> dict[str, int]:
    c = Counter(lines)
    return {s: c[s] for s in SOURCES}


def main() -> None:
    ap = argparse.ArgumentParser(description="Freeze the P and E human-text manifests (protocol v5).")
    ap.add_argument("raw_jsonl", type=Path)
    ap.add_argument("manifest_dir", type=Path, help="folder with the two earlier manifests")
    ap.add_argument("--out", type=Path, help="output folder (default: manifest_dir)")
    args = ap.parse_args()
    out = args.out or args.manifest_dir

    raw = args.raw_jsonl.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == RAW_SHA256, "SubtaskB.jsonl is not the registered file"
    old_file, old_info, listed_sha = {}, {}, {}      # line -> manifest name; manifest -> facts; line -> text_sha256 listed
    for f in OLD:
        data = (args.manifest_dir / f).read_bytes()
        rows = list(csv.DictReader(io.StringIO(data.decode("utf-8"), newline="")))
        lines = {int(r["line_number"]) for r in rows}
        assert len(lines) == len(rows) and not lines & old_file.keys(), f"{f}: line numbers repeat"
        old_file.update(dict.fromkeys(lines, f))
        listed_sha.update({int(r["line_number"]): r["text_sha256"] for r in rows})
        old_info[f] = dict(sha256=hashlib.sha256(data).hexdigest(), rows=len(rows))

    docs = {}                                        # line -> text facts, for every human row and every old-manifest row
    for ln, line in enumerate(raw.splitlines(), start=1):
        it = json.loads(line)
        human = it.get("model") == "human"
        if human or ln in old_file:
            t = it["text"]
            n = norm(t)
            docs[ln] = dict(line=ln, source=it["source"], human=human, words=len(t.split()), sha=sha(t), norm=n, nsha=sha(n),
                            old=old_file.get(ln))
    # The hashes the old manifests list prove that this line numbering is the one they used.
    assert all(docs[ln]["sha"] == h for ln, h in listed_sha.items()), "old manifest line numbers do not match the raw file"
    for f, info in old_info.items():
        info["human_rows"] = sum(docs[ln]["human"] for ln, g in old_file.items() if g == f)

    pool = {ln: d for ln, d in docs.items() if d["human"] and d["source"] in QUOTA and WORDS[0] <= d["words"] <= WORDS[1]}
    steps = [dict(step="0", rule=RULES["0"], remaining=per_source(d["source"] for d in pool.values()))]

    def drop(step: str, lines: set[int], **extra) -> None:
        nonlocal pool
        assert lines <= pool.keys()
        excluded = {}
        for s in SOURCES:
            v = sorted(x for x in lines if pool[x]["source"] == s)
            excluded[s] = dict(n=len(v), lines=v)
        pool = {k: d for k, d in sorted(pool.items()) if k not in lines}
        steps.append(dict(step=step, rule=RULES[step], excluded=excluded,
                          remaining=per_source(d["source"] for d in pool.values()), **extra))

    def matched(hit: dict[int, int]) -> list[list[int]]:
        return [[a, b] for a, b in sorted(hit.items())]

    drop("1a", {ln for ln in pool if ln in old_file})
    first_sha, first_nsha = {}, {}                   # hash -> smallest old line with it
    for ln in sorted(old_file):
        first_sha.setdefault(docs[ln]["sha"], ln)
        first_nsha.setdefault(docs[ln]["nsha"], ln)
    hit = {ln: first_sha[d["sha"]] for ln, d in pool.items() if d["sha"] in first_sha}
    drop("1b", set(hit), matched_old_line=matched(hit))
    hit = {ln: first_nsha[d["nsha"]] for ln, d in pool.items() if d["nsha"] in first_nsha}
    drop("2", set(hit), matched_old_line=matched(hit))
    kept, hit = {}, {}
    for ln, d in sorted(pool.items()):
        if d["nsha"] in kept:
            hit[ln] = kept[d["nsha"]]
        else:
            kept[d["nsha"]] = ln
    drop("3", set(hit), kept_line=matched(hit))

    old_human = [docs[ln] for ln in sorted(old_file) if docs[ln]["human"]]
    pairs = near_dup_pairs([pool[k] for k in sorted(pool)], old_human)
    dup = [p for p in pairs if p["duplicate"]]
    clusters, drop4 = [], set()
    for p in dup:
        if p["set_a"] == p["set_b"] == "candidate":
            touching = [c for c in clusters if p["line_a"] in c or p["line_b"] in c]
            clusters = [c for c in clusters if c not in touching] + [{p["line_a"], p["line_b"]}.union(*touching)]
        else:                                        # a candidate against an old human row: the candidate goes
            drop4.add(p["line_a"] if p["set_a"] == "candidate" else p["line_b"])
    clusters = sorted(sorted(c) for c in clusters)
    drop4 |= {x for c in clusters for x in c[1:]}
    drop("4", drop4, clusters=clusters, pairs=pairs)

    rng = np.random.default_rng(SEED)
    rows = []
    for s in SOURCES:
        c = sorted((d for d in pool.values() if d["source"] == s), key=lambda d: d["nsha"])
        drawn = [c[i] for i in rng.choice(len(c), 2 * QUOTA[s], replace=False)]
        e, p = drawn[:QUOTA[s]], drawn[QUOTA[s]:]
        prompts = sorted(e[:N_PROMPT], key=lambda d: d["sha"])
        for role, ds in (("prompt_A", prompts[:N_HALF]), ("prompt_B", prompts[N_HALF:]), ("reference", e[N_PROMPT:]), ("P", p)):
            rows += [dict(group="P" if role == "P" else "E", role=role, line_number=d["line"], source=s, word_count=d["words"],
                          text_sha256=d["sha"], norm_sha256=d["nsha"]) for d in ds]
    rows.sort(key=lambda r: (ROLES.index(r["role"]), r["source"], r["text_sha256"]))
    rows = [dict(id=i, **r) for i, r in enumerate(rows)]
    recheck = near_dup_pairs([docs[r["line_number"]] for r in rows], old_human)

    # Acceptance (implementation list item 1); nothing is written unless all of it holds.
    after = {st["step"]: st["remaining"] for st in steps}
    excl = {st["step"]: {s: v["n"] for s, v in st["excluded"].items() if v["n"]} for st in steps if "excluded" in st}
    assert all(after["1b"][s] == n for s, n in EXPECT_AFTER_1.items()), after["1b"]
    assert excl["2"] == EXPECT_EXCLUDED_2, excl["2"]
    assert after["3"] == EXPECT_AFTER_3, after["3"]
    assert excl["4"] == EXPECT_EXCLUDED_4, excl["4"]
    assert after["4"] == EXPECT_AFTER_4, after["4"]
    assert drop4 == REVIEW_STEP4, sorted(drop4 ^ REVIEW_STEP4)
    assert [17062, 19641, 19650] in clusters
    assert any(p["line_a"] == 4868 and p["line_b"] == 4869 and p["duplicate"] for p in pairs)
    assert 3943 in pool
    for g in ("P", "E"):
        assert per_source(r["source"] for r in rows if r["group"] == g) == QUOTA
    for role in ("prompt_A", "prompt_B"):
        assert per_source(r["source"] for r in rows if r["role"] == role) == {s: N_HALF for s in SOURCES}
    for key in ("line_number", "text_sha256", "norm_sha256"):
        group = {g: [r[key] for r in rows if r["group"] == g] for g in ("P", "E")}
        assert all(len(set(v)) == len(v) for v in group.values()), f"{key} repeats within a group"
        assert not set(group["P"]) & set(group["E"]), f"P and E share a {key}"
    assert not {r["line_number"] for r in rows} & old_file.keys()
    assert not {r["text_sha256"] for r in rows} & first_sha.keys()
    assert not {r["norm_sha256"] for r in rows} & first_nsha.keys()
    assert not any(p["duplicate"] for p in recheck), [p for p in recheck if p["duplicate"]]

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    manifest = buf.getvalue().encode("utf-8")
    assert list(csv.DictReader(io.StringIO(manifest.decode("utf-8"), newline=""))) == [
        {k: str(v) for k, v in r.items()} for r in rows]
    audit = dict(
        protocol="docs/prereg/8824_protocol.md (plan v5): section 人類文章, implementation list item 1",
        raw_file=args.raw_jsonl.name, raw_sha256=RAW_SHA256, old_manifests=old_info, seed=SEED, quota_per_group=QUOTA,
        steps=steps, sampling=SAMPLING, manifest_order=ORDER,
        selected={role: per_source(r["source"] for r in rows if r["role"] == role) for role in ROLES},
        recheck_P_union_E=dict(rule="step 4 screen and confirmation on P and E against each other and the old human rows",
                               screened_pairs=len(recheck), duplicates=0, pairs=recheck),
        manifest_file=MANIFEST, manifest_rows=len(rows), manifest_sha256=hashlib.sha256(manifest).hexdigest(),
    )
    out.mkdir(parents=True, exist_ok=True)
    for name, data in ((MANIFEST, manifest), (AUDIT, (json.dumps(audit, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))):
        tmp = out / (name + ".tmp")                  # write then rename: a failed run never leaves a half-written file
        tmp.write_bytes(data)
        tmp.replace(out / name)
        assert (out / name).read_bytes() == data

    for st in steps:
        print(f"step {st['step']:>2}: remaining {st['remaining']}")
    print("selected", audit["selected"])
    print(f"step 4 pairs screened {len(pairs)}, duplicates {len(dup)}; P+E recheck screened {len(recheck)}, duplicates 0")
    print(f"{MANIFEST} sha256={audit['manifest_sha256']}")


if __name__ == "__main__":
    main()
