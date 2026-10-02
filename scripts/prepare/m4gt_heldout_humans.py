# Freeze the 1,000 held-out M4GT human texts used as prompts/reference for the pythia test.
#!/usr/bin/env python3
"""Held-out human sample for external validation A: 1,000 human texts from M4GT Subtask B that the Subtask B manifest does not use
(120 to 300 English words, stratified over six sources). The first 300 (50 per source) are continuation prompts; the other 700 are
human references only. Stores line numbers and hashes, no text. Seed 20260926."""
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np

SEED = 20260926
N_PER = {"arxiv": 167, "outfox": 167, "peerread": 167, "reddit": 167, "wikihow": 166, "wikipedia": 166}   # 1,000
N_SOURCE_PER = 50                                                                                        # 300 continuation prompts
raw, out = Path(sys.argv[1]), Path(sys.argv[2])
used = {int(r["line_number"]) for r in csv.DictReader(open(out / "m4gt_subtaskb_manifest.csv", encoding="utf-8"))}
raw_bytes = raw.read_bytes(); raw_sha = hashlib.sha256(raw_bytes).hexdigest()
cands = {s: [] for s in N_PER}
for ln, line in enumerate(raw_bytes.splitlines(), start=1):
    it = json.loads(line)
    if it.get("model") != "human" or it["source"] not in N_PER or ln in used:
        continue
    n = len(it["text"].split())
    if 120 <= n <= 300:
        cands[it["source"]].append(dict(line_number=ln, source=it["source"], word_count=n,
                                        text_sha256=hashlib.sha256(it["text"].encode("utf-8")).hexdigest()))
rng = np.random.default_rng(SEED); src, ref = [], []
for s in sorted(N_PER):                                    # sources in alphabetical order; candidates sorted by hash, independent of file order
    c = sorted(cands[s], key=lambda r: r["text_sha256"])
    for j, i in enumerate(rng.choice(len(c), N_PER[s], replace=False)):
        (src if j < N_SOURCE_PER else ref).append(c[i])
rng.shuffle(src); rng.shuffle(ref)
rows = [dict(id=i, role="source" if i < len(src) else "reference", **r) for i, r in enumerate(src + ref)]
with (out / "m4gt_humans_A_manifest.csv").open("w", newline="", encoding="utf-8") as h:
    w = csv.DictWriter(h, fieldnames=["id", "role", "line_number", "source", "word_count", "text_sha256"]); w.writeheader(); w.writerows(rows)
audit = dict(raw_file=raw.name, raw_sha256=raw_sha, seed=SEED, rule="human rows not in m4gt_subtaskb_manifest.csv; 120-300 whitespace words; per-source counts as N_PER; "
             "candidates sorted by text sha256 then rng.choice without replacement; first 50 per source are continuation sources; ids 0-299 = source, 300-999 = reference",
             eligible_per_source={s: len(c) for s, c in cands.items()}, rows=len(rows),
             manifest_sha256=hashlib.sha256((out / "m4gt_humans_A_manifest.csv").read_bytes()).hexdigest())
(out / "m4gt_humans_A_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(audit, ensure_ascii=False, indent=2))
