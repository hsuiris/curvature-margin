# Freeze the M4GT-Bench Subtask B evaluation manifest (text hashes only).
#!/usr/bin/env python3
"""Freeze a bounded, text-free evaluation manifest from M4GT Subtask B."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


SOURCES = ("arxiv", "outfox", "peerread", "reddit", "wikihow", "wikipedia")


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_jsonl", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--per-group", type=int, default=200)
    args = parser.parse_args()

    raw_bytes = args.raw_jsonl.read_bytes()
    raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    rows: list[dict[str, object]] = []
    counts: Counter[tuple[int, str, str]] = Counter()

    for line_number, line in enumerate(raw_bytes.splitlines(), start=1):
        item = json.loads(line)
        source = item["source"]
        if source not in SOURCES or item.get(f"{source}_0") != "test":
            continue
        label = int(item["label"])
        model = item["model"]
        text = item["text"]
        text_hash = digest(text)
        counts[(label, source, model)] += 1
        rows.append({
            "line_number": line_number,
            "source": source,
            "label": label,
            "model": model,
            "text_chars": len(text),
            "text_sha256": text_hash,
        })

    # Stable selection, independent of file order. Label 1 is human; other labels are generators.
    chosen: list[dict[str, object]] = []
    grouped: dict[tuple[int, str, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((row["label"], row["source"], row["model"]), []).append(row)
    for key in sorted(grouped):
        chosen.extend(sorted(grouped[key], key=lambda row: row["text_sha256"])[:args.per_group])
    chosen.sort(key=lambda row: (row["source"], row["label"], row["model"], row["text_sha256"]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "m4gt_subtaskb_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=chosen[0].keys())
        writer.writeheader()
        writer.writerows(chosen)
    audit = {
        "raw_file": args.raw_jsonl.name,
        "raw_sha256": raw_sha256,
        "selection_rule": {
            "split_field": "{source}_0 == test",
            "sources": SOURCES,
            "per_label_source_model": args.per_group,
            "selection": "lowest SHA-256(text) within each label/source/model group",
            "label_interpretation": "1 = human; 0, 2-6 = generator classes",
        },
        "eligible_counts": [
            {"label": label, "source": source, "model": model, "n": n}
            for (label, source, model), n in sorted(counts.items())
        ],
        "manifest_rows": len(chosen),
    }
    (args.output_dir / "m4gt_subtaskb_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"raw_sha256={raw_sha256}")
    print(f"eligible_rows={len(rows)} manifest_rows={len(chosen)}")


if __name__ == "__main__":
    main()
