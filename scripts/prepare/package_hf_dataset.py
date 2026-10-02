"""Collect the files to publish on Hugging Face into one folder with the layout of data/README.md.

Only generated text and scores are copied. Human texts from M4GT-Bench and RAID are excluded; generated files built on
M4GT prompts have their human fields ("prefix", "human") removed. Usage:
  python scripts/prepare/package_hf_dataset.py <research_dir> <drive_data_dir> <out_dir>
  huggingface-cli upload <hf-user>/curvature-margin <out_dir> . --repo-type dataset
"""
import json, shutil, sys
from pathlib import Path

X, GD, OUT = (Path(a) for a in sys.argv[1:4])  # X: original research workspace; GD: Drive copy of data/
COPY = {  # destination folder -> list of (source folder, glob)
    "01_pilot_decoding": [(GD / "01_pilot_decoding", "*.json"), (X, "pilot3_scores.pkl")],
    "02_pilot_scorer_family": [(GD / "02_pilot_scorer_family", "scores_*.pkl")],
    "03_pilot_temperature": [(X / "pilot5", "*.json"), (X / "pilot5", "*.pkl")],
    "04_pilot_new_generators": [(X / "pilot10", "gen_*.json"), (X / "pilot10", "scores_*.pkl")],
    "05_pilot_paraphrase": [(X / "pilot11", "para_*.json"), (X / "pilot11", "scores_*.pkl")],
    "06_scaled_sweep": [(X / "main1", "*.json"), (X / "main1", "*.pkl")],
    "07_falcon_dual_scoring": [(X / "main2a", "scores_falcon_*.pkl")],
    "08_m4gt_natural_text": [(X / "m4gt", "m4gt_falcon_scores.csv"), (X / "m4gt", "m4gt_falcon_summary.csv"), (X / "m4gt", "m4gt_falcon_run_info.json")],
    "09_heldout_pythia": [(X / "m4gt_pythia", "scores_falcon_*.pkl"), (GD / "09_heldout_pythia", "gen_*.json")],
    "10_official_equivalence": [(GD / "10_official_equivalence", "*")],
    "11_trained_baselines": [(X / "trained", "scores_trained.csv")] + [(GD / "11_trained_baselines", f) for f in (
        "scores_offtheshelf.csv", "scores_diveye.csv", "scores_ada_v2.csv", "diveye_features.csv", "ada_beta_v2.json", "run_info.json", "data_info.json")],
    "mage_falcon_scores": [(X / "scores_falcon-7b-instruct", "*.pkl")],
    "raid_scores": [(X / "raid", "scores_*.pkl")],
    "manifests": [(GD / "manifests", "m4gt_*_manifest.csv"), (GD / "manifests", "m4gt_*_audit.json")],
}
for dest, items in COPY.items():
    (OUT / dest).mkdir(parents=True, exist_ok=True)
    for root, pat in items:
        for f in sorted(root.glob(pat)):
            if f.name.startswith("._"):
                continue
            target = OUT / dest / ("scores.pkl" if f.name == "pilot3_scores.pkl" else f.name)
            if dest == "09_heldout_pythia" and f.suffix == ".json":  # drop M4GT human text
                rows = [{k: v for k, v in r.items() if k not in ("prefix", "human")} for r in json.load(open(f))]
                json.dump(rows, open(target, "w"), ensure_ascii=False)
            else:
                shutil.copy(f, target)
            print(dest, target.name)
shutil.copy(Path(__file__).with_name("hf_dataset_card.md"), OUT / "README.md")
