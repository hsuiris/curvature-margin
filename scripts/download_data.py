"""Download the released continuations and per-token scores into data/ (or $CMARGIN_DATA)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from cmargin.paths import DATA
from huggingface_hub import snapshot_download

REPO_ID = "cookieiweq/curvature-margin"  # https://huggingface.co/datasets/cookieiweq/curvature-margin

if __name__ == "__main__":
    path = snapshot_download(repo_id=REPO_ID, repo_type="dataset", local_dir=str(DATA))
    print("data in", path)
