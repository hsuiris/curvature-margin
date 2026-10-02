"""Acceptance tests for scripts/check_nb15_pair.py (plan 8.8.24, implementation list item 6)."""
import json, pathlib, subprocess, sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_nb15_pair as pair   # noqa: E402

pytestmark = pytest.mark.skipif(not pair.ZH.exists(), reason="the Chinese notebook lives on Google Drive")


def run(zh_nb, tmp_path):
    f = tmp_path / "zh.ipynb"
    f.write_text(json.dumps(zh_nb, ensure_ascii=False), encoding="utf-8")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_nb15_pair.py"), str(pair.EN), str(f)], capture_output=True, text=True)
    return r.returncode, r.stdout


def mutate(old, new):
    nb = json.loads(pair.ZH.read_text(encoding="utf-8"))
    hits = 0
    for c in nb["cells"]:
        s = "".join(c["source"])
        if old in s:
            hits += 1; c["source"] = [s.replace(old, new, 1)]
    assert hits == 1
    return nb


def test_versions_agree():
    assert pair.main([]) == 0


@pytest.mark.parametrize("old,new,where", [("google/gemma-3-1b-pt", "google/gemma-3-1b-it", "第 5 格"),
                                           ("1.06, 1.09", "1.06, 1.08", "第 5 格"),
                                           ("top_k=0, top_p=1.0", "top_k=50, top_p=1.0", "第 5 格"),
                                           ('"/content/drive/MyDrive/AI-Text/03_實驗/data"', '"/content/drive/MyDrive/other"', "第 2 格")])
def test_code_changes_are_caught(tmp_path, old, new, where):
    code, out = run(mutate(old, new), tmp_path)
    assert code != 0 and where in out and "行" in out


def test_comment_change_is_allowed(tmp_path):
    code, _ = run(mutate("# 5. 事先登錄的設定", "# 5. 事先登錄的設定（註解改寫）"), tmp_path)
    assert code == 0


def test_unlisted_translation_is_caught(tmp_path):
    code, out = run(mutate('"在 Colab 上 DRYRUN 必須是 False。"', '"在 Colab 上 DRYRUN 一定要是 False。"'), tmp_path)
    assert code != 0 and "第 1 格" in out
