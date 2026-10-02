"""Compare the English (repo) and Chinese (Drive) versions of notebook 15 (plan 8.8.24, implementation list item 6).

Code cells are compared token by token after IPython magics are turned into Python and comments are removed. A string
literal (f-strings included) may differ only if the pair is listed in scripts/nb15_allowed_diffs.json; the data folder
path may differ only in cell 2. Anything else (a model name, a temperature, a field name, a decoding setting) is a
mismatch. Exit code 0 = the two versions agree; otherwise every mismatch is printed with its cell and line.
Usage: python scripts/check_nb15_pair.py [EN.ipynb ZH.ipynb] [--allowed FILE]
"""
import argparse, hashlib, io, json, pathlib, sys, tokenize
from IPython.core.inputtransformer2 import TransformerManager

ROOT = pathlib.Path(__file__).resolve().parents[1]
EN = ROOT / "notebooks" / "15_replication_test.ipynb"
ZH = pathlib.Path("/Users/xuyunqin/Library/CloudStorage/GoogleDrive-emilyhuang12380@gmail.com/我的雲端硬碟/AI-Text/03_實驗/notebooks/15_replication_test.ipynb")
ALLOWED = ROOT / "scripts" / "nb15_allowed_diffs.json"
SKIP = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING, tokenize.ENDMARKER}
FSTART, FEND = getattr(tokenize, "FSTRING_START", None), getattr(tokenize, "FSTRING_END", None)


def units(src):
    """(text, line) units of a code cell: one per token, except that a whole f-string (Python 3.12 splits it into several
    tokens) is one unit with its exact source text."""
    code = TransformerManager().transform_cell(src)
    lines = code.splitlines(keepends=True)
    out, depth, start = [], 0, None
    for tok in tokenize.generate_tokens(io.StringIO(code).readline):
        if tok.type == FSTART:
            depth += 1
            if depth == 1:
                start = tok.start
            continue
        if depth:
            if tok.type == FEND:
                depth -= 1
                if depth == 0:
                    (r0, c0), (r1, c1) = start, tok.end
                    text = lines[r0 - 1][c0:] + "".join(lines[r0:r1 - 1]) + lines[r1 - 1][:c1] if r1 > r0 else lines[r0 - 1][c0:c1]
                    out.append((text, r0))
            continue
        if tok.type not in SKIP:
            out.append((tok.string, tok.start[0]))
    return out


def compare(en_nb, zh_nb, allowed):
    msgs = {(m["en"], m["zh"]) for m in allowed["messages"]}
    paths = {(p["cell"], p["en"], p["zh"]) for p in allowed["cell2_paths"]}
    errors = []
    ec, zc = en_nb["cells"], zh_nb["cells"]
    if [c["cell_type"] for c in ec] != [c["cell_type"] for c in zc]:
        return [f"格數或格的種類不同：英文版 {len(ec)} 格、中文版 {len(zc)} 格"]
    for i, (a, b) in enumerate(zip(ec, zc)):
        if a["cell_type"] != "code":
            continue
        ua, ub = units("".join(a["source"])), units("".join(b["source"]))
        if len(ua) != len(ub):
            errors.append(f"第 {i} 格：程式記號數不同（英文 {len(ua)}、中文 {len(ub)}）")
        for (ta, la), (tb, lb) in zip(ua, ub):
            if ta == tb or (ta, tb) in msgs or (i, ta, tb) in paths:
                continue
            errors.append(f"第 {i} 格，英文第 {la} 行／中文第 {lb} 行：{ta!r} ≠ {tb!r}")
            if len(errors) > 50:
                return errors
    return errors


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("en", nargs="?", default=str(EN)); ap.add_argument("zh", nargs="?", default=str(ZH))
    ap.add_argument("--allowed", default=str(ALLOWED))
    a = ap.parse_args(argv)
    en_nb, zh_nb = json.load(open(a.en, encoding="utf-8")), json.load(open(a.zh, encoding="utf-8"))
    errors = compare(en_nb, zh_nb, json.load(open(a.allowed, encoding="utf-8")))
    for e in errors:
        print("不一致：" + e)
    sha = hashlib.sha256(pathlib.Path(a.zh).read_bytes()).hexdigest()
    print(f"{'一致' if not errors else f'{len(errors)} 處不一致'}｜中文版 SHA-256 {sha}｜英文版 SHA-256 "
          f"{hashlib.sha256(pathlib.Path(a.en).read_bytes()).hexdigest()}")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
