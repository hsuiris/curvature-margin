# Appendix A (MAGE by generator family): per-family Fast-DetectGPT/Binoculars AUROC with the Falcon dual scorer.
# Pilot 1 of the integrated study (design doc §7.1): is the zero-shot failure on MAGE's
# "weak" generators about model size or about alignment (instruction tuning / RLHF)?
# All choices below were fixed in the design doc before looking at any result:
# continuation prompts only, fixed generator classes, domain-matched AUROC, domains with
# < 30 human texts dropped, stratified bootstrap 1000.
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1 if 'prepare' not in __file__ else 2]))
from cmargin.paths import DATA, RESULTS
(RESULTS / "a01_mage_families").mkdir(parents=True, exist_ok=True)
import re
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

N_BOOT, MIN_HUMAN = 1000, 30
BASE = ["7B", "13B", "30B", "65B", "opt_125m", "opt_350m", "opt_1.3b", "opt_2.7b", "opt_6.7b",
        "opt_13b", "opt_30b", "bloom_7b", "gpt_j", "gpt_neox"]
INSTRUCT = ["flan_t5_small", "flan_t5_base", "flan_t5_large", "flan_t5_xl", "flan_t5_xxl",
            "t0_3b", "t0_11b", "opt_iml_30b", "opt_iml_max_1.3b", "text-davinci-002"]
RLHF = ["text-davinci-003", "gpt-3.5-trubo"]   # MAGE spells turbo "trubo"
CLASSES = {"base model": BASE, "instruction-tuned": INSTRUCT, "RLHF-aligned": RLHF, "GLM-130B (unclassified)": ["GLM130B"]}
PAIRS = {"OPT-30B vs OPT-IML-30B": ("opt_30b", "opt_iml_30b"),
         "OPT-1.3B vs OPT-IML-Max-1.3B": ("opt_1.3b", "opt_iml_max_1.3b")}
SIZE = {"OPT": ["opt_125m", "opt_350m", "opt_1.3b", "opt_2.7b", "opt_6.7b", "opt_13b", "opt_30b"],
        "LLaMA": ["7B", "13B", "30B", "65B"]}


def load(splits):
    d = pd.concat([pd.read_pickle(f"{DATA}/mage_falcon_scores/{k}.pkl") for k in splits], ignore_index=True)
    L = [a[0] for a in d.arr]; X = [a[4] for a in d.arr]; XV = [a[5] for a in d.arr]
    d["fdg"] = [(-l.sum() - x.sum()) / np.sqrt(v.sum()) for l, x, v in zip(L, X, XV)]   # higher = AI
    d["bino"] = [-(l.mean() / -x.mean()) for l, x in zip(L, X)]                         # negated: higher = AI
    d["n"] = [len(l) for l in L]
    m = d.src.str.extract(r"^(.+?)_machine_(continuation|topical|specified)_(.+)$")
    d["dom"] = np.where(d.label == 1, d.src.str.replace("_human", "", regex=False), m[0])
    d["prompt"], d["gen"] = m[1], m[2]
    ood = d.src.str.match(r"^(cnn|dialogsum|imdb|pubmed)_")   # MAGE's GPT-4 OOD file names its generator per domain
    d.loc[ood & (d.label == 0), ["dom", "gen", "prompt"]] = None
    d.loc[ood & (d.label == 0), "dom"] = d.src[ood & (d.label == 0)].str.split("_").str[0]
    d.loc[ood & (d.label == 0), "gen"] = "gpt-4"
    return d


def matched_auroc(d, ai_mask, col, rng=None):
    """AUROC within each domain (AI of the group vs humans of that domain), weighted by #AI."""
    num = den = 0.0
    for dom, g in d[ai_mask].groupby("dom"):
        h = d[(d.label == 1) & (d.dom == dom)]
        if len(h) < MIN_HUMAN:
            continue
        a, hh = g[col].values, h[col].values
        if rng is not None:
            a, hh = rng.choice(a, len(a)), rng.choice(hh, len(hh))
        y = np.r_[np.ones(len(a)), np.zeros(len(hh))]
        num += len(a) * roc_auc_score(y, np.r_[a, hh]); den += len(a)
    return num / den if den else np.nan


def summarize(d, ai_mask, rng):
    out = {"n_ai": int(ai_mask.sum()), "median_tokens": float(np.median(d.n[ai_mask]))}
    for col, name in (("fdg", "Fast-DetectGPT"), ("bino", "Binoculars")):
        boot = [matched_auroc(d, ai_mask, col, rng) for _ in range(N_BOOT)]
        out[name] = matched_auroc(d, ai_mask, col)
        out[name + " CI"] = f"[{np.percentile(boot, 2.5):.3f}, {np.percentile(boot, 97.5):.3f}]"
    return out


def main():
    rng = np.random.default_rng(0)
    d = load(("train", "cal", "test"))
    cont = (d.label == 0) & (d.prompt == "continuation")
    rows = [dict(group=c, **summarize(d, cont & d.gen.isin(g), rng)) for c, g in CLASSES.items()]
    o = load(("ood",))
    rows.append(dict(group="GPT-4 new domains (RLHF, mixed prompts)", **summarize(o, o.label == 0, rng)))
    R = pd.DataFrame(rows)

    per = []
    for c, gens in CLASSES.items():
        for g in gens:
            m = cont & (d.gen == g)
            per.append(dict(group=c, generator=g, n_ai=int(m.sum()), median_tokens=float(np.median(d.n[m])),
                            FDG=matched_auroc(d, m, "fdg"), Binoculars=matched_auroc(d, m, "bino")))
    P = pd.DataFrame(per)

    diffs = []
    for name, (a, b) in PAIRS.items():
        ma, mb = cont & (d.gen == a), cont & (d.gen == b)
        for col in ("fdg", "bino"):
            boot = [matched_auroc(d, mb, col, rng) - matched_auroc(d, ma, col, rng) for _ in range(N_BOOT)]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            diffs.append(dict(contrast=name, score=col, base=matched_auroc(d, ma, col), instruct=matched_auroc(d, mb, col),
                              diff=matched_auroc(d, mb, col) - matched_auroc(d, ma, col), CI=f"[{lo:.3f}, {hi:.3f}]",
                              CI_excludes_0="yes" if lo > 0 or hi < 0 else "no"))
    Dp = pd.DataFrame(diffs)

    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    print("== Grouped by alignment (continuation prompts, domain-matched AUROC) =="); print(R.round(3).to_string(index=False))
    print("\n== Same family and size: instruct minus base =="); print(Dp.round(3).to_string(index=False))
    print("\n== Per generator =="); print(P.round(3).to_string(index=False))
    for fam, gens in SIZE.items():
        s = P.set_index("generator").loc[gens, "FDG"]
        print(f"\nSize trend {fam} (FDG):", "  ".join(f"{g}={v:.3f}" for g, v in s.items()))
    R.to_csv(f"{RESULTS}/a01_mage_families/groups.csv", index=False)
    P.to_csv(f"{RESULTS}/a01_mage_families/per_generator.csv", index=False)
    Dp.to_csv(f"{RESULTS}/a01_mage_families/pairs.csv", index=False)


def check():
    d = pd.DataFrame(dict(label=[1, 1, 0, 0], dom=["x"] * 4, fdg=[0., 1., 2., 3.]))
    global MIN_HUMAN
    MIN_HUMAN = 1
    assert matched_auroc(d, d.label == 0, "fdg") == 1.0
    d.loc[d.label == 0, "fdg"] = [-1., -2.]
    assert matched_auroc(d, d.label == 0, "fdg") == 0.0
    print("check ok")


if __name__ == "__main__":
    import sys
    check() if sys.argv[1:] == ["check"] else main()
