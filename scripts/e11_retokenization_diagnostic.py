# Diagnostic (design doc 8.8.23, pre-registered 2026-09-30 before notebook 14 ran): does re-tokenizing the decoded text shift
# the self-scoring curvature c = mean(H - s) of machine text sampled at T = 1? Notebook 14 scored each of 300 fresh
# continuations per model twice: text path (notebook 13's re-encoding) and id path (the sampled token ids).
# Theory: E[c] = 0 on the id path up to the banned-token term; notebook 13 measured c on the text path at -0.016, -0.007, -0.020, -0.008.
# Pre-registered decisions (pooled = mean of the four model means; 95% intervals from 2,000 bootstrap resamples of texts within models):
#   H1 re-tokenization shifts c: the interval of the pooled paired difference D = c_text - c_ids excludes 0.
#   H2 the id path has no offset: the interval of the pooled c_ids includes 0.
#   H3 re-tokenization explains most of the offset: pooled c_text < 0 with its interval excluding 0, and D / c_text >= 0.5.
#   Verdict "re-tokenization is the main cause" if H1, H2 and H3 hold; "another source" if H2 fails; otherwise "inconclusive".
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1 if 'prepare' not in __file__ else 2]))
from cmargin.paths import DATA, RESULTS
(RESULTS / "e11_retokenization_diagnostic").mkdir(parents=True, exist_ok=True)
import numpy as np, pandas as pd

OUT = RESULTS / "e11_retokenization_diagnostic"
SRC = pathlib.Path(DATA) / "14_retokenization_diagnostic"
TAGS = ("OLMo-2-0425-1B", "granite-3.3-2b-base", "gpt-neo-1.3B", "bloom-1b7")
N_BOOT = 2000


def per_text(path):
    s = pd.read_pickle(path).sort_values("id")
    return pd.DataFrame(dict(id=s.id.values, c=[(a[1] - a[0]).mean() for a in s.arr], V=[a[2].mean() for a in s.arr]))


def ci(draws):
    return np.percentile(draws, 2.5), np.percentile(draws, 97.5)


def main():
    rng = np.random.default_rng(0); rows, boot = [], {k: np.zeros(N_BOOT) for k in ("c_text", "c_ids", "D")}
    for tag in TAGS:
        t, i = per_text(SRC / f"scores_text_{tag}.pkl"), per_text(SRC / f"scores_ids_{tag}.pkl")
        m = t.merge(i, on="id", suffixes=("_text", "_ids")); m["D"] = m.c_text - m.c_ids
        chk = pd.read_csv(SRC / f"ids_check_{tag}.csv")
        idx = rng.integers(0, len(m), (N_BOOT, len(m)))
        draws = {k: m[k].values[idx].mean(1) for k in ("c_text", "c_ids", "D")}
        for k in boot:
            boot[k] += draws[k] / len(TAGS)
        rows.append(dict(model=tag, n=len(m), share_changed=1 - (chk.same_prefix & chk.same_continuation).mean(),
                         mean_token_count_change=chk.token_count_change.mean(),
                         c_text=m.c_text.mean(), c_text_CI=ci(draws["c_text"]), c_ids=m.c_ids.mean(), c_ids_CI=ci(draws["c_ids"]),
                         D=m.D.mean(), D_CI=ci(draws["D"]), V_ids=m.V_ids.mean(),
                         shift_from_D=m.D.mean() / m.V_ids.mean(), shift_from_c_ids=m.c_ids.mean() / m.V_ids.mean()))
    per = pd.DataFrame(rows); per.to_csv(OUT / "per_model.csv", index=False)
    pooled = {k: np.mean([r[k] for r in rows]) for k in ("c_text", "c_ids", "D")}
    lo = {k: ci(boot[k]) for k in boot}
    h1 = not (lo["D"][0] <= 0 <= lo["D"][1])
    h2 = lo["c_ids"][0] <= 0 <= lo["c_ids"][1]
    h3 = pooled["c_text"] < 0 and lo["c_text"][1] < 0 and pooled["D"] / pooled["c_text"] >= 0.5
    verdict = "re-tokenization is the main cause" if (h1 and h2 and h3) else ("another source" if not h2 else "inconclusive")
    summary = dict(pooled_c_text=pooled["c_text"], pooled_c_text_CI=lo["c_text"], pooled_c_ids=pooled["c_ids"], pooled_c_ids_CI=lo["c_ids"],
                   pooled_D=pooled["D"], pooled_D_CI=lo["D"], share_of_offset=pooled["D"] / pooled["c_text"], H1=bool(h1), H2=bool(h2), H3=bool(h3), verdict=verdict)
    pd.DataFrame([summary]).to_csv(OUT / "summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(per.round(4).to_string(index=False)); print()
    for k, v in summary.items():
        print(f"{k}: {np.round(v, 4) if not isinstance(v, (bool, str)) else v}")


if __name__ == "__main__":
    main()
