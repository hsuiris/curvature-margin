# Section 5.5 / RAID: sample, score (GPT-2 XL, Qwen2.5-3B) and analyze the non-adversarial RAID split.
# Pilot 7c (design doc §8.8.4): RAID has per-text decoding labels, MAGE does not.
# Stage "sample": pick N_AI texts per (model, decoding, repetition_penalty, domain) and N_HUMAN humans per
# domain from train_none.csv, fixed seed, before any scoring.
# Stage "score": per-position surprisal / entropy / var / top-10 under one scorer, on MPS if available.
# Stage "analyze": domain-matched AUROC (AI of a cell vs humans of that domain) and the surprisal margin.
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1 if 'prepare' not in __file__ else 2]))
from cmargin.paths import DATA, RESULTS
(RESULTS / "a07_raid").mkdir(parents=True, exist_ok=True)
import os, sys, time
import numpy as np, pandas as pd

DOMAINS = ["news", "wiki", "reddit", "abstracts"]
N_AI, N_HUMAN, SEED, MAX_TOK, MIN_TOK = 40, 200, 0, 512, 64
BASE = {"gpt2", "mpt", "mistral"}                       # open base models; RAID applies repetition penalty only to open models
CHAT = {"llama-chat", "mistral-chat", "mpt-chat", "cohere-chat", "chatgpt", "gpt4", "gpt3", "cohere"}


def sample():
    d = pd.read_csv(f"{DATA}/external/raid/train_none.csv", usecols=["id", "model", "decoding", "repetition_penalty", "domain", "generation"])
    d = d[d.domain.isin(DOMAINS)].dropna(subset=["generation"]).drop_duplicates("generation")
    d["words"] = d.generation.str.split().str.len()
    d = d[d.words >= 80]
    parts = [d[d.model == "human"].groupby("domain").sample(N_HUMAN, random_state=SEED)]
    ai = d[d.model != "human"]
    for key, g in ai.groupby(["model", "decoding", "repetition_penalty", "domain"]):
        parts.append(g.sample(min(N_AI, len(g)), random_state=SEED))
    s = pd.concat(parts, ignore_index=True)
    os.makedirs(f"{DATA}/raid_scores", exist_ok=True)
    s.to_pickle(f"{DATA}/raid_scores/sample.pkl")
    print(s.groupby(["model", "decoding", "repetition_penalty"]).size().to_string())
    print("humans:", (s.model == "human").sum(), "total:", len(s))


def score(scorer):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dt = torch.float16 if dev != "cpu" else torch.float32
    tok = AutoTokenizer.from_pretrained(scorer)
    model = AutoModelForCausalLM.from_pretrained(scorer, dtype=dt).to(dev).eval()
    s = pd.read_pickle(f"{DATA}/raid_scores/sample.pkl")
    # humans and open base models first, so a partial run already answers the family question
    s = s.assign(_pri=np.where(s.model == "human", 0, np.where(s.model.isin(BASE), 1, 2))).sort_values("_pri", kind="stable")
    out_path = f"{DATA}/raid_scores/scores_{scorer.split('/')[-1]}.pkl"
    done = pd.read_pickle(out_path) if os.path.exists(out_path) else None
    done_ids = set(done.id) if done is not None else set()
    rows, t0 = [], time.time()
    with torch.no_grad():
        for i, r in enumerate(s.itertuples()):
            if r.id in done_ids:
                continue
            ids = tok(r.generation, truncation=True, max_length=MAX_TOK)["input_ids"]
            if len(ids) < MIN_TOK:
                continue
            x = torch.tensor([ids], device=dev)
            lp = torch.log_softmax(model(x).logits[0, :-1].float(), -1)
            p = lp.exp(); e1 = (p * lp).sum(-1)
            arr = torch.stack([-lp.gather(1, x[0, 1:, None])[:, 0], -e1, (p * lp * lp).sum(-1) - e1 ** 2,
                               p.topk(10, -1).values.sum(-1)]).cpu().numpy().astype(np.float32)
            rows.append(dict(id=r.id, model=r.model, decoding=r.decoding, repetition_penalty=r.repetition_penalty,
                             domain=r.domain, arr=arr))
            if dev == "mps" and len(rows) % 50 == 0:
                torch.mps.empty_cache()   # the process was SIGKILLed twice after ~2000 texts; keep the MPS cache small
            if len(rows) % 250 == 0:
                print(f"  {scorer}: {i}/{len(s)} {(time.time() - t0) / 60:.0f} min", flush=True)
                pd.concat([done, pd.DataFrame(rows)] if done is not None else [pd.DataFrame(rows)]).to_pickle(out_path)
    pd.concat([done, pd.DataFrame(rows)] if done is not None else [pd.DataFrame(rows)]).to_pickle(out_path)
    print("done", scorer, len(rows), "new rows", f"{(time.time() - t0) / 60:.0f} min")
    open(f"{DATA}/raid_scores/score_{scorer.split('/')[-1]}.done", "w").write("done")


def analyze():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 200)
    allr = []
    for path in sorted(os.listdir(f"{DATA}/raid_scores")):
        if not path.startswith("scores_"):
            continue
        scorer = path[7:-4]
        d = pd.read_pickle(f"{DATA}/raid_scores/{path}")
        d["surprisal"] = [a[0].mean() for a in d.arr]
        d["fdg"] = [(a[1].sum() - a[0].sum()) / np.sqrt(a[2].sum()) for a in d.arr]
        hum = d[d.model == "human"]
        rows = []
        for (m, dec, rp), g in d[d.model != "human"].groupby(["model", "decoding", "repetition_penalty"]):
            num = den = 0.0; margins = []; boot = np.zeros(1000)
            for dom, gg in g.groupby("domain"):
                h = hum[hum.domain == dom]
                y = np.r_[np.ones(len(gg)), np.zeros(len(h))]; sc = np.r_[gg.fdg.values, h.fdg.values]
                num += len(gg) * roc_auc_score(y, sc); den += len(gg)
                margins.append(h.surprisal.mean() - gg.surprisal.mean())
                for b in range(1000):
                    ia, ih = rng.integers(0, len(gg), len(gg)), rng.integers(0, len(h), len(h))
                    boot[b] += len(gg) * roc_auc_score(np.r_[np.ones(len(gg)), np.zeros(len(h))], np.r_[gg.fdg.values[ia], h.fdg.values[ih]])
            boot /= den
            rows.append(dict(scorer=scorer, generator=m, category="base" if m in BASE else "aligned_or_commercial", decoding=dec, repetition_penalty=rp, n_texts=len(g),
                             surprisal_margin=float(np.mean(margins)), AUROC=num / den,
                             CI=f"[{np.percentile(boot, 2.5):.3f}, {np.percentile(boot, 97.5):.3f}]"))
        R = pd.DataFrame(rows).sort_values(["category", "generator", "decoding", "repetition_penalty"])
        print(f"\n== RAID, scorer {scorer}, domain-matched AUROC ==")
        print(R.round(3).to_string(index=False))
        from scipy.stats import spearmanr
        rho = spearmanr(R.surprisal_margin, R.AUROC)
        print(f"Spearman correlation, margin vs AUROC: {rho.statistic:.3f} (p={rho.pvalue:.2e}, n={len(R)})")
        allr.append(R)
    pd.concat(allr).to_csv(f"{RESULTS}/a07_raid/metrics.csv", index=False)


if __name__ == "__main__":
    stage = sys.argv[1]
    sample() if stage == "sample" else score(sys.argv[2]) if stage == "score" else analyze()
