# Score MAGE subsets (used for Appendix A MAGE families with MODEL=tiiuae/falcon-7b-instruct OBSERVER=tiiuae/falcon-7b).
# Score MAGE subsets with GPT-2 XL once; save per-token arrays so every
# feature group / filter can be computed offline from the same forward pass.
# Per position i (predicting token i+1): L = surprisal, E = entropy,
# V = variance of log-prob under the model (Fast-DetectGPT analytic), M = top-10 mass.
# With OBSERVER set (two-model mode, e.g. Falcon-7B + Falcon-7B-Instruct) three more rows:
# X = sum_v q_obs(v) log p(v)  (Fast-DetectGPT mean_ref; -X is Binoculars' cross-entropy),
# XV = sum_v q_obs(v) log p(v)^2 - X^2 (Fast-DetectGPT var_ref), Lo = surprisal under the observer.
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1 if 'prepare' not in __file__ else 2]))
from cmargin.paths import DATA, RESULTS
import os, sys, time
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MAX_TOK, MIN_TOK, BS, SEED = 512, 64, 8, 0
MAGE_CSV = os.environ.get("MAGE_DIR", str(DATA / "external" / "mage"))  # where the MAGE csv files live
MODEL = os.environ.get("MODEL", "gpt2")  # scoring model; DivEye default gpt2, WLiM gpt2-xl
OBSERVER = os.environ.get("OBSERVER")  # optional reference/observer model sharing MODEL's tokenizer
SELECT_TOK = os.environ.get("SELECT_TOK", MODEL)  # tokenizer used to pick documents; keep "gpt2" so every scorer sees the same docs
STRONG = ("gpt-3.5-trubo", "text-davinci-002", "text-davinci-003")  # MAGE spells turbo "trubo"
# split -> (source csv, {group: n}); fixed before seeing any result
SIZES = {
    "train": ("train", {"human": 3000, "ai": 3000}),
    "cal": ("valid", {"human": 1000, "ai": 1000}),
    "test": ("test", {"human": 2000, "strong": 1000, "weak": 1000}),
    "ood": ("test_ood_set_gpt", None),  # all of it: GPT-4 text, unseen domains
}


def group(d):
    ai = d.label == 0
    s = d.src.str.endswith(STRONG)
    return np.where(~ai, "human", np.where(s, "strong", "weak"))


def sample(split, tok, score_tok):
    src, want = SIZES[split]
    d = pd.read_csv(f"{MAGE_CSV}/{src}.csv").dropna(subset=["text"]).drop_duplicates("text")
    d["grp"] = group(d)
    if want:
        d["g2"] = np.where(d.grp == "human", "human", "ai") if "ai" in want else d.grp
        d = pd.concat([d[d.g2 == g].sample(min((d.g2 == g).sum(), n * 3), random_state=SEED) for g, n in want.items()])
    d["ids"] = [tok(t, truncation=True, max_length=MAX_TOK)["input_ids"] for t in d.text]
    n0 = len(d)
    d = d[d.ids.str.len() >= MIN_TOK]
    print(f"{split}: dropped {n0 - len(d)}/{n0} texts shorter than {MIN_TOK} tokens", flush=True)
    if want:
        d = pd.concat([d[d.g2 == g].head(n) for g, n in want.items()])
    if score_tok is not tok:
        d["ids"] = [score_tok(t, truncation=True, max_length=MAX_TOK)["input_ids"] for t in d.text]
    return d.reset_index(drop=True)


@torch.no_grad()
def score(ids, model, dev, pad, obs=None):
    order = np.argsort([len(x) for x in ids])
    out = [None] * len(ids)
    for b in range(0, len(order), BS):
        idx = order[b:b + BS]
        n = max(len(ids[i]) for i in idx)
        x = torch.full((len(idx), n), pad)
        att = torch.zeros((len(idx), n), dtype=torch.long)
        for r, i in enumerate(idx):
            x[r, :len(ids[i])] = torch.tensor(ids[i]); att[r, :len(ids[i])] = 1
        logits = model(x.to(dev), attention_mask=att.to(dev)).logits
        ologits = obs(x.to(dev), attention_mask=att.to(dev)).logits if obs is not None else None
        for r, i in enumerate(idx):
            k = len(ids[i])
            lp = torch.log_softmax(logits[r, :k - 1].float(), -1)
            p = lp.exp()
            tgt = torch.tensor(ids[i][1:], device=dev)
            L = -lp.gather(1, tgt[:, None])[:, 0]
            e1 = (p * lp).sum(-1)
            V = (p * lp * lp).sum(-1) - e1 ** 2
            M = p.topk(10, -1).values.sum(-1)
            rows = [L, -e1, V, M]
            if ologits is not None:
                olp = torch.log_softmax(ologits[r, :k - 1].float(), -1); q = olp.exp()
                X = (q * lp).sum(-1)
                rows += [X, (q * lp * lp).sum(-1) - X ** 2, -olp.gather(1, tgt[:, None])[:, 0]]
            out[i] = torch.stack(rows).cpu().numpy().astype(np.float32)
        if b % (BS * 100) == 0:
            print(f"  {b}/{len(order)} {time.strftime('%X')}", flush=True)
    return out


def main():
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dt = torch.bfloat16 if dev == "cuda" else torch.float16 if "xl" in MODEL else torch.float32
    tok = AutoTokenizer.from_pretrained(MODEL)
    sel_tok = tok if SELECT_TOK == MODEL else AutoTokenizer.from_pretrained(SELECT_TOK)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dt).to(dev).eval()
    obs = AutoModelForCausalLM.from_pretrained(OBSERVER, dtype=dt).to(dev).eval() if OBSERVER else None
    pad = tok.eos_token_id if tok.eos_token_id is not None else 0
    OUT_DIR = os.environ.get("OUT_DIR", str(DATA / "mage_falcon_scores"))
    os.makedirs(OUT_DIR, exist_ok=True)
    for split in sys.argv[1:] or SIZES:
        path = f"{OUT_DIR}/{split}.pkl"
        if os.path.exists(path):
            continue
        d = sample(split, sel_tok, tok)
        print(split, d.grp.value_counts().to_dict(), time.strftime("%X"), flush=True)
        d["arr"] = score(d.ids.tolist(), model, dev, pad, obs)
        d.drop(columns=["text", "ids"]).to_pickle(path)


if __name__ == "__main__":
    main()
