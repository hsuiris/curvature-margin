# Score paraphrase sets locally (Apple MPS or CUDA) for a12_paraphrase.py.
# Pilot 11 scoring on the Mac (MPS): whole-text scoring of human originals, the two AI continuation sources and the
# six paraphrase sets. Same per-position rows as the other pilots. Checkpoints every 100 texts; a .done flag on completion.
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1 if 'prepare' not in __file__ else 2]))
from cmargin.paths import DATA, RESULTS
import glob, json, os, sys, time
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForCausalLM



def records():
    humans = {h["id"]: h["text"] for h in json.load(open(f"{DATA}/01_pilot_decoding/humans.json"))}
    t08 = {r["id"]: r["prefix"] + r["ai"] for r in json.load(open(f"{DATA}/03_pilot_temperature/gen_T0.8.json"))}
    t10 = {r["id"]: r["prefix"] + r["ai"] for r in json.load(open(f"{DATA}/03_pilot_temperature/gen_T1.0.json"))}
    recs = [dict(id=i, group="human", text=t) for i, t in humans.items()]
    recs += [dict(id=i, group="ai_T0.8", text=t) for i, t in t08.items()]
    recs += [dict(id=i, group="ai_T1.0", text=t) for i, t in t10.items()]
    for f in sorted(glob.glob(f"{DATA}/05_pilot_paraphrase/para_*.json")):
        for r in json.load(open(f)):
            recs.append(dict(id=r["id"], group=f"para_{r['source']}_{r['decoding']}", text=r["paraphrase"]))
    return recs


def score(scorer):
    tag = scorer.split("/")[-1]; out_path = f"{DATA}/05_pilot_paraphrase/scores_{tag}.pkl"
    if os.path.exists(f"{DATA}/05_pilot_paraphrase/score_{tag}.done"):
        print("already done", scorer); return
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(scorer)
    model = AutoModelForCausalLM.from_pretrained(scorer, dtype=torch.float16 if dev == "mps" else torch.float32).to(dev).eval()
    done = pd.read_pickle(out_path) if os.path.exists(out_path) else None
    done_keys = set(zip(done.id, done.group)) if done is not None else set()
    rows, t0 = [], time.time()
    with torch.no_grad():
        for i, r in enumerate(records()):
            if (r["id"], r["group"]) in done_keys:
                continue
            ids = tok(r["text"], truncation=True, max_length=512)["input_ids"]
            if len(ids) < 16:
                continue
            x = torch.tensor([ids], device=dev)
            lp = torch.log_softmax(model(x).logits[0, :-1].float(), -1)
            p = lp.exp(); e1 = (p * lp).sum(-1)
            arr = torch.stack([-lp.gather(1, x[0, 1:, None])[:, 0], -e1, (p * lp * lp).sum(-1) - e1 ** 2,
                               p.topk(10, -1).values.sum(-1)]).cpu().numpy().astype(np.float32)
            rows.append(dict(id=r["id"], group=r["group"], scorer=tag, arr=arr))
            if dev == "mps" and len(rows) % 50 == 0:
                torch.mps.empty_cache()
            if len(rows) % 100 == 0:
                pd.concat([done, pd.DataFrame(rows)] if done is not None else [pd.DataFrame(rows)]).to_pickle(out_path)
                print(f"  {tag}: {i} {(time.time() - t0) / 60:.0f} min", flush=True)
    pd.concat([done, pd.DataFrame(rows)] if done is not None else [pd.DataFrame(rows)]).to_pickle(out_path)
    open(f"{DATA}/05_pilot_paraphrase/score_{tag}.done", "w").write("done")
    print("done", scorer, f"{(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    score(sys.argv[1])
