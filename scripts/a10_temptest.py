# Appendix B: TempTest exact vs first/second-order approximations, AUROC by temperature.
# Pilot 9 (design doc §8.8.8): TempTest (Kempton, Burrell, Cheverall, arXiv 2503.20421) vs the margin.
# Per position i with scorer distribution p_i and observed token x_i, temperature tau, beta = 1/tau:
#   TempTest_i = log Z_i(beta) - (beta - 1) log p_i(x_i),  Z_i(beta) = sum_v p_i(v)^beta
#             = log p_i(x_i) - log q_i(x_i)             (q = p^beta / Z, the temperature-scaled scorer)
#   first order in (beta - 1):  -(beta - 1) * (H(p_i) - s_i)            (= minus Fast-DetectGPT curvature)
#   second order adds:          (beta - 1)^2 / 2 * Var_p(log p_i)        (= the Fast-DetectGPT variance term)
# The script computes the exact statistic and both approximations on the pilot-5 texts with GPT-2 as scorer.
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1 if 'prepare' not in __file__ else 2]))
from cmargin.paths import DATA, RESULTS
(RESULTS / "a10_temptest").mkdir(parents=True, exist_ok=True)
import glob, json, sys, time
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModelForCausalLM

SCORER, TAUS, MAX_TOK = "gpt2", [0.7, 0.8, 0.9, 1.1, 1.3], 512


def score_all():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(SCORER)
    model = AutoModelForCausalLM.from_pretrained(SCORER, dtype=torch.float32).to(dev).eval()
    rows, t0 = [], time.time()
    seen_h = set()
    for f in sorted(glob.glob(f"{DATA}/03_pilot_temperature/gen_T*.json")):
        T = float(f.split("gen_T")[1][:-5])
        for r in json.load(open(f)):
            for who in ("human", "ai"):
                if who == "human" and r["id"] in seen_h:
                    continue
                k = len(tok(r["prefix"])["input_ids"])
                ids = tok(r["prefix"] + r[who], truncation=True, max_length=MAX_TOK)["input_ids"]
                x = torch.tensor([ids], device=dev)
                with torch.no_grad():
                    lp = torch.log_softmax(model(x).logits[0, :-1].float(), -1)[k - 1:]   # continuation only
                tgt = x[0, k:]
                s = -lp.gather(1, tgt[:, None])[:, 0]                     # surprisal
                p = lp.exp(); H = -(p * lp).sum(-1)                        # entropy
                V = (p * lp * lp).sum(-1) - (p * lp).sum(-1) ** 2          # Var_p(log p)
                rec = dict(id=r["id"], T=T, who=who, y=int(who == "ai"), n=len(s),
                           s=s.mean().item(), H=H.mean().item(), V=V.mean().item(), curv=(H - s).mean().item())
                for tau in TAUS:
                    b = 1.0 / tau
                    logZ = torch.logsumexp(b * lp, -1)                       # log sum_v p^beta
                    exact = (logZ - (b - 1) * (-s)).mean().item()            # log Z - (beta-1) log p(x)
                    rec[f"tt{tau}"] = exact
                    rec[f"tt1_{tau}"] = (-(b - 1) * (H - s)).mean().item()
                    rec[f"tt2_{tau}"] = (-(b - 1) * (H - s) + (b - 1) ** 2 / 2 * V).mean().item()
                rows.append(rec)
                if who == "human":
                    seen_h.add(r["id"])
        print(f"  T={T} done {(time.time() - t0) / 60:.1f} min", flush=True)
    d = pd.DataFrame(rows); d.to_pickle(f"{DATA}/03_pilot_temperature/temptest_gpt2.pkl"); return d


def analyze(d):
    pd.set_option("display.width", 220)
    hum = d[d.who == "human"]
    print("== Pearson correlation of the approximations with the exact value (700 texts) ==")
    for tau in TAUS:
        print(f"tau={tau}: first order {np.corrcoef(d[f'tt{tau}'], d[f'tt1_{tau}'])[0, 1]:.4f}, second order {np.corrcoef(d[f'tt{tau}'], d[f'tt2_{tau}'])[0, 1]:.4f}, "
              f"correlation with curvature (H-s) {np.corrcoef(d[f'tt{tau}'], d.curv)[0, 1]:.4f}")
    rows = []
    for T, g in d[d.who == "ai"].groupby("T"):
        both = pd.concat([g, hum])
        r = {"gen_temperature": T, "surprisal_margin": hum.s.mean() - g.s.mean(), "AUROC_FDG_numerator": roc_auc_score(both.y, both.curv)}
        for tau in TAUS:
            r[f"TempTest_AUROC_tau{tau}"] = roc_auc_score(both.y, -both[f"tt{tau}"])   # TempTest: lower = machine
            r[f"share_AI_judged_human_tau{tau}"] = (g[f"tt{tau}"] > 0).mean()
        rows.append(r)
    R = pd.DataFrame(rows)
    print("\n== Per generation temperature: AUROC of TempTest (negated; higher = more AI-like) and of the Fast-DetectGPT numerator ==")
    print(R.round(3).to_string(index=False))
    print(f"\nHuman texts: TempTest tau=0.8 mean {hum['tt0.8'].mean():.4f}, share judged human (>0) {(hum['tt0.8'] > 0).mean():.3f}")
    R.to_csv(f"{RESULTS}/a10_temptest/temptest.csv", index=False)


if __name__ == "__main__":
    d = pd.read_pickle(f"{DATA}/03_pilot_temperature/temptest_gpt2.pkl") if sys.argv[1:] == ["analyze"] else score_all()
    analyze(d)
