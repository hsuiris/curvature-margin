"""Synthetic inputs for e12 (plan 8.8.24): per-model human and machine Fast-DetectGPT scores built so that each AUROC
curve falls through 0.5 near a chosen temperature, plus the registered predictions computed from the human scores."""
import copy, hashlib, json, pathlib, pickle, sys

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import e12_replication_test as e12   # noqa: E402

N_P, N_E, N_A, N_B = 1000, 1000, 150, 150


def q(n, rng=None):
    """Deterministic standard-normal quantiles (symmetric around 0), shuffled when rng is given."""
    x = norm.ppf((np.arange(n) + 0.5) / n)
    return x if rng is None else rng.permutation(x)


def ids():
    A = np.arange(N_A); B = np.arange(N_A, N_A + N_B); E = np.arange(N_E); P = np.arange(1000, 1000 + N_P)
    return A, B, E, P


def source_of(i):
    return e12.SOURCES[i % 6]


def model(name, family, T_star, T_hat=None, **kw):
    """Model spec. T_star: crossing of every half; kw can override T_A (A half vs P), T_B (B half vs E minus A),
    pattern ('cross', 'above', 'below', 'up', 'flatA_aboveB'), kappa, missing ('text' / 'ids' / 'humans'), written."""
    d = dict(name=name, family=family, T_star=T_star, T_hat=T_star if T_hat is None else T_hat, T_A=None, T_B=None,
             pattern="cross", kappa=60.0, missing=None, written=True, tok_group=family, lang_group="other", strip_group="keep")
    d.update(kw)
    return d


def build(specs, temps=e12.TEMPS, seed=0):
    """An e12 input dict (the structure load_inputs returns) for the given model specs."""
    A, B, E, P = ids(); temps = np.asarray(temps, float)
    prompts = np.sort(np.r_[A, B]); isA = np.isin(prompts, A)
    inp = dict(temps=temps, P_ids=P, E_ids=E, prompts=prompts, P_src=np.array([source_of(i) for i in P]),
               E_src=np.array([source_of(i) for i in E]),
               E_role=np.array(["prompt_A" if i in set(A) else "prompt_B" if i in set(B) else "reference" for i in E]),
               A_ids=set(A.tolist()), ref_sizes=dict(main=N_E, B=N_E - N_A, direct=N_P), models=[], r1_commit="synthetic", dryrun=False)
    for no, s in enumerate(specs, start=1):
        rng = np.random.default_rng(seed + no)
        m = dict(no=no, name=s["name"], family=s["family"], tok_group=s["tok_group"], lang_group=s["lang_group"],
                 strip_group=s["strip_group"], issues=[], meta={}, pred=None, hP=None, hE=None, m_text=None, m_ids=None,
                 revision="r", start=[], banned=[0], banned_sha256="x", pad=0)
        V = 4.0 + 0.4 * q(N_P, rng); c = (1 - s["T_hat"]) * V + 0.05 * q(N_P, rng)
        fdgP = q(N_P, rng); sv = 30 + 3 * q(N_P, rng)
        if s["pattern"] == "flatA_aboveB":
            fdgP = np.zeros(N_P)          # A half and P humans all tie: the A-half AUROC is exactly 0.5 at every temperature
        hP = dict(c=c, V=V, fdg=fdgP, sv=sv)
        VE = 4.0 + 0.4 * q(N_E, rng); hE = dict(c=(1 - s["T_hat"] - 0.001) * VE + 0.05 * q(N_E, rng), V=VE, fdg=q(N_E, rng), sv=30 + 3 * q(N_E, rng))
        z = q(len(prompts), rng); k = s["kappa"]
        TA = s["T_A"] if s["T_A"] is not None else s["T_star"]; TB = s["T_B"] if s["T_B"] is not None else s["T_star"]
        mu = np.where(isA[None, :], -k * (temps[:, None] - TA), -k * (temps[:, None] - TB))
        if s["pattern"] == "above":
            mu = np.full_like(mu, 3.0) - 0.5 * (temps[:, None] - 1)
        elif s["pattern"] == "below":
            mu = np.full_like(mu, -3.0) - 0.5 * (temps[:, None] - 1)
        elif s["pattern"] == "up":
            mu = k * (temps[:, None] - s["T_star"]) + 0 * mu
        M = mu + z[None, :]
        if s["pattern"] == "flatA_aboveB":
            M = np.where(isA[None, :], 0.0, 5.0 + z[None, :] + 0 * mu)
        if s["missing"] != "humans":
            m["hP"], m["hE"] = hP, hE
        if s["missing"] != "text":
            m["m_text"] = M
        if s["missing"] != "ids":
            m["m_ids"] = M - 0.05
        if s["written"] and s["missing"] != "humans":
            vb, vs = e12.variant_b(hP["fdg"], hP["sv"])
            m["pred"] = dict(status="written", T_hat=e12.formula(hP["c"], hP["V"]), T_hat_E=e12.formula(hE["c"], hE["V"]),
                             T_A=e12.variant_a(hP["fdg"], hP["sv"]), variant_B=dict(value=vb, status=vs),
                             baselines=e12.baseline_values(float(np.mean(hP["c"]))))
        elif not s["written"]:
            m["issues"].append("預測未寫出：synthetic")
        if s["missing"] == "text":
            m["issues"].append("文字路徑：synthetic file missing")
        inp["models"].append(m)
    return inp


def standard(**over):
    """12 models, 11 families (Danube has two), failure temperatures spread like the known models, formula close."""
    T = [1.004, 1.011, 1.017, 1.006, 1.013, 1.0035, 1.002, 1.009, 1.015, 1.0075, 1.019, 1.012]
    fams = ["Llama", "Gemma", "TinyLlama", "RedPajama", "StableLM", "Danube", "Danube", "EuroLLM", "Salamandra", "XGLM", "Mamba", "Pleias"]
    specs = [model(f"m{i + 1}", f, t, T_hat=t + (0.0006 if i % 2 else -0.0005)) for i, (f, t) in enumerate(zip(fams, T))]
    for i, kw in over.items():
        specs[int(i[1:])].update(kw)
    return specs


# ------------------------------------------------------------------ notebook-format files, for the load_inputs test
WRITTEN = "2026-10-20T12:00:00+00:00"          # every synthetic stage-2 file; stage 2 "started" STARTED
STARTED = "2026-10-15T00:00:00+00:00"


ENV = dict(python="3.12.11", torch="2.8.0+cu126", numpy="2.0.2", scipy="1.16.2", cuda="12.6")


def write_files(inp, data_dir, temps=e12.TEMPS, written=WRITTEN, started=STARTED, attempts=None, env_of=None):
    """env_of(model name, temperature) -> the environment recorded in that stage-2 file (default: one environment)."""
    env_of = env_of or (lambda name, t: ENV)
    """Writes the files notebook 15 would write (predictions, r1_models, run_config, stage2_go, humans, gen, scores),
    a manifest and a registration.json, from a synthetic input. Machine texts get one scored token per value."""
    data_dir = pathlib.Path(data_dir); out = data_dir / e12.OUT_NAME; (data_dir / "manifests").mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    A, B, E, P = ids()
    rows = []
    for i in E:
        role = "prompt_A" if i < N_A else "prompt_B" if i < N_A + N_B else "reference"
        rows.append((int(i), "E", role))
    rows += [(int(i), "P", "P") for i in P]
    import pandas as pd
    man = pd.DataFrame([dict(id=i, group=g, role=r, line_number=10 * i + 1, source=source_of(i), word_count=150,
                             text_sha256="0" * 64, norm_sha256="0" * 64) for i, g, r in rows])
    man.to_csv(data_dir / e12.MANIFEST, index=False)
    man_sha = e12.sha256_file(data_dir / e12.MANIFEST)
    entries, preds = [], {}
    for m in inp["models"]:
        e = dict(no=m["no"], name=m["name"], revision=m["revision"], family=m["family"], tok_group=m["tok_group"],
                 lang_group=m["lang_group"], strip_group=m["strip_group"], start=[], banned=[0], banned_sha256="x", pad=0)
        entries.append(e)
        preds[m["name"]] = dict(m["pred"], model=m["name"]) if m["pred"] else dict(status="not_written", model=m["name"], reason="synthetic")

    def cfg(e, kind, path, temp):
        return dict(schema=e12.SCHEMA, kind=kind, path=path, model=e["name"], revision=e["revision"], start=e["start"],
                    banned_sha256=e["banned_sha256"], pad=e["pad"], temperature=temp, manifest_sha256=man_sha, r1_commit="synthetic")

    for m, e in zip(inp["models"], entries):
        tag = e12.tag_of(m["name"])
        if m["hP"] is not None:
            srows = []
            for g, idl in (("hP", P), ("hE", E)):
                for j, i in enumerate(idl):
                    fdg = m[g]["fdg"][j]
                    srows.append(dict(id=int(i), ids=np.arange(31, dtype=np.int32), k=30, cut=0, j=30, len_p=30, same_prefix=True,
                                      same_cont=True, arr=np.array([[0.0], [fdg], [1.0], [0.5]], np.float32)))
            c = cfg(e, "human", "text", None)
            pickle.dump(dict(schema=e12.SCHEMA, kind="human", path="text", config=c, config_hash=e12.config_hash(c), environment=ENV,
                             rows=srows), open(out / f"humans_{tag}.pkl", "wb"))
        for ti, t in enumerate(temps):
            if m["m_text"] is None and m["m_ids"] is None:
                continue
            c = cfg(e, "gen", "gen", float(t))
            gen = [dict(id=int(i), j=1, prompt=[5], target=1, ids=[7]) for i in inp["prompts"]]
            json.dump(dict(schema=e12.SCHEMA, kind="gen", model=m["name"], temperature=float(t), config=c, config_hash=e12.config_hash(c),
                           written_utc=written, environment=env_of(m["name"], t), rows=gen), open(out / f"gen_{tag}_T{e12.fmt_t(t)}.json", "w"))
            for path, M in (("text", m["m_text"]), ("ids", m["m_ids"])):
                if M is None:
                    continue
                c = cfg(e, "machine", path, float(t))
                srows = [dict(id=int(i), ids=np.array([5, 7], np.int32), k=1, cut=0, j=1, len_p=1, same_prefix=True, same_cont=True,
                              arr=np.array([[0.0], [M[ti, j]], [1.0], [0.5]], np.float32)) for j, i in enumerate(inp["prompts"])]
                pickle.dump(dict(schema=e12.SCHEMA, kind="machine", path=path, config=c, config_hash=e12.config_hash(c),
                                 written_utc=written, environment=env_of(m["name"], t), rows=srows),
                            open(out / f"scores_{path}_{tag}_T{e12.fmt_t(t)}.pkl", "wb"))
    # predictions must be what the stored human scores give (the synthetic arr has c = fdg, V = 1)
    for m, e in zip(inp["models"], entries):
        if m["pred"] is None or m["hP"] is None:
            continue
        cP = m["hP"]["fdg"].astype(np.float32).astype(float); cE = m["hE"]["fdg"].astype(np.float32).astype(float)
        one = np.ones_like(cP)
        vb, vs = e12.variant_b(cP, one)
        preds[m["name"]].update(T_hat=e12.formula(cP, one), T_hat_E=e12.formula(cE, np.ones_like(cE)), T_A=e12.variant_a(cP, one),
                                variant_B=dict(value=vb, status=vs), baselines=e12.baseline_values(float(np.mean(cP))))
    json.dump(dict(r1_commit="synthetic", models=preds), open(out / "predictions.json", "w"))
    json.dump(dict(r1_tag="prereg-R1", r1_commit="synthetic", models=entries), open(out / "r1_models.json", "w"))
    json.dump(dict(dryrun=False, temps=list(temps), P_ids=P.tolist(), E_ids=E.tolist(), A_ids=A.tolist(), B_ids=B.tolist(),
                   quota={k: list(v) for k, v in e12.QUOTA.items()}), open(out / "run_config.json", "w"))
    psha = e12.sha256_file(out / "predictions.json")
    json.dump(dict(predictions_sha256=psha, r2_tag="prereg-R2", r2_commit="x", ots_block_height=1,
                   ots_block_time="2026-10-14T00:00:00+00:00"), open(out / "stage2_go.json", "w"))
    json.dump(dict(utc=started), open(out / "stage2_started.json", "w"))
    json.dump(attempts or {}, open(out / "attempts.json", "w"))
    reg = dict(R0=dict(pe_manifest_sha256=man_sha), R1=dict(models=entries), R2=dict(predictions_sha256=psha))
    json.dump(reg, open(data_dir / "registration.json", "w"))
    return data_dir / "registration.json"
