# Data layout

`scripts/download_data.py` fills this folder from the Hugging Face dataset. Notebooks write the same layout when run from scratch.

Numbered folders carry the number of the notebook that produces them.

| Folder | Produced by | Contents |
| --- | --- | --- |
| `01_pilot_decoding/` | `notebooks/01` | 100 MAGE human prompts, pure/top-p continuations of three generators, GPT-2 XL scores. |
| `02_pilot_scorer_family/` | `notebooks/02` | The same texts scored by each generator itself. |
| `03_pilot_temperature/` | `notebooks/03` | Qwen2.5-3B at six temperatures; scores under GPT-2, GPT-2 XL, Qwen2.5-3B; TempTest cache. |
| `04_pilot_new_generators/` | `notebooks/04` | phi-2 and SmolLM2-1.7B at six temperatures. |
| `05_pilot_paraphrase/` | `notebooks/05`, `scripts/prepare/score_paraphrase.py` | Paraphrase sets and scores. |
| `06_scaled_sweep/` | `notebooks/06` | 1,000 MAGE humans, 8,100 continuations (3 generators, 7–13 temperatures), GPT-2 XL and self scores. |
| `07_falcon_dual_scoring/` | `notebooks/07` | Falcon dual scores (12 per-token rows) of `06_scaled_sweep`. |
| `08_m4gt_natural_text/` | `notebooks/08` | Falcon scores of the M4GT-Bench Subtask B evaluation manifest (no text). |
| `09_heldout_pythia/` | `notebooks/09` | pythia-2.8b continuations of held-out M4GT prompts (machine text only) and Falcon scores. |
| `10_official_equivalence/` | `notebooks/10` | Scores from our batched scorer and from the official detector code on the same 80 texts. |
| `11_trained_baselines/` | `notebooks/11`, `12` | Trained-baseline scores, DivEye features, AdaDetectGPT witness weights. |
| `13_theory_prediction_test/` | `notebooks/13` | Prospectively specified test of the human-only failure-temperature estimate (file names keep the word prediction): `predictions.json` (written before any generation), `predictions_corrected.json` (banned-token correction, written before the GPT-Neo and BLOOM text), self-scores of the M4GT held-out humans and of four new base models' continuations (machine text only). |
| `14_retokenization_diagnostic/` | `notebooks/14` | Pre-registered diagnostic: 300 fresh continuations per notebook-13 model at T = 1 with the sampled token ids kept (`gen_*.json`), scored through notebook 13's text re-encoding (`scores_text_*.pkl`) and through the sampled ids (`scores_ids_*.pkl`); `ids_check_*.csv` records whether re-encoding reproduced the context and continuation. |
| `mage_falcon_scores/` | `scripts/prepare/score_mage.py` | Falcon dual per-token scores of MAGE subsets (`train`, `cal`, `test`, `ood`). |
| `raid_scores/` | `scripts/a07_raid.py` | GPT-2 XL and Qwen2.5-3B per-token scores of the RAID sample (no text). |
| `manifests/` | `scripts/prepare/m4gt_*.py` | Frozen M4GT manifests with SHA-256 of each selected text. |

Third-party raw data are not redistributed. Put them in `external/` only if you re-generate from scratch:

| Path | Source | Used by |
| --- | --- | --- |
| `external/m4gt/SubtaskB.jsonl` | [M4GT-Bench](https://github.com/mbzuai-nlp/M4GT-Bench) Subtask B | `notebooks/08`, `09`, `11` (checked against `manifests/`) |
| `external/raid/train_none.csv` | [RAID](https://github.com/liamdugan/raid) | `scripts/a07_raid.py sample` |
| `external/mage/*.csv` | [MAGE](https://github.com/yafuly/MAGE) | `scripts/prepare/score_mage.py` |

The `set` column in `11_trained_baselines/` and the `src` column in `10_official_equivalence/` keep the names used
during the study: `main1` = `06_scaled_sweep`,
`extA` = `09_heldout_pythia`, `m4gt` = `08_m4gt_natural_text`, `train` = the training texts drawn in `notebooks/11`.

Per-token score arrays (`arr`) are stored only for the scored continuation. Rows are, in order: scorer surprisal, entropy,
log-probability variance, top-10 mass (4-row files); Falcon files add reference expectation and variance, observer surprisal,
ranks and cumulative probability below the observed token under both models, and observer entropy (12 rows).

Licenses: MAGE is released under Apache-2.0. M4GT-Bench and RAID texts are not included; follow their own terms when you
download them. Generated continuations are released for research use.
