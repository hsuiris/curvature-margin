---
license: other
language:
- en
task_categories:
- text-classification
tags:
- machine-generated-text-detection
- zero-shot-detection
- ai-text-detection
pretty_name: Curvature Margin - paired continuations and per-token scores
size_categories:
- 10K<n<100K
---

# Curvature Margin: paired continuations and per-token scores

Released data for *Estimating Self-Scoring AI-Text Detectors' Failure Temperatures from Human Text*. Code: https://github.com/hsuiris/curvature-margin

## Contents
- **Paired continuations**: base generators (Qwen2.5-3B, phi-2, SmolLM2-1.7B, pythia-2.8b) continue the first 30 tokens of human
  documents at 7–13 sampling temperatures (top-p 1, top-k off) to the length of the human continuation; 300 per cell.
- **Per-token scores** under GPT-2 XL, each generator itself, and Falcon-7B-Instruct with Falcon-7B as reference (12 rows per token).
- **Trained-baseline scores** (MAGE Longformer, RoBERTa OpenAI detector, DivEye, AdaDetectGPT).

The folder layout, file formats and the notebook that produced each folder are described in `data/README.md` of the code repository.

## Not included
The first AdaDetectGPT run, whose witness function degenerated to a constant (see the paper), is omitted; `11_trained_baselines/scores_ada_v2.csv` is the run reported in the paper.

Human texts from M4GT-Bench and RAID, and any M4GT text embedded in generated files (fields `prefix` and `human` are removed from
`09_heldout_pythia/gen_*.json`). Rebuild them from the public sources with the frozen manifests in the code repository.

## Licenses
MAGE human texts: Apache-2.0. Generated continuations: released for research use; they are subject to the licenses of the
generating models (Qwen2.5-3B, phi-2, SmolLM2-1.7B, pythia-2.8b).

## Citation
```bibtex
@misc{hsu2027curvature,
  title  = {Estimating Self-Scoring AI-Text Detectors' Failure Temperatures from Human Text},
  author = {Hsu, Yun-Chin and Yang, Te-Lun},
  year   = {2027},
  note   = {Code: https://github.com/hsuiris/curvature-margin}
}
```
