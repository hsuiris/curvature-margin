# Result tables

Each folder holds the tables written by the script of the same name in `scripts/`
(for example `e3_falcon_dual/` is written by `scripts/e3_falcon_dual.py`).

| Table | Rows |
| --- | --- |
| `a01_mage_families/groups.csv` | 5 |
| `a01_mage_families/pairs.csv` | 4 |
| `a01_mage_families/per_generator.csv` | 27 |
| `a02_decoding_alignment/contrasts.csv` | 5 |
| `a02_decoding_alignment/metrics.csv` | 6 |
| `a02_decoding_alignment/two_sided.csv` | 6 |
| `a03_scorer_family/decomposition.csv` | 6 |
| `a03_scorer_family/metrics.csv` | 12 |
| `a04_temperature_sweep/crossing.csv` | 3 |
| `a04_temperature_sweep/metrics.csv` | 18 |
| `a04_temperature_sweep/two_sided.csv` | 6 |
| `a05_blindspot_features/metrics.csv` | 60 |
| `a05_blindspot_features/single_feature.csv` | 14 |
| `a06_gate_stack/auroc_Qwen2.5-3B.csv` | 5 |
| `a06_gate_stack/auroc_gpt2-xl.csv` | 5 |
| `a06_gate_stack/auroc_gpt2.csv` | 5 |
| `a06_gate_stack/tpr_Qwen2.5-3B.csv` | 5 |
| `a06_gate_stack/tpr_gpt2-xl.csv` | 5 |
| `a06_gate_stack/tpr_gpt2.csv` | 5 |
| `a07_raid/metrics.csv` | 68 |
| `a08_raid_compare/compare.csv` | 68 |
| `a09_signal_ablation/ablation_Qwen2.5-3B.csv` | 25 |
| `a09_signal_ablation/ablation_gpt2-xl.csv` | 25 |
| `a09_signal_ablation/worst_Qwen2.5-3B.csv` | 8 |
| `a09_signal_ablation/worst_gpt2-xl.csv` | 8 |
| `a10_temptest/temptest.csv` | 6 |
| `a11_new_generators/crossing.csv` | 4 |
| `a11_new_generators/metrics.csv` | 24 |
| `a11_new_generators/transfer.csv` | 10 |
| `a12_paraphrase/metrics.csv` | 16 |
| `a12_paraphrase/transfer.csv` | 8 |
| `e10_blind_spot_combination/development.csv` | 10 |
| `e10_blind_spot_combination/test.csv` | 4 |
| `e10_blind_spot_combination/test_curves.csv` | 40 |
| `e11_retokenization_diagnostic/per_model.csv` | 4 |
| `e11_retokenization_diagnostic/summary.csv` | 1 |
| `e1_scaled_sweep/crossing.csv` | 6 |
| `e1_scaled_sweep/domains.csv` | 21 |
| `e1_scaled_sweep/metrics.csv` | 54 |
| `e2_curvature_margin/crossing_curvature.csv` | 6 |
| `e3_falcon_dual/crossing.csv` | 3 |
| `e3_falcon_dual/domains.csv` | 21 |
| `e3_falcon_dual/metrics.csv` | 27 |
| `e4_m4gt_natural/cells.csv` | 35 |
| `e4_m4gt_natural/models.csv` | 9 |
| `e4_m4gt_natural/summary.csv` | 7 |
| `e5_heldout_pythia/crossing.csv` | 1 |
| `e5_heldout_pythia/domains.csv` | 6 |
| `e5_heldout_pythia/metrics.csv` | 9 |
| `e6_zero_shot_benchmark/crossing.csv` | 32 |
| `e6_zero_shot_benchmark/domain_crossing.csv` | 216 |
| `e6_zero_shot_benchmark/domains.csv` | 1944 |
| `e6_zero_shot_benchmark/metrics.csv` | 288 |
| `e7_trained_baselines/crossing.csv` | 26 |
| `e7_trained_baselines/domain_crossing.csv` | 177 |
| `e7_trained_baselines/domains.csv` | 1593 |
| `e7_trained_baselines/m4gt.csv` | 210 |
| `e7_trained_baselines/metrics.csv` | 234 |
| `e8_margin_generality/extrapolation.csv` | 20 |
| `e8_margin_generality/half_split.csv` | 200 |
| `e8_margin_generality/half_split_summary.csv` | 4 |
| `e8_margin_generality/own_margin.csv` | 32 |
| `e8_margin_generality/same_data.csv` | 10 |
| `e9_theory_prediction/development.csv` | 3 |
| `e9_theory_prediction/test.csv` | 4 |
| `e9_theory_prediction/test_corrected.csv` | 4 |
| `e9_theory_prediction/test_diagnostics.csv` | 4 |
| `e9_theory_prediction/test_summary.csv` | 1 |

## Column reference

Suffix `_FDG` = Fast-DetectGPT, `_Bino` = Binoculars. `fail_temp` is the failure temperature, the sampling temperature
at which AUROC crosses 0.5 (linear interpolation). `TPR_1pct` is the true-positive rate at a 1% false-positive rate. Yes/no columns hold
`yes` or `no`. Wide tables use one column per temperature (`T0.6` ... `T1.2`); `TempTest_AUROC_tau<t>` and
`share_AI_judged_human_tau<t>` are TempTest's AUROC and the share of machine texts it labels human at its temperature `t`;
`two_sided_<scorer>` is the two-sided Fast-DetectGPT AUROC under that scorer.

| Column | Meaning |
| --- | --- |
| `group` | group of texts or generators |
| `n_ai` | number of machine texts |
| `median_tokens` | median token count |
| `Fast-DetectGPT` | Fast-DetectGPT AUROC |
| `Fast-DetectGPT CI` | 95% bootstrap CI of the Fast-DetectGPT AUROC |
| `Binoculars` | Binoculars AUROC |
| `Binoculars CI` | 95% bootstrap CI of the Binoculars AUROC |
| `contrast` | models compared (instruction-tuned vs base of the same family and size) |
| `score` | detector used for the comparison |
| `base` | domain-matched AUROC of the base model |
| `instruct` | domain-matched AUROC of the instruction-tuned model |
| `diff` | difference between the two compared values |
| `CI` | 95% bootstrap confidence interval, written as [low, high] |
| `CI_excludes_0` | CI excludes 0 |
| `generator` | model that generated the machine texts |
| `FDG` | Fast-DetectGPT AUROC |
| `comparison` | the two conditions compared |
| `CI_low` | 95% CI lower |
| `CI_high` | 95% CI upper |
| `decoding` | decoding setting (pure sampling, top-p, or a sampling temperature) |
| `n_texts` | number of texts |
| `AUROC` | AUROC, machine texts as the positive class |
| `MAGE_reference` | MAGE reference value |
| `surprisal_human` | mean human surprisal |
| `surprisal_ai` | mean surprisal of machine texts |
| `fdg_human` | mean Fast-DetectGPT score of human texts |
| `fdg_ai` | mean Fast-DetectGPT score of machine texts |
| `median_tokens_human` | median human token count |
| `median_tokens_ai` | median machine token count |
| `AUROC_two_sided` | two-sided AUROC |
| `effective_entropy` | effective entropy H(g) |
| `family_increase` | increase due to scorer family (KL term) |
| `surprisal_margin` | surprisal margin M: mean human surprisal minus mean machine surprisal |
| `AUROC_GPT2XL_scorer` | AUROC with GPT-2 XL scorer |
| `scorer` | model that computes the per-token scores |
| `AUROC_predicted` | forward-predicted AUROC Phi(M/sigma) |
| `error` | predicted minus observed AUROC |
| `scorer_entropy` | scorer entropy |
| `score_ai` | mean machine score |
| `score_human` | mean human score |
| `fail_temp_predicted` | prediction 1 - E_h_c / V_human (human text only) |
| `fail_temp_observed` | observed failure temperature |
| `AUROC_monotone_decreasing` | AUROC decreases monotonically with temperature |
| `temperature` | sampling temperature |
| `two_sided_max_of_three` | two-sided combination (max of three scorers) |
| `best_single_one_sided` | best one-sided AUROC of a single scorer at this temperature |
| `best_single_two_sided` | best two-sided AUROC of a single scorer at this temperature |
| `combination_minus_best_single` | combination minus best single two-sided |
| `evaluation` | evaluation setting (within_blind_spot or cross_temperature) |
| `method` | detection method |
| `Unnamed: 0` | method (row label) |
| `0` | AUROC of the single feature |
| `min` | minimum over temperatures |
| `category` | RAID generator category (base or aligned_or_commercial) |
| `repetition_penalty` | repetition penalty |
| `AUROC_one_sided` | one-sided AUROC |
| `sign_agrees` | sign of margin agrees with side of 0.5 |
| `within_blind_spot` | cross-validation within T = 1.0 |
| `cross_temperature` | trained on the other temperatures |
| `gen_temperature` | generation temperature |
| `AUROC_FDG_numerator` | AUROC of the Fast-DetectGPT numerator |
| `monotone_decreasing` | AUROC decreases monotonically |
| `two_sided` | two-sided Fast-DetectGPT score (distance from the human median) |
| `feature_classifier` | feature classifier |
| `stack` | stacked classifier on features and scores |
| `gate` | gate: two-sided score for texts far from the human median, feature classifier otherwise |
| `setting` | generator / scorer configuration |
| `lowest_T` | lowest temperature of the grid |
| `min_FDG one-sided` | lowest AUROC over temperatures, one-sided Fast-DetectGPT |
| `min_FDG two-sided` | the same, two-sided Fast-DetectGPT |
| `min_Likelihood two-sided` | the same, two-sided Likelihood |
| `min_max two-sided` | the same, maximum of the two two-sided scores |
| `gain` | lowest AUROC of the combination minus that of the better single two-sided score |
| `loss_at_lowest_T` | one-sided Fast-DetectGPT minus the combination at the lowest temperature |
| `criterion1` | gain >= 0.05 |
| `criterion2` | loss at the lowest temperature <= 0.03 |
| `role` | prospective (registered before the text existed) or supplementary |
| `T` | sampling temperature |
| `FDG one-sided` | AUROC, one-sided Fast-DetectGPT |
| `FDG two-sided` | AUROC, two-sided Fast-DetectGPT |
| `Likelihood two-sided` | AUROC, two-sided Likelihood |
| `max two-sided` | AUROC, maximum of the two two-sided scores |
| `max_CI_half_width_near_crossing` | largest AUROC CI half-width near the crossing |
| `domain` | text domain |
| `n_ai_per_temperature` | machine texts per temperature |
| `CI_half_width` | CI half-width |
| `TPR_1pct` | TPR at 1% FPR (threshold from held-out human texts) |
| `FPR_realized` | realized FPR on held-out human texts |
| `AUROC_predicted_pilot_sigma` | forward-predicted AUROC with the pilot sigma |
| `surprisal_margin_pred` | failure temperature predicted by M |
| `surprisal_margin_error` | observed minus M-predicted failure temperature |
| `curvature_margin_pred` | failure temperature predicted by K |
| `curvature_margin_error` | observed minus K-predicted failure temperature |
| `entropy_gap_near_crossing` | human minus machine mean entropy near the crossing (context term) |
| `fail_temp_observed_FDG` | observed failure temperature, Fast-DetectGPT |
| `FDG_CI_low` | Fast-DetectGPT AUROC CI lower |
| `FDG_CI_high` | Fast-DetectGPT AUROC CI upper |
| `fail_temp_observed_Bino` | observed failure temperature, Binoculars |
| `fail_temp_diff_FDG_Bino` | absolute difference between the Fast-DetectGPT and Binoculars failure temperatures |
| `fail_temp_GPT2XL_e1` | GPT-2 XL failure temperature from e1 |
| `monotone_decreasing_FDG` | Fast-DetectGPT AUROC decreases monotonically with temperature |
| `monotone_decreasing_Bino` | Binoculars AUROC decreases monotonically with temperature |
| `max_CI_half_width_near_crossing_FDG` | largest Fast-DetectGPT CI half-width near the crossing |
| `curvature_margin_self_pred` | failure temperature predicted by own-entropy K |
| `curvature_margin_self_error` | observed minus own-entropy-K-predicted failure temperature |
| `curvature_margin_dual_pred` | failure temperature predicted by dual K |
| `curvature_margin_dual_error` | observed minus dual-K-predicted failure temperature |
| `n_human` | number of human texts |
| `curvature_margin_self` | curvature margin with the scorer's own entropy |
| `curvature_margin_dual` | curvature margin K with the reference model's cross-entropy (dual) |
| `AUROC_FDG` | AUROC of Fast-DetectGPT |
| `FDG_CI_half_width` | Fast-DetectGPT AUROC CI half-width |
| `TPR_1pct_FDG` | TPR at 1% FPR, Fast-DetectGPT |
| `FDG_TPR_CI_low` | Fast-DetectGPT TPR CI lower |
| `FDG_TPR_CI_high` | Fast-DetectGPT TPR CI upper |
| `FPR_realized_FDG` | realized FPR, Fast-DetectGPT |
| `TPR_1pct_same_data_FDG` | TPR at 1% FPR with the threshold set on all human texts, Fast-DetectGPT |
| `AUROC_Bino` | AUROC of Binoculars |
| `Bino_CI_low` | Binoculars AUROC CI lower |
| `Bino_CI_high` | Binoculars AUROC CI upper |
| `Bino_CI_half_width` | Binoculars AUROC CI half-width |
| `TPR_1pct_Bino` | TPR at 1% FPR, Binoculars |
| `Bino_TPR_CI_low` | Binoculars TPR CI lower |
| `Bino_TPR_CI_high` | Binoculars TPR CI upper |
| `FPR_realized_Bino` | realized FPR, Binoculars |
| `TPR_1pct_same_data_Bino` | the same for Binoculars |
| `TPR_1pct_GPT2XL_e1` | TPR at 1% FPR, GPT-2 XL (from e1) |
| `AUROC_GPT2XL_e1` | AUROC with the GPT-2 XL scorer, taken from e1_scaled_sweep |
| `curvature_margin` | curvature margin K: machine minus human mean of (entropy - surprisal) |
| `mean_tokens_ai` | mean machine token count |
| `mean_tokens_human` | mean human token count |
| `sign_agrees_curvature` | sign of K agrees with the side of 0.5 |
| `sign_agrees_surprisal` | sign of M agrees with the side of 0.5 |
| `n_domains` | number of domains |
| `AUROC_domain_matched_FDG` | domain-matched AUROC, Fast-DetectGPT |
| `AUROC_domain_matched_Bino` | domain-matched AUROC, Binoculars |
| `AUROC_lowest_domain` | AUROC in the lowest domain |
| `lowest_domain` | domain with the lowest AUROC |
| `scope` | scope (overall / domain / domain:generator) |
| `FPR_realized_split_FDG` | realized FPR with the split-calibrated threshold, Fast-DetectGPT |
| `TPR_split_FDG` | TPR with the split-calibrated threshold, Fast-DetectGPT |
| `FPR_realized_split_Bino` | the same for Binoculars |
| `TPR_split_Bino` | the same for Binoculars |
| `n_human_calibration` | human texts used to set the threshold |
| `n_human_evaluation` | human texts used to measure realized FPR |
| `dataset` | test set, with the study labels: main1 / main2a = MAGE continuations (06 / 07), extA = held-out pythia (09), m4gt = M4GT-Bench (08) |
| `fail_temp` | failure temperature (AUROC = 0.5) |
| `share_bootstrap_crossing` | share of bootstrap curves that cross 0.5 |
| `AUROC_lowest_temp` | AUROC at the lowest temperature |
| `AUROC_highest_temp` | AUROC at the highest temperature |
| `TPR_CI_low` | TPR CI lower |
| `TPR_CI_high` | TPR CI upper |
| `detector` | detector |
| `AUROC_at_T1_0` | AUROC at T = 1.0 |
| `TPR_1pct_same_data` | TPR at 1% FPR, threshold set on all human texts |
| `TPR_split` | TPR with the split-calibrated threshold |
| `FPR_realized_split` | realized FPR with the split-calibrated threshold |
| `cutoff` | highest temperature used for the extrapolation |
| `n_temperatures` | temperatures used |
| `K_error` | observed crossing minus K's zero |
| `mean_score_error` | observed crossing minus the mean-score zero (e8 part C: of its linear extrapolation) |
| `AUROC_probit_error` | observed crossing minus the zero of a linear fit to probit(AUROC) |
| `K` | mean absolute error of K's zero over the ten settings, one row per half split |
| `mean_score` | the same for the mean score difference |
| `median_score` | the same for the median score difference |
| `AUROC_half` | the same for the AUROC crossing estimated on the other half |
| `predictor` | predictor of the failure temperature |
| `mean` | mean over splits |
| `25%` | first quartile over splits |
| `50%` | median over splits |
| `75%` | third quartile over splits |
| `share_K_better` | share of splits in which K has the lower error |
| `statistic` | zero-shot statistic |
| `own_margin_zero` | temperature where the statistic's own mean score difference crosses 0 |
| `own_margin_error` | observed crossing minus the own-margin zero |
| `K_zero` | temperature where the curvature margin K crosses 0 |
| `mean_score_zero` | temperature where the mean detector-score difference (machine minus human) crosses 0 |
| `median_score_zero` | temperature where the median detector-score difference crosses 0 |
| `median_score_error` | observed crossing minus the median-score zero |
| `K_sign_disagreements` | temperatures where the sign of K disagrees with the side of AUROC 0.5 |
| `mean_score_sign_disagreements` | the same for the mean score difference |
| `E_h_c` | mean over human texts of mean(H - s) under the model itself |
| `V_human` | mean over human texts of the per-position variance of log p |
| `fail_temp_CI_low` | 95% bootstrap CI of the observed crossing, lower |
| `fail_temp_CI_high` | 95% bootstrap CI of the observed crossing, upper |
| `crossing_share` | share of bootstrap curves that cross 0.5 |
| `n_humans` | number of human texts |
| `temperatures` | number of temperatures |
| `fail_temp_error` | observed minus predicted failure temperature |
| `error_guess_T1` | observed failure temperature minus 1 (baseline: guess T* = 1) |
| `error_dev_mean` | observed failure temperature minus 1.011 (baseline: development mean) |
| `prediction_written_at_utc` | time the prediction was written, before any generation |
| `delta` | banned-token correction: mean of H - E_p'[s] on human contexts (design doc 8.8.20) |
| `fail_temp_predicted_corrected` | prediction 1 + (delta - E_h_c) / V_restricted |
| `fail_temp_error_corrected` | observed minus corrected prediction |
| `corrected_written_at_utc` | time the corrected prediction was written |
| `machine_c_at_T1` | mean(H - s) of machine texts at T = 1 (zero in theory) |
| `machine_V_at_T1` | mean Var[log p] of machine texts at T = 1 |
| `human_c` | mean(H - s) of human texts |
| `human_V` | mean Var[log p] of human texts |
| `n_models` | models tested |
| `n_determined` | models whose crossing lies in the grid |
| `MAE_theory` | mean absolute error of the prediction |
| `MAE_guess_T1` | mean absolute error of guessing T* = 1 |
| `MAE_dev_mean` | mean absolute error of guessing 1.011 |
| `criterion1_accuracy` | pre-registered criterion 1 met |
| `criterion2_advantage_over_baselines` | pre-registered criterion 2 met |
| `MAE_corrected` | mean absolute error of the corrected prediction |
| `corrected_criterion1` | criterion 1 met by the corrected prediction |
| `corrected_criterion2` | criterion 2 met by the corrected prediction |
