---
name: compile_ldm_policy
description: Compile a measured-feedback residual GP prior and LDM acquisition weights.
---

Inspect the active policy contract first. Both prior_mean@1 and ldm_weights@1 are enabled. Read the current question with get_public_task and inspect same-question drafts with get_public_history; request detailed CIFs by draft ID. Only past measured scalar correctness is evidence. Unmeasured current drafts have no score. Never obtain private targets, judge internals, answer tables or other questions' observations.

The host fixes the public geometry encoder, exact residual RBF GP, UCB, empirical proposal frequencies, evaluation count and Gumbel sampling. Geometry features are signed log1p public input-to-draft summaries; availability flags distinguish missing parsing. Do not replace the kernel or evaluator. History utilities are actual binary correctness labels. Return a standardized prior mean, not a probability, with one finite value per query row. The host subtracts history priors before GP fitting and adds query priors afterward; covariance and standardization remain fixed by measured history.

Write optimization_policy.py with POLICY_API_VERSION = 1, CAPABILITIES = {"prior_mean": 1, "ldm_weights": 1}, compute_prior_mean(history_features, history_utilities, query_features, context), and choose_ldm_weights(context). The latter returns {"stage": "evidence_based_stage", "alpha": finite_nonnegative_float, "eta": finite_nonnegative_float}. Selection weights are proportional to q0**alpha * exp(eta * robust_z(UCB)); they are not an argmax rule. Inspect the supplied weight context and prediction feedback before adapting weights. Support empty query arrays and preserve row order.

Use inspect_policy_contract, validate_policy_draft and evaluate_policy_draft. Diagnostics use chronological whole-round holdouts and compare baseline versus residual-prior GP predictions; current-pool probability changes do not establish unseen correctness. Keep artifact code pure, deterministic NumPy with no IO, network or external dependencies. Submit action="replace", artifact_path="optimization_policy.py" through submit_optimization_policy. Use keep or disable when justified; disable means zero prior plus declared default weights, not a different optimizer.
