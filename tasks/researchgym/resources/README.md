# ResearchGym Task Resources

| Path | Contents |
| --- | --- |
| `upstream_contract.json` | ResearchGym source URL, pinned commit `0adc08e9`, and SHA-256 of every tracked file in the four case directories. Regenerate only with `scripts/pin_upstream.py`, which reads Git objects at the commit. |
| `cases/catalog.json` | Declarative case definitions: entry symbol and candidate file, slot injection anchors, job matrix, official grader command, complete coverage identity, metric, protocol identifier and deviations, and the public case contract shown to research sessions. |
| `cases/seeds/` | Standalone restatements of each case's released baseline (integrated gradients, SimpleCIL, Tent, uniform replay). Used for `shared_start` and as the direct-request interface reference. |
| `cases/support/improving_replay_buffers/rl_runner.py` | Task-local SAC training stack. The pinned upstream task ships no training loop; this file is fixed evaluator code, not search space. |
| `harness/profiles/` | `program_research` (template for every independent proposal session) and `policy_architect` `AGENTS.md`. |
| `harness/skills/` | Task-authored runtime Skills: `researchgym-method-research` (with `scripts/build_candidates.py`), `ml-experiment-analysis`, and `compile-ldm-policy`. |
| `harness/tools/` | `researchgym_tools.mjs` (case contract, measured history, candidate checker), `program_rules.py` (the admission rules imported unchanged by the host validator), and `check_candidates.py`. |
| `harness/policy_diagnostics.py` | Trusted draft diagnostics for the policy MCP (`evaluate_policy_draft`). |
| `harness/image/` | Guest recipe: Python 3.11 with NumPy, SciPy, pandas, scikit-learn, matplotlib and CPU PyTorch; `smoke.sh` exercises each promised capability. |
| `qualification_evidence.json` | Gate evidence; currently `mock_verified`. |

All Skills are task-authored; `compile-ldm-policy` is adapted from this
repository's ReaSyn task. No third-party Skill text is vendored.
