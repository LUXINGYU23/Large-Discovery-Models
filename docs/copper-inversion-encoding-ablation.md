# Copper code encoding ablation — 2026-10-23

## Fixed design

Six independent campaigns: two 3D cases × AST-only / Qwen-only / equal fusion.
Seed 47; identical initial geometry; 20 research rounds, plus one baseline.
Each shared Coverage 2+2 panel has four researchers, each proposing four standalone
solver files. All 16 distinct byte identities pass the global static barrier and
fixed read-only interface fixtures before BO selects four for independent scoring.
Turn messages deliver every new measurement as a compact summary; the complete
measured code, slices and annotations remain in the authoritative exported
history.json, available for targeted reading. This avoids injecting all historical
source into an old branch when it rejoins after many rounds.
Each campaign has 81 formal evaluation attempts and up to 320 proposed algorithms.
Failures retain their status and do not acquire invented losses. This is a
single-seed engineering comparison, not a multi-seed significance claim.

Reuse the existing shared engine, Coverage tree, exact RBF GP and LCB. No compiled
policy is added in this encoding ablation. Compared with the referenced NucleoBench
ablation (20 rounds, 64 proposal occurrences / up to 48 unique / select 16), this
uses 16 complete programs / select 4 to account for interface and solver cost.

All researcher/main/provider requests use official https://api.deepseek.com,
deepseek-flash (V4.1 Flash), low, via Pi Responses. Each research turn has 1200s,
8 bash calls each at most 60s, no web tools. Solver remains 2 CPU / 4 GiB / 120s.
No query_surrogate is exposed. All scientific tests and runs are remote on H20.

## Representation

Parse original source with Python AST and unparse to normalized code. Preserve
identifiers, numeric/string constants, docstrings and statement order; remove
ordinary comments/formatting. This representation is not claimed to establish
semantic equivalence. Execute only the original frozen source.

AST: deterministic signed SHA256 counts of node types and scalar fields, 256D,
unit normalized. Qwen: frozen Qwen/Qwen3-Embedding-0.6B revision
97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3, 1024D, Sentence Transformers local-only,
float32 inference, native last-token pooling, no prompt, unit normalized. Reject
inputs beyond 32768 tokens before inference; never silently truncate or substitute
AST. Weights/config/tokenizer are verified against cached provenance on load.
Fusion: concat(sqrt(0.5)*AST, sqrt(0.5)*Qwen), 1280D. No PCA or fitted projection.
All three arms retain the same GP lengthscale 1.5, noise 1e-4, LCB beta 1,
feature/target scaling floors 1; no extra per-coordinate whitening after fusion.
Each round fits only independently measured historical losses, never self-reports.

## Evidence and fairness

Persist every accepted source, proposal lineage, normalized source, feature vector,
component vectors where computed, code/model/version digests, model load and encode
time, complete pool predictions and acquisition scores, selected IDs, interface
receipts, independent score/resource records, raw Pi/provider traces and events.
Unselected candidates have features/predictions but no ground-truth score. Offline
counterfactual evaluation would be a separate run, not online training feedback.

Report best loss by round and formal evaluation count, end-to-end wall time,
research/embedding/model load/solver/scorer time, provider usage, failed attempts,
and selected-only evaluation compliance. This comparison controls campaign budgets,
not total internal calls to public loss or identical LLM proposals. Embedding timing
includes cold model loading; each campaign has independent history and local cache.
Reuse server model cache read-only; no downloads or changes to other experiments.

## Execution gates

Run registration/dependency/config gates, remote unit and real isolation tests,
then six one-round native preflights with the same 16/4 batch. Only after inspecting
each preflight's actual submissions, model/effort, GP selection and independent
scores may all six new 20-round campaigns launch. Preserve source identity and all
failed attempts; never change implementation in an active campaign or force-resume
under a different identity. Archive final runs under the authorized Ceph archive.
