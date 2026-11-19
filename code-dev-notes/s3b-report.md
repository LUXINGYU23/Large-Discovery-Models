# S3b report

- `tasks/copper_inversion/experiments/v2/parallel_runtime.py:285` adds the policy-verified `rsi-q1` entry point, installing the shared queue and startup limit. Recreate the policy against the changed runtime hash before use; preserve and independently verify the existing load-qualification binding.
- `tasks/copper_inversion/experiments/rsi/q1.py:88` dispatches up to `batch_workers` through `ParallelService`, with tier-specific case concurrency. Layout identities are written before dispatch, completed receipts are replayed, and mappings are checked on rerun. Reference validation uses the same queue and concurrent program limit. REF unseal is unchanged. For Q1-only suites the S-tier case limit is used; pass `--tier S` for reference validation.
- `tasks/copper_inversion/experiments/rsi/isolation_evidence.py:17` gathers the four actual input records and live account, directory and access checks. It records input hashes, failures and limited scope. It does not establish an adversarial attack suite. Root must provide all private roots and socket directories that are in scope, and a real private release file; this tool cannot discover omitted roots.
- `tasks/copper_inversion/tests/test_rsi_q1.py` and `test_rsi_isolation_evidence.py` exercise a fake scheduler comparison/replay and evidence failures. Focused H20 test: `python3 /home/lu/workspaces/rsi4geo-paper-review/code-dev-notes/h20sync.py test /home/lu/workspaces/rsi4geo-paper-review/wt/s3b s3b tasks/copper_inversion/tests/test_rsi_q1.py tasks/copper_inversion/tests/test_rsi_isolation_evidence.py -q`: 6 passed. The fake tests do not prove actual scheduling throughput or host isolation.

Estimated Q1 solver allocation: `(60*(12*30 + 24*120 + 24*30) + 12*12*3*30)/3600 = 69.6` container-hours. At G=32 the ideal work-conservation lower bound is 2.175 hours. It excludes interface checks, startup, queueing, stragglers, tier/batch limits, and references; actual elapsed time will be longer.

## Coordinator commands

Paths/variables below must refer to the real deployment and evidence; do not substitute invented records. Use the current checkout and `$SERVICE`, `$R`, `$FORMAL`, `$IMAGE`, `$Q1_WORK`, `$REF_WORK` and `refs.json` from the runbook. Rebuild the policy as root with the existing qualification in `$VALUES` and the actual queue (`$QUEUE`):

```sh
# [root]
"$SERVICE" -m tasks.copper_inversion.experiments.rsi.launch policy --values "$VALUES" --queue-root "$QUEUE" --out "$POLICY"
# [991]
"$SERVICE" -m tasks.copper_inversion.experiments.v2.parallel_runtime --policy "$POLICY" --tier S rsi-q1 evaluate --selection "$Q1_WORK/q1-selection.json" --release "$FORMAL" --output "$Q1_WORK/evaluation" --image "$IMAGE"
"$SERVICE" -m tasks.copper_inversion.experiments.v2.parallel_runtime --policy "$POLICY" --tier S rsi-q1 references --references "$REFS" --release "$FORMAL" --output "$REF_WORK" --image "$IMAGE"
# [root] real evidence paths and actual client group; repeat --private-root and --socket-dir as needed
"$SERVICE" -m tasks.copper_inversion.experiments.rsi.isolation_evidence --root "$R" --guest "$GUEST_PROBE" --audit "$ACCESS_AUDIT" --rpc "$RPC_PROBE" --positive "$AT04_CONTROL" --private-root "$PRIVATE_ROOT" --release-file "$PRIVATE_RELEASE_FILE" --socket-dir "$SOCKET_DIR" --client zsgpu --client-group "$CLIENT_GROUP" --operator "$OPERATOR" --output "$ISOLATION"
```

If the access audit is absent, the provisioning tool can produce it as root: `"$SERVICE" -m tasks.copper_inversion.experiments.v2.setup_evaluator_account`. Review its fixed legacy base and existing account requirements before running: it creates/changes permissions there. Guest, RPC and positive-control inputs must be collected independently using the actual deployment; the RPC probe performs real evaluations and requires a fresh two-slot development service. The assembler does not generate those records. Root must inspect the evidence and failed checks before using the output for release.

The H20 rehearsal explicitly forbids Docker and running service/probes as other accounts, so actual host isolation and real evaluator throughput remain H20 acceptance-phase work.

Full copper suite: **not completed**. An attempted run with an autouse `execution.docker` guard was interrupted after inspection found modules that import Docker by alias, so that guard could not guarantee the rehearsal's no-Docker rule. No full-suite pass is claimed. The coordinator must run the unguarded full suite only when Docker testing is authorized, or provide a test-level guard that covers all Docker call paths. The `h20sync.py test` helper pipes pytest through `tail` without `pipefail`; inspect the pytest summary, not merely its process exit code.
