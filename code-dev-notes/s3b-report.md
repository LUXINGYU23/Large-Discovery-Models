# S3b report

- `tasks/copper_inversion/experiments/v2/parallel_runtime.py:285` adds the policy-verified `rsi-q1` entry point, installing the shared queue and startup limit. Recreate the policy against the changed runtime hash before use; preserve and independently verify the existing load-qualification binding.
- `tasks/copper_inversion/experiments/rsi/q1.py:88` dispatches up to `batch_workers` through `ParallelService`, with tier-specific case concurrency. Layout identities are written before dispatch, completed receipts are replayed, and mappings are checked on rerun. Reference validation uses the same queue and concurrent program limit. REF unseal is unchanged. For Q1-only suites the S-tier case limit is used; pass `--tier S` for reference validation.
- `tasks/copper_inversion/experiments/rsi/isolation_evidence.py` gathers the four actual input records and live account, directory and access checks. It records input hashes, failures and limited scope. It does not establish an adversarial attack suite. Root must provide all private roots and socket directories that are in scope, and a real private release file; this tool cannot discover omitted roots.
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
```

## S3b-2: deployment evidence probes

`tasks/copper_inversion/experiments/rsi/access_audit.py` creates or verifies the project-local uid-991 canary and probes both it and the private release file as the evaluator and client. The access record is bound to the supplied project root, client and release file. The assembler accepts the actual A2 positive-control format and checks every field of the new risk-only RPC record, including its measured slot cap. `tasks/copper_inversion/tests/rsi_rpc_probe.py` tests the public contract and source whitelist with a fresh service. All RPC evaluations are real, consume the service's entire slot allowance, and must use a disposable development service; its evidence is not an adversarial attack suite. The probe records only check results and paths, not private feedback values.

Coordinator: set `C` to the actual checkout and export `PYTHONPATH="$C"`; set `$R`, `$SERVICE`, `$IMAGE`, `$CLIENT_GROUP`, `$OPERATOR`, `$GUEST_PROBE`, `$AT04_CONTROL`, `$ISOLATION`, and the following paths to the real deployment. `$ACCESS_AUDIT` must be a new root-writable path. `$PRIVATE_RELEASE_FILE` must be a real file under `$R/private` readable by uid 991 and inaccessible to zsgpu. `$TRAIN_MANIFEST` and `$SMOKE_RELEASE_DIR` must identify the smoke development release; its tier S must actually specify 10 seconds and 13 slots. `$RPC_PRIVATE` must be a new private service directory, `$RPC_SOCKET_DIR` a new socket directory, and `$RPC_PROBE` a new file in a directory writable by zsgpu and readable by root. Collect the guest and A2 records separately; do not invent them.

```sh
# [root] project-specific access evidence; does not use /Aidata/shared/ldm
"$SERVICE" -m tasks.copper_inversion.experiments.rsi.access_audit --root "$R" --client zsgpu --release-file "$PRIVATE_RELEASE_FILE" --output "$ACCESS_AUDIT"
# [root] start a fresh disposable smoke service; preserve its deployment.json
"$SERVICE" -m tasks.copper_inversion.rsi.launch_service --manifest "$TRAIN_MANIFEST" --private-root "$RPC_PRIVATE" --socket-dir "$RPC_SOCKET_DIR" --service-user ldm-copper-eval --caller-uid "$(id -u zsgpu)" --caller-gid "$(getent group "$CLIENT_GROUP" | cut -d: -f3)" --image "$IMAGE" --seconds 10 --slots 13 --tier S
# [root] run the socket client as zsgpu; this performs real evaluations and exhausts all 13 slots
runuser -u zsgpu -- env PYTHONPATH="$C" "$SERVICE" -m tasks.copper_inversion.tests.rsi_rpc_probe --socket "$RPC_SOCKET_DIR/e.sock" --private-root "$RPC_PRIVATE" --release-dir "$SMOKE_RELEASE_DIR" --output "$RPC_PROBE"
# [root] assemble and review; repeat private-root/socket-dir options for every covered directory
"$SERVICE" -m tasks.copper_inversion.experiments.rsi.isolation_evidence --root "$R" --guest "$GUEST_PROBE" --audit "$ACCESS_AUDIT" --rpc "$RPC_PROBE" --positive "$AT04_CONTROL" --private-root "$PRIVATE_ROOT" --private-root "$RPC_PRIVATE" --release-file "$PRIVATE_RELEASE_FILE" --socket-dir "$SOCKET_DIR" --socket-dir "$RPC_SOCKET_DIR" --client zsgpu --client-group "$CLIENT_GROUP" --operator "$OPERATOR" --output "$ISOLATION"
```

The release assembler still checks only the aggregate status; root must inspect the source records, paths, scopes and each check before using it. The legacy `setup_evaluator_account` writes outside this project and must not be used here. The fresh smoke service should be stopped and its resources reconciled by the operator after preserving evidence.

Focused H20 tests: `python3 /home/lu/workspaces/rsi4geo-paper-review/code-dev-notes/h20sync.py test /home/lu/workspaces/rsi4geo-paper-review/wt/s3b s3b tasks/copper_inversion/tests/test_rsi_isolation_evidence.py tasks/copper_inversion/tests/test_rsi_evidence_probes.py -q` — 11 passed. Fakes cover positive/negative predicates, project access checks, RPC replay, budget, whitelist and probe failures. Live probes have not been run. The coordinator runs the full suite.

The H20 rehearsal explicitly forbids Docker and running service/probes as other accounts, so actual host isolation and real evaluator throughput remain H20 acceptance-phase work.

Full copper suite: **not completed**. An attempted run with an autouse `execution.docker` guard was interrupted after inspection found modules that import Docker by alias, so that guard could not guarantee the rehearsal's no-Docker rule. No full-suite pass is claimed. The coordinator must run the unguarded full suite only when Docker testing is authorized, or provide a test-level guard that covers all Docker call paths. The `h20sync.py test` helper pipes pytest through `tail` without `pipefail`; inspect the pytest summary, not merely its process exit code.
