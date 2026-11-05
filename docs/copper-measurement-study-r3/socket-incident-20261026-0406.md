# 04:06 monitoring incident

Formal training is paused. At 03:44, MCMC3407 hit the private evaluator client's fixed 7200-second socket read timeout. Supervisor 3919838 then paused dispatch; all nine clients have exited. Existing private services remain available for reconciliation.

The private evaluator produced a receipt at 03:43:17 and later encountered BrokenPipe while responding. There are no newly failed candidate receipts in this interval. This is not a new Docker start timeout or a recurrence of the SIGALRM failure.

The current 88 solver/scorer slots can require more than two hours for queued suites: 144 simultaneous sources × 48 cases × 120 seconds / 88 slots is about 9425 seconds before overhead. This capacity estimate does not establish the exact queue history of the failed request.

Next repair: separate long-running evaluation state from the blocking socket lifetime, with bounded status/liveness checks, authenticated peers and idempotent source identity. Validate progress, stalls, disconnect recovery, budget preservation, private closure and full concurrent load before continuation. Do not simply keep extending the socket deadline. The 120-second solver limit and frozen scientific identity remain unchanged.

Evidence: socket-incident-20261026-0406.json lists ten unfinished jobs without secret resource keys. Inspect their containers and preserve all ledger/receipt/result hashes before taking over. Do not rerun any previous controller or blindly remove either pause marker. The transport repair has not yet been implemented or qualified.

The last verified common comparison remains 179 charged sources. Formal validation and test have not begun.
