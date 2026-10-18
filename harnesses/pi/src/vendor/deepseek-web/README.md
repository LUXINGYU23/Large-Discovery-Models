# DeepSeek Harness web tool source

Source: [deepseek-ai/deepseek-harness](https://github.com/deepseek-ai/deepseek-harness/tree/687f282e8625c6b8d11ff58211a90524be2e4317), revision `687f282e8625c6b8d11ff58211a90524be2e4317`. Copyright (c) 2026 DeepSeek, MIT (see LICENSE).

`contract.ts` adapts algorithms from:

- `packages/web/tool-web/src/search.ts`: query validation, source formatting, concurrent batching and round-robin merging.
- `packages/web/tool-web/src/fetch.ts`: bounded output formatting.
- `packages/web/web-deepseek-api/src/search.ts`: mapSearchResponse, boundedQuestion.

Constants and the external-content notice match upstream defaults. The subset uses a search callback, source-only results and Markdown strings directly. Limits are fixed to the current tool contract: four queries, eight sources and 200,000 fetch-output characters. URLs are validated by Pi before formatting. No generic provider context, summary-answer branch, HTML converter, configurable limits or provider-truncation flag is included. Transport status and truncation metadata are produced at the Pi boundary.

`../../deepseek-web.ts` adapts registration, schema validation, prompt guidance, tool result metadata, 30-second cancellation and API-key bootstrap to Pi. API field projection, query batching and output formatting follow upstream. As upstream does, it sends one request per unique query, no automatic retries; a failed query cancels and discards the batch. The tool call budget counts batches, each containing up to four API requests.

Project-specific differences are intentional: the API's 128-Unicode-character query limit is enforced by the tool schema and runtime validation; network domain filtering and forbidden benchmark queries remain enforced; optional original-question context is omitted when forbidden by policy; transport errors do not echo provider bodies or secrets. The latest user message supplies ori_question (whitespace collapsed and bounded to 128 Unicode characters). Pi's policy layer may append a separate tool-budget notice after the bounded fetch result.

The Pi boundary additionally provides credential-shared bounded scheduling, structured error/timeout classification and local request metrics (`web-runtime.ts`). Context-pressure metering and lossless evidence archiving live outside these vendored functions (`context-meter.ts`, `context-pressure.ts`, `web-evidence.ts`). Those are Pi adaptations inspired by the upstream agent-loop, timeout-policy, token-meter and tool-result-pruner designs; they are not verbatim upstream implementations. Web evidence uses section-indexed recall receipts rather than upstream's blind head/tail pruning.

Upstream's API provider supports a broader account/configuration framework. Importing the plugin would pull that framework into Pi; this source subset retains its relevant algorithms without a second agent runtime. When updating, compare these functions against a newly pinned upstream revision and run the behavioral tests on the remote test host.
